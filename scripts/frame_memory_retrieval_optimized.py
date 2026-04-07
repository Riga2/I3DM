import torch
import math
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from einops import rearrange
from torch.nn import functional as F
from torch.amp import autocast
import os
from torchvision.utils import save_image
import time

def generate_points_in_sphere(n_points: int, radius: float, device: torch.device = None) -> torch.Tensor:
    """
    在球体内均匀生成随机点。
    
    Args:
        n_points: 采样点数量
        radius: 球体半径
        device: 设备
    
    Returns:
        (n_points, 3) 采样点坐标
    """
    if device is None:
        device = torch.device('cpu')
    
    samples_r = torch.rand(n_points, device=device)
    samples_phi = torch.rand(n_points, device=device)
    samples_u = torch.rand(n_points, device=device)
    
    r = radius * torch.pow(samples_r, 1/3)
    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)
    
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    
    points = torch.stack((x, y, z), dim=1)
    return points


def project_points_to_camera(points_w: torch.Tensor, 
                              c2w: torch.Tensor, 
                              fxfycxcy: torch.Tensor,
                              H: int = 352, 
                              W: int = 640) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将世界坐标系下的点投影到相机图像平面。
    
    Args:
        points_w: (N_P, 3) 世界坐标点
        c2w: (T, 4, 4) 相机到世界变换矩阵
        fxfycxcy: (T, 4) 相机内参 [fx, fy, cx, cy]
        H: 图像高度
        W: 图像宽度
    
    Returns:
        uv: (N_P, T, 2) 投影后的像素坐标 (u, v)
        in_fov: (N_P, T) 布尔张量，表示点是否在视野内
    """
    N_P = points_w.shape[0]
    T = c2w.shape[0]
    device = points_w.device
    
    w2c = torch.inverse(c2w)  # (T, 4, 4)
    
    points_homo = torch.cat([points_w, torch.ones(N_P, 1, device=device)], dim=1)  # (N_P, 4)
    
    # (T, 4, 4) @ (4, N_P) -> (T, 4, N_P)
    points_cam = w2c @ points_homo.T  # (T, 4, N_P)
    points_cam = points_cam[:, :3, :].permute(2, 0, 1)  # (N_P, T, 3)
    
    Z_c = points_cam[..., 2]  # (N_P, T)
    
    fx = fxfycxcy[:, 0]  # (T,)
    fy = fxfycxcy[:, 1]
    cx = fxfycxcy[:, 2]
    cy = fxfycxcy[:, 3]
    
    x_norm = points_cam[..., 0] / (Z_c + 1e-8)  # (N_P, T)
    y_norm = points_cam[..., 1] / (Z_c + 1e-8)
    
    u = fx.unsqueeze(0) * x_norm + cx.unsqueeze(0)  # (N_P, T)
    v = fy.unsqueeze(0) * y_norm + cy.unsqueeze(0)
    
    uv = torch.stack([u, v], dim=-1)  # (N_P, T, 2)
    
    in_front = Z_c > 0
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    in_fov = in_front & in_bounds  # (N_P, T)
    
    return uv, in_fov

class KeyFrameMemoryBankLearnedOccRetrieval:
    def __init__(self,
                 n_sample_points: int = 10000,
                 sample_radius: float = 30.0,
                 image_size: Tuple[int, int] = (352, 640),
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = torch.device('cuda'),
                 ):
        self.n_sample_points = n_sample_points
        self.sample_radius = sample_radius
        self.H, self.W = image_size
        self.device = device
        self.dtype = dtype      
        self.target_size = (256, 256)
        
        self.points_per_cam_cached = max(self.n_sample_points // 20, 200)
        self.cached_points_local = generate_points_in_sphere(
            self.points_per_cam_cached, 
            self.sample_radius, 
            self.device
        )
        
        self.c2w_list = torch.empty((0, 4, 4), device=device)
        self.fxfycxcy_list = torch.empty((0, 4), device=device)
        self.rgb_list = torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=device)
        self.rgb_list_processed = torch.empty((0, 3, self.target_size[0], self.target_size[1]), dtype=torch.float32, device=device)
        self.fxfycxcy_list_processed = torch.empty((0, 4), device=device)
        self.frames_num = 0

        scene_decoder_only_ckpt = 'trained_models/LVSM/ckpt_0000000000016000.pt'
        from base_diffsynth.models.wan_scene_decoder_retrieval_occ import SceneDecoderOnlyOccAnalysis
        self.nvs_model = SceneDecoderOnlyOccAnalysis(load_path=scene_decoder_only_ckpt, exit_layer=6).to(device='cuda', dtype=torch.bfloat16).eval()
    def reset(self):
        self.c2w_list = torch.empty((0, 4, 4), device=self.device)
        self.fxfycxcy_list = torch.empty((0, 4), device=self.device)
        self.rgb_list = torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=self.device)
        self.rgb_list_processed = torch.empty((0, 3, self.target_size[0], self.target_size[1]), dtype=torch.float32, device=self.device)
        self.fxfycxcy_list_processed = torch.empty((0, 4), device=self.device)
        self.frames_num = 0

    def preprocess_poses(
            self,
            in_c2ws: torch.Tensor,
            scene_scale_factor=1.35,
        ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def preprocess_batch_poses(self, batch_c2ws: torch.Tensor, scene_scale_factor: float = 1.35) -> torch.Tensor:
        """
        Batch preprocess poses.
        Args:
            batch_c2ws: [B, N, 4, 4]
        Returns:
            batch_c2ws_processed: [B, N, 4, 4]
        """
        B = batch_c2ws.shape[0]
        
        # center: [B, 3] -> mean of translation
        centers = batch_c2ws[:, :, :3, 3].mean(dim=1) 
        
        # forward: z axis
        avg_forwards = F.normalize(batch_c2ws[:, :, :3, 2].mean(dim=1), dim=-1) # [B, 3]
        
        # down: y axis
        avg_downs_temp = batch_c2ws[:, :, :3, 1].mean(dim=1) # [B, 3]
        
        # right = cross(down, forward)
        avg_rights = F.normalize(torch.cross(avg_downs_temp, avg_forwards, dim=-1), dim=-1) # [B, 3]
        
        # correct down = cross(forward, right)
        avg_downs = F.normalize(torch.cross(avg_forwards, avg_rights, dim=-1), dim=-1) # [B, 3]
        
        # Construct avg_pose [B, 4, 4]
        avg_poses = torch.eye(4, device=self.device, dtype=batch_c2ws.dtype).unsqueeze(0).repeat(B, 1, 1)
        avg_poses[:, :3, 0] = avg_rights
        avg_poses[:, :3, 1] = avg_downs
        avg_poses[:, :3, 2] = avg_forwards
        avg_poses[:, :3, 3] = centers
        
        # Invert avg_poses
        avg_poses_inv = torch.linalg.inv(avg_poses) # [B, 4, 4]
        
        # Apply transform
        # batch_c2ws: [B, N, 4, 4]
        # avg_poses_inv: [B, 4, 4] -> [B, 1, 4, 4]
        batch_c2ws_processed = avg_poses_inv.unsqueeze(1) @ batch_c2ws
        
        # Rescale
        # scene_scale = max over N points for each batch
        scene_scales = torch.max(torch.abs(batch_c2ws_processed[:, :, :3, 3]), dim=1)[0].max(dim=-1)[0] # [B]
        # scale factor
        scene_scales = scene_scale_factor * scene_scales
        
        batch_c2ws_processed[:, :, :3, 3] /= scene_scales.view(B, 1, 1)
        
        return batch_c2ws_processed

    def preprocess_intrinsics_only(self, fxfycxcy, src_H, src_W, target_H=256, target_W=256):
        """
        Preprocess intrinsics only without image data.
        """
        # Calculate resize scale to cover the target size while maintaining aspect ratio
        scale = max(target_H / src_H, target_W / src_W)
        resized_H = int(round(src_H * scale))
        resized_W = int(round(src_W * scale))

        # Calculate crop parameters (Center Crop)
        start_h = (resized_H - target_H) // 2
        start_w = (resized_W - target_W) // 2

        # Adjust intrinsics
        scale_H = resized_H / src_H
        scale_W = resized_W / src_W

        fxfycxcy_processed = fxfycxcy.clone()
        fxfycxcy_processed[:, 0] *= scale_W  # fx
        fxfycxcy_processed[:, 1] *= scale_H  # fy
        fxfycxcy_processed[:, 2] *= scale_W  # cx
        fxfycxcy_processed[:, 3] *= scale_H  # cy
        
        fxfycxcy_processed[:, 2] -= start_w  # cx offset
        fxfycxcy_processed[:, 3] -= start_h  # cy offset

        return fxfycxcy_processed
    
    def preprocess_images_fxfycxcy(self, rgbs, fxfycxcy, target_H=256, target_W=256):
        """
        Preprocess images and intrinsics to the target size.
        """
        _, _, H, W = rgbs.shape
        
        # Calculate resize scale to cover the target size while maintaining aspect ratio
        scale = max(target_H / H, target_W / W)
        resized_H = int(round(H * scale))
        resized_W = int(round(W * scale))

        # Resize images
        rgbs_resized = F.interpolate(rgbs, size=(resized_H, resized_W), mode='bilinear', align_corners=False)

        # Calculate crop parameters (Center Crop)
        start_h = (resized_H - target_H) // 2
        start_w = (resized_W - target_W) // 2

        # Crop images
        rgbs_cropped = rgbs_resized[:, :, start_h:start_h+target_H, start_w:start_w+target_W]

        # Adjust intrinsics
        scale_H = resized_H / H
        scale_W = resized_W / W

        fxfycxcy_processed = fxfycxcy.clone()
        fxfycxcy_processed[:, 0] *= scale_W  # fx
        fxfycxcy_processed[:, 1] *= scale_H  # fy
        fxfycxcy_processed[:, 2] *= scale_W  # cx
        fxfycxcy_processed[:, 3] *= scale_H  # cy
        
        fxfycxcy_processed[:, 2] -= start_w  # cx offset
        fxfycxcy_processed[:, 3] -= start_h  # cy offset

        return rgbs_cropped, fxfycxcy_processed
        
    def retrieve_camera_occ_perf_test(self,
                                   query_c2w: torch.Tensor,
                                   query_fxfycxcy: torch.Tensor,
                                   k: int = 4,
                                   batch_size: int = 16) -> List[int]:
        if self.frames_num <= k:
            top_indices = list(range(self.frames_num))
        else:
            last_idx = self.frames_num - 1
            
            confidence_maps_all_views = []

            # Optimize: Utilize cached processed images and intrinsics
            all_rgbs_processed = self.rgb_list_processed
            all_fxfycxcys_processed = self.fxfycxcy_list_processed
            
            # Process query intrinsics
            # Directly preprocess intrinsics without dummy image, using known dimensions
            query_fxfycxcy_processed = self.preprocess_intrinsics_only(
                query_fxfycxcy, 
                src_H=self.H, 
                src_W=self.W, 
                target_H=self.target_size[0], 
                target_W=self.target_size[1]
            )

            candidate_indices = list(range(self.frames_num - 1))
            
            for i in range(0, len(candidate_indices), batch_size):
                # prepare_start_time = time.time()
                batch_indices = candidate_indices[i : i + batch_size]
                current_batch_size = len(batch_indices)
                
                # Prepare batch data
                # We need to construct [B, 2+V, 4, 4] for poses to preprocess them together
                
                # 1. Gather C2Ws
                # last_idx is fixed. candidate_idx varies. query is fixed.
                # Shape: [B, 2+V, 4, 4]
                # dim 1: [last_idx, candidate_idx, query_0, query_1, ... query_V]
                
                # last_c2w: [1, 4, 4]
                last_c2w = self.c2w_list[last_idx:last_idx+1]
                
                # cand_c2ws: [B, 4, 4]
                cand_c2ws = self.c2w_list[batch_indices]
                
                # query_c2ws: [V, 4, 4]
                # We need to repeat query and last for each batch item.
                
                # Stack them:
                # We want a tensor `batch_c2ws` of shape [B, 2+V, 4, 4]
                # Component 0: last_c2w (repeated) -> [B, 1, 4, 4]
                batch_last_c2w = last_c2w.unsqueeze(0).expand(current_batch_size, -1, -1, -1)
                
                # Component 1: cand_c2ws -> [B, 1, 4, 4]
                batch_cand_c2w = cand_c2ws.unsqueeze(1)
                
                # Component 2+: query_c2ws (repeated) -> [B, V, 4, 4]
                batch_query_c2w = query_c2w.unsqueeze(0).expand(current_batch_size, -1, -1, -1)
                
                batch_union_c2ws = torch.cat([batch_last_c2w, batch_cand_c2w, batch_query_c2w], dim=1) # [B, 2+V, 4, 4]
                
                # Preprocess Poses (need to modify preprocess_poses to handle [B, N, 4, 4] or flatten it)
                # Original preprocess_poses expects [N, 4, 4]. 
                # We can reshape to [B*(2+V), 4, 4] but normalization is per-set.
                # So we must modify preprocess_poses to compute mean per batch item.
                # OR, we can do it manually here to be fast and correct.
                
                # --- Manual Preprocess Poses (Vectorized) ---
                batch_union_c2ws_processed = self.preprocess_batch_poses(batch_union_c2ws, scene_scale_factor=1.35)
                # --- End Manual Preprocess Poses ---

                # Slice Inputs/Targets
                # Input C2W: first 2 frames [B, 2, 4, 4]
                batch_input_c2w_final = batch_union_c2ws_processed[:, :2]
                # Target C2W: remaining frames [B, V, 4, 4]
                batch_target_c2w_final = batch_union_c2ws_processed[:, 2:]
                
                # Prepare Images (Already processed)
                # Need to gather [last_idx, cand_idx] for each batch
                # all_rgbs_processed: [Total_Mem, 3, H, W]
                # We want [B, 2, 3, H, W]
                # indices: [B, 2]
                
                # indices for gathering
                indices_last = torch.full((current_batch_size,), last_idx, device=self.device, dtype=torch.long)
                indices_cand = torch.tensor(batch_indices, device=self.device, dtype=torch.long)
                
                # Gather Input Images
                # [B, 3, H, W]
                imgs_last = all_rgbs_processed[indices_last]
                imgs_cand = all_rgbs_processed[indices_cand]
                batch_input_images_final = torch.stack([imgs_last, imgs_cand], dim=1) # [B, 2, 3, H, W]
                
                # Prepare Intrinsics (Already processed)
                # Input Intrinsics: gather like images
                # all_fxfycxcys_processed: [Total_Mem, 4]
                fx_last = all_fxfycxcys_processed[indices_last]
                fx_cand = all_fxfycxcys_processed[indices_cand]
                batch_input_fx_final = torch.stack([fx_last, fx_cand], dim=1) # [B, 2, 4]
                
                # Target Intrinsics: query intrinsics (processed)
                # query_fxfycxcy_processed: [V, 4]
                # Repeated for batch: [B, V, 4]
                batch_target_fx_final = query_fxfycxcy_processed.unsqueeze(0).expand(current_batch_size, -1, -1)

                data_input_batch = {
                    "image": batch_input_images_final.to(dtype=self.dtype),
                    "c2w": batch_input_c2w_final.to(dtype=self.dtype),
                    "fxfycxcy": batch_input_fx_final.to(dtype=self.dtype),
                }

                data_target_batch = {
                    "c2w": batch_target_c2w_final.to(dtype=self.dtype),
                    "fxfycxcy": batch_target_fx_final.to(dtype=self.dtype),
                }

                torch.cuda.synchronize()
                # nvs_prepare_time = time.time() - prepare_start_time

                nvs_start_time = time.time()
                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        conf_map = self.nvs_model.calc_conf_batch(data_input_batch, data_target_batch).detach().float()
                
                        # [B, V, H, W]
                        if conf_map.dim() == 5:
                            conf_map = conf_map.squeeze(2)
                        elif conf_map.dim() == 3:
                             conf_map = conf_map.unsqueeze(1)
                # torch.cuda.synchronize()
                # nvs_end_time = time.time() - nvs_start_time
                # print(f"NVS prepare time: {nvs_prepare_time:.4f} seconds, NVS inference time for batch {i//batch_size}: {nvs_end_time:.4f} seconds")

                for b in range(current_batch_size):
                    confidence_maps_all_views.append(conf_map[b])


            # select_start_time = time.time()
            num_to_select = min(k - 1, len(confidence_maps_all_views))
            
            selected_candidate_indices = []
            candidate_indices = list(range(len(confidence_maps_all_views)))
            
            if len(confidence_maps_all_views) > 0:
                V, H, W = confidence_maps_all_views[0].shape
                device = confidence_maps_all_views[0].device
            else:
                return [last_idx], []

            current_best_coverage = torch.full((V, H, W), -float('inf')).to(device=device)
            for step in range(num_to_select):
                best_gain = 0
                best_idx = -1
                
                for idx in candidate_indices:
                    map_candidate = confidence_maps_all_views[idx]
                    potential_coverage = torch.max(current_best_coverage, map_candidate)
                    score_gain = (potential_coverage - current_best_coverage).mean().item()
                    
                    if score_gain > best_gain:
                        best_gain = score_gain
                        best_idx = idx
                
                if best_idx != -1:
                    selected_candidate_indices.append(best_idx)
                    candidate_indices.remove(best_idx)
                    current_best_coverage = torch.max(current_best_coverage, confidence_maps_all_views[best_idx])
                else:
                    break
            
            top_indices = [last_idx] + selected_candidate_indices

            # torch.cuda.synchronize()
            # select_time = time.time() - select_start_time
            # print(f"Selection time: {select_time:.4f} seconds")

        while len(top_indices) < k:
            top_indices.append(random.choice(top_indices))
        
        return top_indices

    def random_retrieve(self, k: int = 4) -> List[int]:
        # last_idx = self.frames_num - 1
        if self.frames_num <= k:
            return list(range(self.frames_num))
        else:
            candidate_indices = list(range(self.frames_num))
            random_indices = random.sample(candidate_indices, k)
            top_indices = random_indices
            return top_indices

    def latest_retrieve(self, k: int = 4) -> List[int]:
        return list(range(self.frames_num))[-k:]
            
    def add_key_frames(self,
                        frames_c2w: torch.Tensor,                   # (F, 4, 4)
                        frames_fxfycxcy: torch.Tensor,              # (F, 4)
                        frames_rgb: Optional[torch.Tensor] = None   # (F, 3, H, W)
                        ):

        self.c2w_list = torch.cat([self.c2w_list, frames_c2w.to(device=self.device)], dim=0)
        self.fxfycxcy_list = torch.cat([self.fxfycxcy_list, frames_fxfycxcy.to(device=self.device)], dim=0)
        self.rgb_list = torch.cat([self.rgb_list, frames_rgb.to(device=self.device)], dim=0)
        
        if frames_rgb is not None:
            frames_rgb_gpu = frames_rgb.to(device=self.device)
            frames_fxfycxcy_gpu = frames_fxfycxcy.to(device=self.device)
            
            frames_rgb_processed, frames_fxfycxcy_processed = self.preprocess_images_fxfycxcy(
                frames_rgb_gpu, 
                frames_fxfycxcy_gpu, 
                target_H=self.target_size[0], 
                target_W=self.target_size[1]
            )
            
            self.rgb_list_processed = torch.cat([self.rgb_list_processed, frames_rgb_processed], dim=0)
            self.fxfycxcy_list_processed = torch.cat([self.fxfycxcy_list_processed, frames_fxfycxcy_processed], dim=0)

        self.frames_num += len(frames_c2w)
        

    def retrieve_top_k_cameras(self,
                               query_c2w: torch.Tensor,
                               query_fxfycxcy: torch.Tensor,
                               k: int = 44,
                               remove_overlap: bool = True) -> List[int]:
        """
        检索与查询相机视野重叠最大的 Top-K 个相机索引。
        
        Args:
            query_c2w: (4, 4) 查询相机的 C2W 矩阵
            query_fxfycxcy: (4,) 查询相机的内参
            k: 返回的相机数量
            remove_overlap: 是否在选择下一个相机时移除已覆盖区域
        
        Returns:
            camera_indices: 相机索引列表
        """
        if query_c2w.dim() == 2:
            query_c2w = query_c2w.unsqueeze(0)
            query_fxfycxcy = query_fxfycxcy.unsqueeze(0)

        query_centers = query_c2w[:, :3, 3] # (N, 3)
        n_query = query_centers.shape[0]
        points_per_cam = max(self.n_sample_points // n_query, 200)
        
        points_local = generate_points_in_sphere(
            points_per_cam, 
            self.sample_radius, 
            self.device
        )
        
        # (N, M, 3) -> (N*M, 3)
        points_w = (query_centers.unsqueeze(1) + points_local.unsqueeze(0)).reshape(-1, 3)
        
        _, in_fov_query = project_points_to_camera(
            points_w, 
            query_c2w,
            query_fxfycxcy,
            self.H, self.W
        )
        in_fov_query = in_fov_query.any(dim=1)  # (N_P,)
        
        _, in_fov_all = project_points_to_camera(
            points_w,
            self.c2w_list,
            self.fxfycxcy_list,
            self.H, self.W
        )  # (N_P, N)
        in_fov_all = in_fov_all.T  # (N, N_P)
        
        if not remove_overlap:
            query_fov_count = in_fov_query.sum().float()
            if query_fov_count == 0:
                return list(range(k))
            
            overlap_counts = (in_fov_all & in_fov_query.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count
            
            _, top_indices = overlap_ratios.topk(min(k, len(self.c2w_list)))
            return top_indices.tolist()
        
        remaining_query_fov = in_fov_query.clone()
        selected_indices = []
        
        for _ in range(min(k, len(self.c2w_list))):
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count == 0:
                break
            
            overlap_counts = (in_fov_all & remaining_query_fov.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count
            
            for idx in selected_indices:
                overlap_ratios[idx] = -1.0
            
            best_idx = overlap_ratios.argmax().item()
            
            # if best_score <= 0:
            #     break

            selected_indices.append(best_idx)
            
            remaining_query_fov = remaining_query_fov & ~in_fov_all[best_idx]
        
        if len(selected_indices) < k:
            if len(selected_indices) > 0:
                min_idx = min(selected_indices)
                max_idx = max(selected_indices)
                candidates = [i for i in range(min_idx, max_idx + 1) if i not in selected_indices]
                num_needed = k - len(selected_indices)
                
                if len(candidates) < num_needed:
                    candidates = [i for i in range(len(self.c2w_list)) if i not in selected_indices]
                
                if len(candidates) > 0:
                    pick_count = min(len(candidates), num_needed)
                    picked = np.random.choice(candidates, pick_count, replace=False)
                    for p in picked:
                        selected_indices.append(int(p))
            
            for idx in range(len(self.c2w_list)):
                if len(selected_indices) >= k:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)

        return selected_indices

    def retrieve_top_k_cameras_fov_optimized(self,
                               query_c2w: torch.Tensor,
                               query_fxfycxcy: torch.Tensor,
                               k: int = 44,
                               remove_overlap: bool = True) -> List[int]:
        """
        检索与查询相机视野重叠最大的 Top-K 个相机索引。
        
        Args:
            query_c2w: (4, 4) 查询相机的 C2W 矩阵
            query_fxfycxcy: (4,) 查询相机的内参
            k: 返回的相机数量
            remove_overlap: 是否在选择下一个相机时移除已覆盖区域
        
        Returns:
            camera_indices: 相机索引列表
        """
        if self.frames_num <= k:
            selected_indices = list(range(self.frames_num))
        else:
            if query_c2w.dim() == 2:
                query_c2w = query_c2w.unsqueeze(0)
                query_fxfycxcy = query_fxfycxcy.unsqueeze(0)

            query_centers = query_c2w[:, :3, 3] # (N, 3)
            # n_query = query_centers.shape[0]
            
            points_local = self.cached_points_local
            
            # (N, M, 3) -> (N*M, 3)
            points_w = (query_centers.unsqueeze(1) + points_local.unsqueeze(0)).reshape(-1, 3)
            
            _, in_fov_query = project_points_to_camera(
                points_w, 
                query_c2w,
                query_fxfycxcy,
                self.H, self.W
            )
            in_fov_query = in_fov_query.any(dim=1)  # (N_P,)
            
            query_fov_count_total = in_fov_query.sum().item()
            
            if query_fov_count_total == 0:
                if not remove_overlap:
                    return list(range(k))
                selected_indices = []
            else:
                points_w_in_query = points_w[in_fov_query]
                
                _, in_fov_all = project_points_to_camera(
                    points_w_in_query,
                    self.c2w_list,
                    self.fxfycxcy_list,
                    self.H, self.W
                )  # (N_P_q, N)
                in_fov_all = in_fov_all.T  # (N, N_P_q)
                
                if not remove_overlap:
                    overlap_counts = in_fov_all.sum(dim=1).float()
                    overlap_ratios = overlap_counts / float(query_fov_count_total)
                    
                    _, top_indices = overlap_ratios.topk(min(k, len(self.c2w_list)))
                    return top_indices.tolist()
                
                remaining_query_fov = torch.ones(query_fov_count_total, dtype=torch.bool, device=self.device)
                selected_indices = []
                
                valid_candidates = torch.ones(len(self.c2w_list), dtype=torch.bool, device=self.device)
                
                for _ in range(min(k, len(self.c2w_list))):
                    query_fov_count = remaining_query_fov.sum().item()
                    if query_fov_count == 0:
                        break
                    
                    overlap_counts = (in_fov_all & remaining_query_fov).sum(dim=1)
                    
                    overlap_counts[~valid_candidates] = -1
                    
                    best_idx = overlap_counts.argmax().item()
                    best_count = overlap_counts[best_idx].item()
                    
                    # best_score = float(best_count) / float(query_fov_count)

                    selected_indices.append(best_idx)
                    valid_candidates[best_idx] = False
                    
                    if best_count > 0:
                        remaining_query_fov &= ~in_fov_all[best_idx]
        
        if len(selected_indices) < k:
            if len(selected_indices) > 0:
                min_idx = min(selected_indices)
                max_idx = max(selected_indices)
                candidates = [i for i in range(min_idx, max_idx + 1) if i not in selected_indices]
                num_needed = k - len(selected_indices)
                
                if len(candidates) < num_needed:
                    candidates = [i for i in range(len(self.c2w_list)) if i not in selected_indices]
                
                if len(candidates) > 0:
                    pick_count = min(len(candidates), num_needed)
                    picked = np.random.choice(candidates, pick_count, replace=False)
                    for p in picked:
                        selected_indices.append(int(p))
            
            for idx in range(len(self.c2w_list)):
                if len(selected_indices) >= k:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)

        return selected_indices
    
    def get_retrieved_data_frames(self, frame_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        根据相机索引列表获取对应的帧数据。
        
        Args:
            frame_indices: 相机索引列表
        
        Returns:
            data_dict: 包含具体数据的字典
        """
        if not frame_indices:
            return {}
        
        indices_tensor = torch.tensor(frame_indices, device=self.device, dtype=torch.long)
        
        retrieved_c2ws = self.c2w_list[indices_tensor]
        retrieved_fxfycxcy = self.fxfycxcy_list[indices_tensor]
        retrieved_rgb = self.rgb_list[indices_tensor]
        
        return {
            'frames_c2ws': retrieved_c2ws,           # (K, 4, 4)
            'frames_fxfycxcy': retrieved_fxfycxcy,  # (K, 4)
            'frames_rgb': retrieved_rgb,             # (K, 3, H, W)
        }
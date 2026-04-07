# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

"""
基于视野重叠的相机检索模块。
通过在空间中采样点并投影到各相机视野中，计算视野重叠率来检索最相关的相机组。
"""

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


class KeyFrameMemoryBank:
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
        
        self.c2w_list = torch.empty((0, 4, 4), device=device)
        self.fxfycxcy_list = torch.empty((0, 4), device=device)
        self.rgb_list = torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=device)
        self.frames_num = 0
    
    def add_key_frames(self,
                        frames_c2w: torch.Tensor,                   # (F, 4, 4)
                        frames_fxfycxcy: torch.Tensor,              # (F, 4)
                        frames_rgb: Optional[torch.Tensor] = None   # (F, 3, H, W)
                        ):

        self.c2w_list = torch.cat([self.c2w_list, frames_c2w.to(device=self.device)], dim=0)
        self.fxfycxcy_list = torch.cat([self.fxfycxcy_list, frames_fxfycxcy.to(device=self.device)], dim=0)
        self.rgb_list = torch.cat([self.rgb_list, frames_rgb.to(device=self.device)], dim=0)

        self.frames_num += len(frames_c2w)
        
    def retrieve_top_k_cameras_keep_last(self,
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
        selected_ratios = []

        if len(self.c2w_list) > 0:
            last_idx = len(self.c2w_list) - 1
            
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count > 0:
                overlap_count = (in_fov_all[last_idx] & remaining_query_fov).sum().float()
                ratio = (overlap_count / query_fov_count).item()
            else:
                ratio = 0.0
                
            selected_indices.append(last_idx)
            selected_ratios.append(ratio)
            
            remaining_query_fov = remaining_query_fov & ~in_fov_all[last_idx]
        
        num_needed = k - len(selected_indices)
        for _ in range(min(num_needed, len(self.c2w_list))):
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count == 0:
                break
            
            overlap_counts = (in_fov_all & remaining_query_fov.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count
            
            for idx in selected_indices:
                overlap_ratios[idx] = -1.0
            
            best_idx = overlap_ratios.argmax().item()
            best_score = overlap_ratios[best_idx].item()
            
            # if best_score <= 0:
            #     break

            selected_indices.append(best_idx)
            selected_ratios.append(best_score)
            
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
                        selected_ratios.append(0.0)
            
            for idx in range(len(self.c2w_list)):
                if len(selected_indices) >= k:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    selected_ratios.append(0.0)

        return selected_indices, selected_ratios
    
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
        selected_ratios = []
        
        for _ in range(min(k, len(self.c2w_list))):
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count == 0:
                break
            
            overlap_counts = (in_fov_all & remaining_query_fov.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count
            
            for idx in selected_indices:
                overlap_ratios[idx] = -1.0
            
            best_idx = overlap_ratios.argmax().item()
            best_score = overlap_ratios[best_idx].item()
            
            # if best_score <= 0:
            #     break

            selected_indices.append(best_idx)
            selected_ratios.append(best_score)
            
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
                        selected_ratios.append(0.0)
            
            for idx in range(len(self.c2w_list)):
                if len(selected_indices) >= k:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    selected_ratios.append(0.0)

        return selected_indices, selected_ratios
    
    def calculate_query_coverage(self,
                               query_c2w: torch.Tensor,
                               query_fxfycxcy: torch.Tensor,
                               selected_mem_indices: List[int],
                               ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        计算每个查询相机的视野被当前记忆库覆盖的比例。
        
        Args:
            query_c2w: (N, 4, 4) 查询相机的 C2W 矩阵
            query_fxfycxcy: (N, 4) 查询相机的内参
            selected_mem_indices: 选中的记忆库索引列表
        
        Returns:
            total_coverage: (N,) 每个查询相机的总覆盖率 [0, 1]
            max_single_coverage: (N,) 每个查询相机被单个记忆帧覆盖的最大比例
            best_mem_indices: (N,) 对应的最佳记忆帧索引
        """
        if query_c2w.dim() == 2:
            query_c2w = query_c2w.unsqueeze(0)
            query_fxfycxcy = query_fxfycxcy.unsqueeze(0)

        if len(selected_mem_indices) == 0:
            N = query_c2w.shape[0]
            return (torch.zeros(N, device=self.device), 
                    torch.zeros(N, device=self.device), 
                    [-1] * N)
            
        selected_c2w = self.c2w_list[selected_mem_indices]
        selected_fxfycxcy = self.fxfycxcy_list[selected_mem_indices]
        
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
        ) # (N*M, N)
        
        _, in_fov_all = project_points_to_camera(
            points_w,
            selected_c2w,
            selected_fxfycxcy,
            self.H, self.W
        ) # (N*M, K)
        
        is_covered_by_bank = in_fov_all.any(dim=1) # (N*M,)
        
        ratios = []
        max_single_ratios = []
        best_indices = []
        
        for i in range(n_query):
            start_idx = i * points_per_cam
            end_idx = (i + 1) * points_per_cam
            
            valid_mask = in_fov_query[start_idx:end_idx, i] # (M,)
            
            if valid_mask.sum() == 0:
                ratios.append(0.0)
                max_single_ratios.append(0.0)
                best_indices.append(-1)
                continue
                
            covered_mask = is_covered_by_bank[start_idx:end_idx] # (M,)
            covered_valid = valid_mask & covered_mask
            ratio = covered_valid.sum().float() / valid_mask.sum().float()
            ratios.append(ratio.item())
            
            # in_fov_all slice: (M, K)
            mem_fovs = in_fov_all[start_idx:end_idx]
            
            # (M, K) & (M, 1) -> (M, K) -> sum(0) -> (K,)
            intersections = (mem_fovs & valid_mask.unsqueeze(1)).sum(dim=0).float()
            single_ratios = intersections / valid_mask.sum().float()
            
            max_val, max_idx = single_ratios.max(dim=0)
            max_single_ratios.append(max_val.item())
            best_indices.append(selected_mem_indices[max_idx.item()])
            
        return (torch.tensor(ratios, device=self.device), 
                torch.tensor(max_single_ratios, device=self.device), 
                best_indices)

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
        
        self.c2w_list = torch.empty((0, 4, 4), device=device)
        self.fxfycxcy_list = torch.empty((0, 4), device=device)
        self.rgb_list = torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=device)
        self.frames_num = 0

        scene_decoder_only_ckpt = 'trained_models/LVSM/ckpt_0000000000016000.pt'
        from base_diffsynth.models.wan_scene_decoder_retrieval_occ import SceneDecoderOnlyOccAnalysis
        self.nvs_model = SceneDecoderOnlyOccAnalysis(load_path=scene_decoder_only_ckpt, exit_layer=6).to(device='cuda', dtype=torch.bfloat16).eval()

    def reset(self):
        self.c2w_list = torch.empty((0, 4, 4), device=self.device)
        self.fxfycxcy_list = torch.empty((0, 4), device=self.device)
        self.rgb_list = torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=self.device)
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
        
    def retrieve_top_k_cameras_learned(self,
                               query_c2w: torch.Tensor,
                               query_fxfycxcy: torch.Tensor,
                               k: int = 4,
                               gt_rgbs: Optional[torch.Tensor] = None,
                               debug_dir: Optional[str] = None,
                               remove_overlap: bool = True) -> List[int]:
        last_idx = self.frames_num - 1

        confs = []
        psnrs = []

        # concat_c2ws = torch.cat([self.c2w_list, query_c2w], dim=0)
        # concat_c2ws_processed = self.preprocess_poses(concat_c2ws)
        # concat_fxfycxcys = torch.cat([self.fxfycxcy_list, query_fxfycxcy], dim=0)
        # if gt_rgbs is not None:
        #     concat_rgbs = torch.cat([self.rgb_list, gt_rgbs], dim=0)
        # else:
        #     concat_rgbs = self.rgb_list
        # concat_rgbs_processed, concat_fxfycxcys_processed = self.preprocess_images_fxfycxcy(concat_rgbs, concat_fxfycxcys, target_H=256, target_W=256)

        for i in range(self.frames_num - 1):
            candidate_c2ws = torch.cat([self.c2w_list[last_idx:last_idx+1], self.c2w_list[i:i+1]], dim=0)
            candidate_fxfycxcys = torch.cat([self.fxfycxcy_list[last_idx:last_idx+1], self.fxfycxcy_list[i:i+1]], dim=0)
            candidate_rgbs = torch.cat([self.rgb_list[last_idx:last_idx+1], self.rgb_list[i:i+1]], dim=0)

            union_c2ws = torch.cat([candidate_c2ws, query_c2w], dim=0)
            union_c2ws_processed = self.preprocess_poses(union_c2ws)

            union_fxfycxcys = torch.cat([candidate_fxfycxcys, query_fxfycxcy], dim=0)
            if gt_rgbs is not None:
                union_rgbs = torch.cat([candidate_rgbs, gt_rgbs], dim=0)
            else:
                union_rgbs = candidate_rgbs
            union_rgbs_processed, union_fxfycxcys_processed = self.preprocess_images_fxfycxcy(union_rgbs, union_fxfycxcys, target_H=256, target_W=256)

            data_input_batch = {
                "image": union_rgbs_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                "c2w": union_c2ws_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                "fxfycxcy": union_fxfycxcys_processed[:2][None].to('cuda', dtype=torch.bfloat16),
            }

            data_target_batch = {
                "c2w": union_c2ws_processed[2:][None].to('cuda', dtype=torch.bfloat16),
                "fxfycxcy": union_fxfycxcys_processed[2:][None].to('cuda', dtype=torch.bfloat16),
            }
            # data_input_batch = {
            #     "image": concat_rgbs_processed[[last_idx, i]][None].to('cuda', dtype=torch.bfloat16),
            #     "c2w": concat_c2ws_processed[[last_idx, i]][None].to('cuda', dtype=torch.bfloat16),
            #     "fxfycxcy": concat_fxfycxcys_processed[[last_idx, i]][None].to('cuda', dtype=torch.bfloat16),
            # }

            # data_target_batch = {
            #     "c2w": concat_c2ws_processed[self.frames_num:][None].to('cuda', dtype=torch.bfloat16),
            #     "fxfycxcy": concat_fxfycxcys_processed[self.frames_num:][None].to('cuda', dtype=torch.bfloat16),
            # }

            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Optimization: Process only 1 target frame at a time to reduce peak memory
                    nvs_ret = self.nvs_model(data_input_batch, data_target_batch)
                    conf_map = nvs_ret['conf_log_var'] # v_target, 1, H, W

            if debug_dir is not None:
                save_results_dir = f'{debug_dir}/retrieval_debug'
                os.makedirs(f"{save_results_dir}/debug_clip_{i}", exist_ok=True)
                save_image(union_rgbs_processed[:2], f"{save_results_dir}/debug_clip_{i}/nvs_input_debug.png")
                save_image(nvs_ret["rendered_images"][0], f"{save_results_dir}/debug_clip_{i}/nvs_res_debug.png")
                save_image(union_rgbs_processed[2:], f"{save_results_dir}/debug_clip_{i}/nvs_gt_debug.png")

            # save_image(concat_rgbs_processed[[last_idx, i]], f"{save_results_dir}/debug_clip_{i}/nvs_input_debug.png")
            # save_image(concat_rgbs_processed[self.frames_num:], f"{save_results_dir}/debug_clip_{i}/nvs_gt_debug.png")

            # calc psnr between nvs rendered and gt
            mse = F.mse_loss(nvs_ret["rendered_images"][0], union_rgbs_processed[2:])
            psnr = -10.0 * torch.log10(mse + 1e-10)
            psnrs.append(psnr.item())
            if debug_dir is not None:
                with open(f"{save_results_dir}/debug_clip_{i}/nvs_psnr_debug.txt", "w") as f:
                    f.write(f"PSNR: {psnr.item()}\n")

            avg_conf = conf_map.mean()
            confs.append(avg_conf.cpu())

            if debug_dir is not None:
                with open(f"{save_results_dir}/debug_clip_{i}/nvs_conf_debug.txt", "w") as f:
                    f.write(f"Average confidence: {avg_conf.cpu().item()}\n")

            print(f"Done retrieval analysis for memory frame {i} / {self.frames_num - 1}")

        if debug_dir is not None:
            with open(f"{save_results_dir}/analysis_results.txt", "w") as f:
                for r, p in zip(confs, psnrs):
                    f.write(f"{r} {p}\n")

            with open(f"{save_results_dir}/psnrs.txt", "w") as f:
                f.write('\n'.join(map(str, psnrs)))

            with open(f"{save_results_dir}/confs.txt", "w") as f:
                f.write('\n'.join(map(str, confs)))

        confs_tensor = torch.tensor(confs)
        _, top_indices = confs_tensor.topk(min(k - 1, len(confs_tensor)))
        top_indices = top_indices.tolist()
        top_indices = [last_idx] + top_indices
        return top_indices, confs_tensor.tolist()

        # psnrs_tensor = torch.tensor(psnrs)
        # _, top_indices = psnrs_tensor.topk(min(k - 1, len(psnrs_tensor)))
        # top_indices = top_indices.tolist()
        # top_indices = [last_idx] + top_indices
        # return top_indices, psnrs_tensor.tolist()
    
        # ratios_tensor = torch.tensor(ratios)
        # _, top_indices = ratios_tensor.topk(min(k - 1, len(ratios_tensor)))
        # top_indices = top_indices.tolist()
        # top_indices = [last_idx] + top_indices
        # return top_indices, ratios_tensor.tolist()


    def _calculate_adaptive_alpha(self, base_map, target_saturation_val=2.0, quantile=0.9):
        """
        根据基准图的分布计算 alpha。
        base_map: [V, H, W] 的置信度图 (固定帧)
        target_saturation_val: tanh(x) 到达饱和所需的输入值 (tanh(2.0) ~= 0.96)
        quantile: 我们希望场景中前百分之多少的像素达到饱和 (0.9 代表前 10%)
        """
        flat_vals = base_map.flatten()
        
        kth_val = torch.quantile(flat_vals, quantile).item()
        
        robust_denominator = max(kth_val, 0.1)
        
        alpha = target_saturation_val / robust_denominator
        
        print(f"[Adaptive Alpha] P{int(quantile*100)}={kth_val:.4f} => Alpha={alpha:.4f}")
        return alpha


    def retrieve_top_k_cameras_learned_topK(self,
                               query_c2w: torch.Tensor,
                               query_fxfycxcy: torch.Tensor,
                               k: int = 4,
                               gt_rgbs: Optional[torch.Tensor] = None
                               ) -> List[int]:
        last_idx = self.frames_num - 1

        confs = []
        psnrs = []

        print(f"Start evaluating {self.frames_num - 1} candidate frames...")

        for i in range(self.frames_num - 1):
            candidate_c2ws = torch.cat([self.c2w_list[last_idx:last_idx+1], self.c2w_list[i:i+1]], dim=0)
            candidate_fxfycxcys = torch.cat([self.fxfycxcy_list[last_idx:last_idx+1], self.fxfycxcy_list[i:i+1]], dim=0)
            candidate_rgbs = torch.cat([self.rgb_list[last_idx:last_idx+1], self.rgb_list[i:i+1]], dim=0)

            union_c2ws = torch.cat([candidate_c2ws, query_c2w], dim=0)
            union_c2ws_processed = self.preprocess_poses(union_c2ws)

            union_fxfycxcys = torch.cat([candidate_fxfycxcys, query_fxfycxcy], dim=0)
            if gt_rgbs is not None:
                union_rgbs = torch.cat([candidate_rgbs, gt_rgbs], dim=0)
            else:
                union_rgbs = candidate_rgbs
            union_rgbs_processed, union_fxfycxcys_processed = self.preprocess_images_fxfycxcy(union_rgbs, union_fxfycxcys, target_H=256, target_W=256)

            data_input_batch = {
                "image": union_rgbs_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                "c2w": union_c2ws_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                "fxfycxcy": union_fxfycxcys_processed[:2][None].to('cuda', dtype=torch.bfloat16),
            }

            data_target_batch = {
                "c2w": union_c2ws_processed[2:][None].to('cuda', dtype=torch.bfloat16),
                "fxfycxcy": union_fxfycxcys_processed[2:][None].to('cuda', dtype=torch.bfloat16),
            }

            # with torch.no_grad():
            #     with autocast(device_type="cuda", dtype=torch.bfloat16):
            #         # Optimization: Process only 1 target frame at a time to reduce peak memory
            #         nvs_ret = self.nvs_model(data_input_batch, data_target_batch)
            #         conf_map = nvs_ret['conf_log_var'].detach().float() # v_target, 1, H, W

            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    conf_map = self.nvs_model.calc_conf(data_input_batch, data_target_batch).detach().float()
                    
                    if conf_map.dim() == 4 and conf_map.shape[1] == 1:
                        conf_map = conf_map.squeeze(1)
                    elif conf_map.dim() == 4 and conf_map.shape[0] == 1:
                         conf_map = conf_map.squeeze(0)

            avg_conf = conf_map.mean()
            confs.append(avg_conf.cpu())

            print(f"Done retrieval analysis for memory frame {i} / {self.frames_num - 1}")

        confs_tensor = torch.tensor(confs)
        _, top_indices = confs_tensor.topk(min(k - 1, len(confs_tensor)))
        top_indices = top_indices.tolist()

        top_indices = [last_idx] + top_indices

        while len(top_indices) < k:
            top_indices.append(random.choice(top_indices))

        return top_indices, confs_tensor.tolist()

    def retrieve_top_k_cameras_learned_new(self,
                                   query_c2w: torch.Tensor,
                                   query_fxfycxcy: torch.Tensor,
                                   k: int = 4,
                                   gt_rgbs: Optional[torch.Tensor] = None,
                                   debug_dir: Optional[str] = None,
                                   remove_overlap: bool = True) -> List[int]:
        """
        利用 Confidence Map 进行最大化覆盖（Maximal Coverage）筛选。
        目标：选出 k 个帧，使得它们组合后在所有 Query 视角下的联合置信度最高。
        """
        last_idx = self.frames_num - 1
        
        confs = []
        psnrs = []
        confidence_maps_all_views = []
        
        print(f"Start evaluating {self.frames_num - 1} candidate frames...")

        # ==========================================
        # ==========================================
        for i in range(self.frames_num - 1):
            candidate_c2ws = torch.cat([self.c2w_list[last_idx:last_idx+1], self.c2w_list[i:i+1]], dim=0)
            candidate_fxfycxcys = torch.cat([self.fxfycxcy_list[last_idx:last_idx+1], self.fxfycxcy_list[i:i+1]], dim=0)
            candidate_rgbs = torch.cat([self.rgb_list[last_idx:last_idx+1], self.rgb_list[i:i+1]], dim=0)

            union_c2ws = torch.cat([candidate_c2ws, query_c2w], dim=0)
            union_c2ws_processed = self.preprocess_poses(union_c2ws)

            union_fxfycxcys = torch.cat([candidate_fxfycxcys, query_fxfycxcy], dim=0)
            
            if gt_rgbs is not None:
                union_rgbs = torch.cat([candidate_rgbs, gt_rgbs], dim=0)
            else:
                union_rgbs = candidate_rgbs 
            
            union_rgbs_processed, union_fxfycxcys_processed = self.preprocess_images_fxfycxcy(
                union_rgbs, union_fxfycxcys, target_H=256, target_W=256
            )

            data_input_batch = {
                "image": union_rgbs_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                "c2w": union_c2ws_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                "fxfycxcy": union_fxfycxcys_processed[:2][None].to('cuda', dtype=torch.bfloat16),
            }

            data_target_batch = {
                "c2w": union_c2ws_processed[2:][None].to('cuda', dtype=torch.bfloat16),
                "fxfycxcy": union_fxfycxcys_processed[2:][None].to('cuda', dtype=torch.bfloat16),
            }

            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    nvs_ret = self.nvs_model(data_input_batch, data_target_batch)
                    
                    conf_map = nvs_ret['conf_log_var'].detach().float()
                    
                    if conf_map.dim() == 4 and conf_map.shape[1] == 1:
                        conf_map = conf_map.squeeze(1)
                    elif conf_map.dim() == 4 and conf_map.shape[0] == 1:
                         conf_map = conf_map.squeeze(0)

            confidence_maps_all_views.append(conf_map)

            avg_conf = conf_map.mean().item()
            confs.append(avg_conf)

            if debug_dir:
                if len(union_rgbs_processed) > 2:
                    mse = F.mse_loss(nvs_ret["rendered_images"][0], union_rgbs_processed[2:])
                    psnr = -10.0 * torch.log10(mse + 1e-10)
                    psnrs.append(psnr.item())
                else:
                    psnrs.append(0.0)
            
            print(f"Analyzed candidate {i} / {self.frames_num - 2}")

        if debug_dir:
            save_results_dir = f'{debug_dir}/retrieval_debug'
            os.makedirs(save_results_dir, exist_ok=True)
            with open(f"{save_results_dir}/analysis_results.txt", "w") as f:
                for r, p in zip(confs, psnrs):
                    f.write(f"{r} {p}\n")


        # ==========================================
        # ==========================================
        
        num_to_select = min(k - 1, len(confidence_maps_all_views))
        
        selected_candidate_indices = []
        candidate_indices = list(range(len(confidence_maps_all_views)))
        
        if len(confidence_maps_all_views) > 0:
            V, H, W = confidence_maps_all_views[0].shape
            device = confidence_maps_all_views[0].device
        else:
            return [last_idx], []

        # sample_maps = torch.stack(random.sample(confidence_maps_all_views, min(10, len(confidence_maps_all_views)))) # [10, V, H, W]
        # sample_maps = torch.stack(confidence_maps_all_views) # [N, V, H, W]
        # adaptive_alpha = self._calculate_adaptive_alpha(sample_maps, quantile=1.0)
        
        # ============================================================

        current_best_coverage = torch.full((V, H, W), -float('inf')).to(device=device)

        print(f"Start Multi-View Greedy Selection for {num_to_select} frames...")

        for step in range(num_to_select):
            best_gain = -float('inf')
            best_idx = -1
            
            for idx in candidate_indices:
                map_candidate = confidence_maps_all_views[idx]
                potential_coverage = torch.max(current_best_coverage, map_candidate)
                
                # utility_map = torch.tanh(potential_coverage * adaptive_alpha)
                
                score = potential_coverage.mean().item()
                # ===================================================
                
                if score > best_gain:
                    best_gain = score
                    best_idx = idx
            
            if best_idx != -1:
                selected_candidate_indices.append(best_idx)
                candidate_indices.remove(best_idx)
                current_best_coverage = torch.max(current_best_coverage, confidence_maps_all_views[best_idx])
                print(f"  [Select Step {step+1}] Picked {best_idx}, Utility: {best_gain:.4f}")
            else:
                break
        
        top_indices = [last_idx] + selected_candidate_indices
        
        confs_tensor = torch.tensor(confs)
        
        return top_indices, confs_tensor.tolist()

    def retrieve_top_k_cameras_learned_use(self,
                                   query_c2w: torch.Tensor,
                                   query_fxfycxcy: torch.Tensor,
                                   k: int = 4) -> List[int]:
        """
        利用 Confidence Map 进行最大化覆盖（Maximal Coverage）筛选。
        目标：选出 k 个帧，使得它们组合后在所有 Query 视角下的联合置信度最高。
        """

        if self.frames_num <= k:
            top_indices = list(range(self.frames_num))
        else:
            last_idx = self.frames_num - 1
            
            confs = []
            confidence_maps_all_views = []
            
            print(f"Start evaluating {self.frames_num - 1} candidate frames...")

            # ==========================================
            # ==========================================
            for i in range(self.frames_num - 1):
                candidate_c2ws = torch.cat([self.c2w_list[last_idx:last_idx+1], self.c2w_list[i:i+1]], dim=0)
                candidate_fxfycxcys = torch.cat([self.fxfycxcy_list[last_idx:last_idx+1], self.fxfycxcy_list[i:i+1]], dim=0)
                candidate_rgbs = torch.cat([self.rgb_list[last_idx:last_idx+1], self.rgb_list[i:i+1]], dim=0)

                union_c2ws = torch.cat([candidate_c2ws, query_c2w], dim=0)
                union_c2ws_processed = self.preprocess_poses(union_c2ws)

                union_fxfycxcys = torch.cat([candidate_fxfycxcys, query_fxfycxcy], dim=0)
                
                union_rgbs = candidate_rgbs 
                
                union_rgbs_processed, union_fxfycxcys_processed = self.preprocess_images_fxfycxcy(
                    union_rgbs, union_fxfycxcys, target_H=256, target_W=256
                )

                data_input_batch = {
                    "image": union_rgbs_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                    "c2w": union_c2ws_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                    "fxfycxcy": union_fxfycxcys_processed[:2][None].to('cuda', dtype=torch.bfloat16),
                }

                data_target_batch = {
                    "c2w": union_c2ws_processed[2:][None].to('cuda', dtype=torch.bfloat16),
                    "fxfycxcy": union_fxfycxcys_processed[2:][None].to('cuda', dtype=torch.bfloat16),
                }

                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        conf_map = self.nvs_model.calc_conf(data_input_batch, data_target_batch).detach().float()
                        
                        if conf_map.dim() == 4 and conf_map.shape[1] == 1:
                            conf_map = conf_map.squeeze(1)
                        elif conf_map.dim() == 4 and conf_map.shape[0] == 1:
                            conf_map = conf_map.squeeze(0)

                confidence_maps_all_views.append(conf_map)

                avg_conf = conf_map.mean().item()
                confs.append(avg_conf)

                print(f"Analyzed candidate {i} / {self.frames_num - 2}")

            # ==========================================
            # ==========================================
            
            num_to_select = min(k - 1, len(confidence_maps_all_views))
            
            selected_candidate_indices = []
            candidate_indices = list(range(len(confidence_maps_all_views)))
            
            if len(confidence_maps_all_views) > 0:
                V, H, W = confidence_maps_all_views[0].shape
                device = confidence_maps_all_views[0].device
            else:
                return [last_idx], []

            # sample_maps = torch.stack(random.sample(confidence_maps_all_views, min(10, len(confidence_maps_all_views)))) # [10, V, H, W]
            # sample_maps = torch.stack(confidence_maps_all_views) # [N, V, H, W]
            # adaptive_alpha = self._calculate_adaptive_alpha(sample_maps, quantile=1.0)
            
            # ============================================================

            current_best_coverage = torch.full((V, H, W), -float('inf')).to(device=device)

            print(f"Start Multi-View Greedy Selection for {num_to_select} frames...")

            for step in range(num_to_select):
                best_gain = -float('inf')
                best_idx = -1
                
                for idx in candidate_indices:
                    map_candidate = confidence_maps_all_views[idx]
                    potential_coverage = torch.max(current_best_coverage, map_candidate)
                    
                    # utility_map = torch.tanh(potential_coverage * adaptive_alpha)
                    
                    score = potential_coverage.mean().item()
                    # ===================================================
                    
                    if score > best_gain:
                        best_gain = score
                        best_idx = idx
                
                if best_idx != -1:
                    selected_candidate_indices.append(best_idx)
                    candidate_indices.remove(best_idx)
                    current_best_coverage = torch.max(current_best_coverage, confidence_maps_all_views[best_idx])
                    print(f"  [Select Step {step+1}] Picked {best_idx}, Utility: {best_gain:.4f}")
                else:
                    break
        
            top_indices = [last_idx] + selected_candidate_indices

        while len(top_indices) < k:
            top_indices.append(random.choice(top_indices))

        return top_indices
        
        # confs_tensor = torch.tensor(confs)
        
        # return top_indices, confs_tensor.tolist()

    def retrieve_top_k_cameras_perf_test(self,
                                   query_c2w: torch.Tensor,
                                   query_fxfycxcy: torch.Tensor,
                                   k: int = 4,
                                   batch_size: int = 12) -> List[int]:

        start_time = time.time()

        last_idx = self.frames_num - 1
        
        confidence_maps_all_views = []

        # Pre-process all images (resize & crop) once since they don't depend on poses
        # We need to process: last_idx frame, all candidate frames, query frames (if we had access to their RGBs, but query only has c2w/K).
        # Actually query RGBs are not used here, only poses.
        # So we only need to preprocess memory images.
        
        # Preprocess all memory images at once
        all_rgbs_processed, all_fxfycxcys_processed = self.preprocess_images_fxfycxcy(
            self.rgb_list, self.fxfycxcy_list, target_H=256, target_W=256
        )
        
        # Preprocess query intrinsics (assuming dummy RGBs for size calculation if needed, but actually we just need K scaling)
        # We can reuse the same scaling factor from memory images since we assume same image size or we can calculate it.
        # But wait, query might have different resolution.
        # Let's trust preprocess_images_fxfycxcy handles intrinsics correctly without RGBs if we modify it,
        # but current implementation requires RGB to get H, W.
        # Let's assume standard behavior: use existing function on query if possible or replicate logic.
        # But query RGBs are not passed! `retrieve_top_k_cameras_perf_test` takes `query_c2w` and `query_fxfycxcy`.
        # The original code did:
        #   union_rgbs = candidate_rgbs  (for input AND target)
        #   union_fxfycxcys = torch.cat([candidate_fxfycxcys, query_fxfycxcy], dim=0)
        #   union_rgbs_processed, union_fxfycxcys_processed = self.preprocess_images_fxfycxcy(...)
        # 
        # Wait, `union_rgbs` only included `candidate_rgbs` (2 frames).
        # `union_fxfycxcys` included BOTH candidate and query.
        # The function `preprocess_images_fxfycxcy` takes `rgbs, fxfycxcy`.
        # Inside it uses `rgbs.shape` to determine H, W.
        # It assumes `fxfycxcy` corresponds to `rgbs`.
        # BUT here `fxfycxcy` has MORE elements than `rgbs` (2+V vs 2).
        # If `fxfycxcy` is longer than `rgbs`, the code might crash or misbehave if it tries to iterate.
        # Let's check `preprocess_images_fxfycxcy`:
        #   _, _, H, W = rgbs.shape
        #   ...
        #   fxfycxcy_processed = fxfycxcy.clone()
        #   fxfycxcy_processed[:, 0] *= scale_W ...
        # It processes all `fxfycxcy` based on `rgbs` size (assuming all images share the same spatial dimensions H,W).
        # So we can pre-calculate the processed fxfycxcy for query provided we know the image size.
        # We can use `self.H, self.W` or the size from `self.rgb_list`.
        
        # Optimize: 
        # 1. Process all memory images/intrinsics once.
        # 2. Process query intrinsics once.
        # 3. In loop, just slice and concat poses.

        # Process query intrinsics
        # We need a dummy tensor for RGB shape to use the helper, or just manually scale.
        # Let's use the helper with slice of memory RGB to get scale factors correct.
        dummy_rgb = self.rgb_list[:1]
        _, query_fxfycxcy_processed = self.preprocess_images_fxfycxcy(
            dummy_rgb, query_fxfycxcy, target_H=256, target_W=256
        )

        candidate_indices = list(range(self.frames_num - 1))
        
        for i in range(0, len(candidate_indices), batch_size):
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
            # center: [B, 3] -> mean of translation of (last + cand + queries)
            # Actually original code:
            # center = in_c2ws[:, :3, 3].mean(0)
            # Here: in_c2ws is [B, 2+V, 4, 4]
            # center = in_c2ws[:, :, :3, 3].mean(1)  # [B, 3]
            
            centers = batch_union_c2ws[:, :, :3, 3].mean(dim=1) # [B, 3]
            
            # forward: z axis. 
            # avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
            avg_forwards = F.normalize(batch_union_c2ws[:, :, :3, 2].mean(dim=1), dim=-1) # [B, 3]
            
            # down: y axis
            avg_downs_temp = batch_union_c2ws[:, :, :3, 1].mean(dim=1) # [B, 3]
            
            # right = cross(down, forward)
            avg_rights = F.normalize(torch.cross(avg_downs_temp, avg_forwards, dim=-1), dim=-1) # [B, 3]
            
            # correct down = cross(forward, right)
            avg_downs = F.normalize(torch.cross(avg_forwards, avg_rights, dim=-1), dim=-1) # [B, 3]
            
            # Construct avg_pose [B, 4, 4]
            avg_poses = torch.eye(4, device=self.device, dtype=batch_union_c2ws.dtype).unsqueeze(0).repeat(current_batch_size, 1, 1)
            avg_poses[:, :3, 0] = avg_rights
            avg_poses[:, :3, 1] = avg_downs
            avg_poses[:, :3, 2] = avg_forwards
            avg_poses[:, :3, 3] = centers
            
            # Invert avg_poses
            avg_poses_inv = torch.linalg.inv(avg_poses) # [B, 4, 4]
            
            # Apply transform
            # batch_union_c2ws: [B, 2+V, 4, 4]
            # avg_poses_inv: [B, 4, 4] -> [B, 1, 4, 4]
            batch_union_c2ws_processed = avg_poses_inv.unsqueeze(1) @ batch_union_c2ws
            
            # Rescale
            # scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
            # Here: max over (2+V) points for each batch
            scene_scales = torch.max(torch.abs(batch_union_c2ws_processed[:, :, :3, 3]), dim=1)[0].max(dim=-1)[0] # [B]
            # scale factor
            scene_scales = 1.35 * scene_scales
            
            batch_union_c2ws_processed[:, :, :3, 3] /= scene_scales.view(current_batch_size, 1, 1)
            
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
                "image": batch_input_images_final.to('cuda', dtype=torch.bfloat16),
                "c2w": batch_input_c2w_final.to('cuda', dtype=torch.bfloat16),
                "fxfycxcy": batch_input_fx_final.to('cuda', dtype=torch.bfloat16),
            }

            data_target_batch = {
                "c2w": batch_target_c2w_final.to('cuda', dtype=torch.bfloat16),
                "fxfycxcy": batch_target_fx_final.to('cuda', dtype=torch.bfloat16),
            }

            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    conf_map = self.nvs_model.calc_conf(data_input_batch, data_target_batch).detach().float()
                    
                    if conf_map.dim() == 5: # [B, V, 1, H, W]
                         conf_map = conf_map.squeeze(2) # -> [B, V, H, W]
                    elif conf_map.dim() == 4 and conf_map.shape[1] == 1:
                         conf_map = conf_map.squeeze(1)

            for b in range(current_batch_size):
                confidence_maps_all_views.append(conf_map[b])

        # ==========================================
        # ==========================================
        
        num_to_select = min(k - 1, len(confidence_maps_all_views))
        
        selected_candidate_indices = []
        candidate_indices = list(range(len(confidence_maps_all_views)))
        
        if len(confidence_maps_all_views) > 0:
            V, H, W = confidence_maps_all_views[0].shape
            device = confidence_maps_all_views[0].device
        else:
            return [last_idx], []


        current_best_coverage = torch.full((V, H, W), -float('inf')).to(device=device)

        # print(f"Start Multi-View Greedy Selection for {num_to_select} frames...")

        for step in range(num_to_select):
            best_gain = -float('inf')
            best_idx = -1
            
            for idx in candidate_indices:
                map_candidate = confidence_maps_all_views[idx]
                potential_coverage = torch.max(current_best_coverage, map_candidate)
                score = potential_coverage.mean().item()
                
                if score > best_gain:
                    best_gain = score
                    best_idx = idx
            
            if best_idx != -1:
                selected_candidate_indices.append(best_idx)
                candidate_indices.remove(best_idx)
                current_best_coverage = torch.max(current_best_coverage, confidence_maps_all_views[best_idx])
                # print(f"  [Select Step {step+1}] Picked {best_idx}, Utility: {best_gain:.4f}")
            else:
                break
        
        top_indices = [last_idx] + selected_candidate_indices

        while len(top_indices) < k:
            top_indices.append(random.choice(top_indices))

        torch.cuda.synchronize()
        selection_end_time = time.time()
        
        return top_indices, (selection_end_time - start_time)

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

        self.frames_num += len(frames_c2w)
        
    def retrieve_top_k_cameras_keep_last(self,
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
        selected_ratios = []

        if len(self.c2w_list) > 0:
            last_idx = len(self.c2w_list) - 1
            
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count > 0:
                overlap_count = (in_fov_all[last_idx] & remaining_query_fov).sum().float()
                ratio = (overlap_count / query_fov_count).item()
            else:
                ratio = 0.0
                
            selected_indices.append(last_idx)
            selected_ratios.append(ratio)
            
            remaining_query_fov = remaining_query_fov & ~in_fov_all[last_idx]
        
        num_needed = k - len(selected_indices)
        for _ in range(min(num_needed, len(self.c2w_list))):
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count == 0:
                break
            
            overlap_counts = (in_fov_all & remaining_query_fov.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count
            
            for idx in selected_indices:
                overlap_ratios[idx] = -1.0
            
            best_idx = overlap_ratios.argmax().item()
            best_score = overlap_ratios[best_idx].item()
            
            # if best_score <= 0:
            #     break

            selected_indices.append(best_idx)
            selected_ratios.append(best_score)
            
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
                        selected_ratios.append(0.0)
            
            for idx in range(len(self.c2w_list)):
                if len(selected_indices) >= k:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    selected_ratios.append(0.0)

        while len(selected_indices) < k:
            selected_indices.append(random.choice(selected_indices))

        return selected_indices, selected_ratios


    def retrieve_top_k_cameras_keep_last_worldmem(self,
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
        selected_ratios = []

        if len(self.c2w_list) > 0:
            last_idx = len(self.c2w_list) - 1
            
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count > 0:
                overlap_count = (in_fov_all[last_idx] & remaining_query_fov).sum().float()
                ratio = (overlap_count / query_fov_count).item()
            else:
                ratio = 0.0
                
            selected_indices.append(last_idx)
            selected_ratios.append(ratio)
            
            remaining_query_fov = remaining_query_fov & ~in_fov_all[last_idx]
        
        num_needed = k - len(selected_indices)
        for _ in range(min(num_needed, len(self.c2w_list))):
            query_fov_count = remaining_query_fov.sum().float()
            
            overlap_counts = (in_fov_all & remaining_query_fov.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count

            
            for idx in selected_indices:
                overlap_ratios[idx] = -1.0
            
            best_idx = overlap_ratios.argmax().item()
            best_score = overlap_ratios[best_idx].item()

            selected_indices.append(best_idx)
            selected_ratios.append(best_score)
            
            remaining_query_fov = remaining_query_fov & ~in_fov_all[best_idx]

        return selected_indices, selected_ratios


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
        selected_ratios = []
        
        for _ in range(min(k, len(self.c2w_list))):
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count == 0:
                break
            
            overlap_counts = (in_fov_all & remaining_query_fov.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count
            
            for idx in selected_indices:
                overlap_ratios[idx] = -1.0
            
            best_idx = overlap_ratios.argmax().item()
            best_score = overlap_ratios[best_idx].item()
            
            # if best_score <= 0:
            #     break

            selected_indices.append(best_idx)
            selected_ratios.append(best_score)
            
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
                        selected_ratios.append(0.0)
            
            for idx in range(len(self.c2w_list)):
                if len(selected_indices) >= k:
                    break
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    selected_ratios.append(0.0)

        while len(selected_indices) < k:
            selected_indices.append(random.choice(selected_indices))

        return selected_indices, selected_ratios
    
    def calculate_query_coverage(self,
                               query_c2w: torch.Tensor,
                               query_fxfycxcy: torch.Tensor,
                               selected_mem_indices: List[int],
                               ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        计算每个查询相机的视野被当前记忆库覆盖的比例。
        
        Args:
            query_c2w: (N, 4, 4) 查询相机的 C2W 矩阵
            query_fxfycxcy: (N, 4) 查询相机的内参
            selected_mem_indices: 选中的记忆库索引列表
        
        Returns:
            total_coverage: (N,) 每个查询相机的总覆盖率 [0, 1]
            max_single_coverage: (N,) 每个查询相机被单个记忆帧覆盖的最大比例
            best_mem_indices: (N,) 对应的最佳记忆帧索引
        """
        if query_c2w.dim() == 2:
            query_c2w = query_c2w.unsqueeze(0)
            query_fxfycxcy = query_fxfycxcy.unsqueeze(0)

        if len(selected_mem_indices) == 0:
            N = query_c2w.shape[0]
            return (torch.zeros(N, device=self.device), 
                    torch.zeros(N, device=self.device), 
                    [-1] * N)
            
        selected_c2w = self.c2w_list[selected_mem_indices]
        selected_fxfycxcy = self.fxfycxcy_list[selected_mem_indices]
        
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
        ) # (N*M, N)
        
        _, in_fov_all = project_points_to_camera(
            points_w,
            selected_c2w,
            selected_fxfycxcy,
            self.H, self.W
        ) # (N*M, K)
        
        is_covered_by_bank = in_fov_all.any(dim=1) # (N*M,)
        
        ratios = []
        max_single_ratios = []
        best_indices = []
        
        for i in range(n_query):
            start_idx = i * points_per_cam
            end_idx = (i + 1) * points_per_cam
            
            valid_mask = in_fov_query[start_idx:end_idx, i] # (M,)
            
            if valid_mask.sum() == 0:
                ratios.append(0.0)
                max_single_ratios.append(0.0)
                best_indices.append(-1)
                continue
                
            covered_mask = is_covered_by_bank[start_idx:end_idx] # (M,)
            covered_valid = valid_mask & covered_mask
            ratio = covered_valid.sum().float() / valid_mask.sum().float()
            ratios.append(ratio.item())
            
            # in_fov_all slice: (M, K)
            mem_fovs = in_fov_all[start_idx:end_idx]
            
            # (M, K) & (M, 1) -> (M, K) -> sum(0) -> (K,)
            intersections = (mem_fovs & valid_mask.unsqueeze(1)).sum(dim=0).float()
            single_ratios = intersections / valid_mask.sum().float()
            
            max_val, max_idx = single_ratios.max(dim=0)
            max_single_ratios.append(max_val.item())
            best_indices.append(selected_mem_indices[max_idx.item()])
            
        return (torch.tensor(ratios, device=self.device), 
                torch.tensor(max_single_ratios, device=self.device), 
                best_indices)

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
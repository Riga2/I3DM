import torch
import math
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from einops import rearrange
from torch.nn import functional as F
from torch.amp import autocast

def generate_points_in_sphere(n_points: int, radius: float, device: torch.device = None) -> torch.Tensor:
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
                 retrieval_ckpt_path,
                 image_size: Tuple[int, int] = (352, 640),
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = torch.device('cuda'),
                 ):
        self.H, self.W = image_size
        self.device = device
        self.dtype = dtype      
        
        self.c2w_list = torch.empty((0, 4, 4), device=device)
        self.fxfycxcy_list = torch.empty((0, 4), device=device)
        self.rgb_list = torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=device)
        self.frames_num = 0

        from base_diffsynth.models.wan_scene_decoder_retrieval_occ import SceneDecoderOnlyOccAnalysis
        self.nvs_model = SceneDecoderOnlyOccAnalysis(load_path=retrieval_ckpt_path, exit_layer=6).to(device='cuda', dtype=torch.bfloat16).eval()

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

    def retrieve_top_k_cameras_learned_use(self,
                                   query_c2w: torch.Tensor,
                                   query_fxfycxcy: torch.Tensor,
                                   k: int = 4) -> List[int]:

        if self.frames_num <= k:
            top_indices = list(range(self.frames_num))
        else:
            last_idx = self.frames_num - 1
            
            confs = []
            confidence_maps_all_views = []
            
            print(f"Start evaluating {self.frames_num - 1} candidate frames...")

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

            num_to_select = min(k - 1, len(confidence_maps_all_views))
            
            selected_candidate_indices = []
            candidate_indices = list(range(len(confidence_maps_all_views)))
            
            if len(confidence_maps_all_views) > 0:
                V, H, W = confidence_maps_all_views[0].shape
                device = confidence_maps_all_views[0].device
            else:
                return [last_idx], []

            current_best_coverage = torch.full((V, H, W), -float('inf')).to(device=device)

            print(f"Start Multi-View Greedy Selection for {num_to_select} frames...")

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
                    print(f"  [Select Step {step+1}] Picked {best_idx}, Utility: {best_gain:.4f}")
                else:
                    break
        
            top_indices = [last_idx] + selected_candidate_indices

        while len(top_indices) < k:
            top_indices.append(random.choice(top_indices))

        return top_indices
        
    def add_key_frames(self,
                        frames_c2w: torch.Tensor,                   # (F, 4, 4)
                        frames_fxfycxcy: torch.Tensor,              # (F, 4)
                        frames_rgb: Optional[torch.Tensor] = None   # (F, 3, H, W)
                        ):

        self.c2w_list = torch.cat([self.c2w_list, frames_c2w.to(device=self.device)], dim=0)
        self.fxfycxcy_list = torch.cat([self.fxfycxcy_list, frames_fxfycxcy.to(device=self.device)], dim=0)
        self.rgb_list = torch.cat([self.rgb_list, frames_rgb.to(device=self.device)], dim=0)

        self.frames_num += len(frames_c2w)

    def get_retrieved_data_frames(self, frame_indices: List[int]) -> Dict[str, torch.Tensor]:
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
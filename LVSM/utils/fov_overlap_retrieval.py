
import torch
import math
import numpy as np
from typing import List, Tuple, Dict, Optional


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


class FOVOverlapRetriever:
    
    def __init__(self, 
                 c2w_list: torch.Tensor,
                 fxfycxcy_list: torch.Tensor,
                 union_size: int = 11,
                 n_sample_points: int = 10000,
                 sample_radius: float = 30.0,
                 image_size: Tuple[int, int] = (352, 640)):
        self.c2w_list = c2w_list
        self.fxfycxcy_list = fxfycxcy_list
        self.union_size = union_size
        self.n_sample_points = n_sample_points
        self.sample_radius = sample_radius
        self.H, self.W = image_size
        
        self.device = c2w_list.device
        self.num_cameras = c2w_list.shape[0]
        
        self._create_camera_unions()
    
    def _create_camera_unions(self):
        self.camera_unions = []
        
        for start_idx in range(0, self.num_cameras - self.union_size + 1, self.union_size):
            end_idx = start_idx + self.union_size
            
            union_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'c2w': self.c2w_list[start_idx:end_idx],
                'fxfycxcy': self.fxfycxcy_list[start_idx:end_idx],
            }
            self.camera_unions.append(union_info)
    
    def compute_fov_overlap(self, 
                            query_c2w: torch.Tensor,
                            query_fxfycxcy: torch.Tensor,
                            union_idx: int) -> float:
        union_info = self.camera_unions[union_idx]
        union_c2w = union_info['c2w']
        union_fxfycxcy = union_info['fxfycxcy']
        
        query_center = query_c2w[:3, 3]
        
        points_local = generate_points_in_sphere(
            self.n_sample_points, 
            self.sample_radius, 
            self.device
        )
        points_w = points_local + query_center.unsqueeze(0)  # (N_P, 3)
        
        _, in_fov_query = project_points_to_camera(
            points_w, 
            query_c2w.unsqueeze(0),  # (1, 4, 4)
            query_fxfycxcy.unsqueeze(0),  # (1, 4)
            self.H, self.W
        )
        in_fov_query = in_fov_query.squeeze(-1)  # (N_P,)
        
        _, in_fov_union = project_points_to_camera(
            points_w,
            union_c2w,  # (union_size, 4, 4)
            union_fxfycxcy,  # (union_size, 4)
            self.H, self.W
        )
        in_fov_union_any = in_fov_union.any(dim=1)  # (N_P,)
        
        query_fov_count = in_fov_query.sum().float()
        if query_fov_count == 0:
            return 0.0
        
        overlap_count = (in_fov_query & in_fov_union_any).sum().float()
        overlap_ratio = (overlap_count / query_fov_count).item()
        
        return overlap_ratio
    
    def retrieve_top_k_unions(self,
                              query_c2w: torch.Tensor,
                              query_fxfycxcy: torch.Tensor,
                              k: int = 4,
                              remove_overlap: bool = True) -> List[Dict]:
        if not remove_overlap:
            overlaps = []
            for union_idx in range(len(self.camera_unions)):
                overlap = self.compute_fov_overlap(query_c2w, query_fxfycxcy, union_idx)
                overlaps.append({
                    'union_idx': union_idx,
                    'score': overlap,
                    'start_idx': self.camera_unions[union_idx]['start_idx'],
                    'end_idx': self.camera_unions[union_idx]['end_idx'],
                })
            
            overlaps.sort(key=lambda x: x['score'], reverse=True)
            return overlaps[:min(k, len(overlaps))]
        
        query_center = query_c2w[:3, 3]
        
        points_local = generate_points_in_sphere(
            self.n_sample_points, 
            self.sample_radius, 
            self.device
        )
        points_w = points_local + query_center.unsqueeze(0)  # (N_P, 3)
        
        _, in_fov_query = project_points_to_camera(
            points_w, 
            query_c2w.unsqueeze(0),
            query_fxfycxcy.unsqueeze(0),
            self.H, self.W
        )
        in_fov_query = in_fov_query.squeeze(-1)  # (N_P,)
        
        union_fov_masks = []
        for union_info in self.camera_unions:
            _, in_fov_union = project_points_to_camera(
                points_w,
                union_info['c2w'],
                union_info['fxfycxcy'],
                self.H, self.W
            )
            in_fov_union_any = in_fov_union.any(dim=1)  # (N_P,)
            union_fov_masks.append(in_fov_union_any)
        
        union_fov_masks = torch.stack(union_fov_masks, dim=0)  # (num_unions, N_P)
        
        remaining_query_fov = in_fov_query.clone()
        selected_indices = []
        results = []
        
        for _ in range(min(k, len(self.camera_unions))):
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count == 0:
                break
            
            overlap_counts = (union_fov_masks & remaining_query_fov.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count
            
            for idx in selected_indices:
                overlap_ratios[idx] = -1.0
            
            best_idx = overlap_ratios.argmax().item()
            best_score = overlap_ratios[best_idx].item()
            
            selected_indices.append(best_idx)
            results.append({
                'union_idx': best_idx,
                'score': best_score,
                'start_idx': self.camera_unions[best_idx]['start_idx'],
                'end_idx': self.camera_unions[best_idx]['end_idx'],
            })
            
            remaining_query_fov = remaining_query_fov & ~union_fov_masks[best_idx]
        
        return results
    
    def retrieve_top_k_cameras(self,
                               query_c2w: torch.Tensor,
                               query_fxfycxcy: torch.Tensor,
                               k: int = 44,
                               remove_overlap: bool = True) -> List[int]:
        query_center = query_c2w[:3, 3]
        
        points_local = generate_points_in_sphere(
            self.n_sample_points, 
            self.sample_radius, 
            self.device
        )
        points_w = points_local + query_center.unsqueeze(0)  # (N_P, 3)
        
        _, in_fov_query = project_points_to_camera(
            points_w, 
            query_c2w.unsqueeze(0),
            query_fxfycxcy.unsqueeze(0),
            self.H, self.W
        )
        in_fov_query = in_fov_query.squeeze(-1)  # (N_P,)
        
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
            
            _, top_indices = overlap_ratios.topk(min(k, self.num_cameras))
            return top_indices.tolist()
        
        remaining_query_fov = in_fov_query.clone()
        selected_indices = []
        
        for _ in range(min(k, self.num_cameras)):
            query_fov_count = remaining_query_fov.sum().float()
            if query_fov_count == 0:
                break
            
            overlap_counts = (in_fov_all & remaining_query_fov.unsqueeze(0)).sum(dim=1).float()
            overlap_ratios = overlap_counts / query_fov_count
            
            for idx in selected_indices:
                overlap_ratios[idx] = -1.0
            
            best_idx = overlap_ratios.argmax().item()
            selected_indices.append(best_idx)
            
            remaining_query_fov = remaining_query_fov & ~in_fov_all[best_idx]
        
        return selected_indices


def create_fov_overlap_retriever(c2w_list: torch.Tensor,
                                  fxfycxcy_list: torch.Tensor,
                                  union_size: int = 11,
                                  n_sample_points: int = 10000,
                                  sample_radius: float = 30.0,
                                  image_size: Tuple[int, int] = (352, 640)) -> FOVOverlapRetriever:
    return FOVOverlapRetriever(
        c2w_list=c2w_list,
        fxfycxcy_list=fxfycxcy_list,
        union_size=union_size,
        n_sample_points=n_sample_points,
        sample_radius=sample_radius,
        image_size=image_size
    )

# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
import math
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union
from einops import rearrange
from torch.nn import functional as F
from torch.amp import autocast
import os
from torchvision.utils import save_image
import sys
from copy import deepcopy
from huggingface_hub import hf_hub_download
import PIL.Image as Image
# try:
from util import average_camera_pose, visualize_depth, Octree
# except ImportError:
#     # If not found (e.g. if util.py is not in path), define simple fallback or assume it's available
#     from .util import average_camera_pose, visualize_depth # Try relative import if module
    
# For surfel-based retrieval
sys.path.append("./examples/wanvideo/model_training/extern/CUT3R")

from extern.CUT3R.surfel_inference import run_inference_from_pil, prepare_input_from_pil
from extern.CUT3R.add_ckpt_path import add_path_to_dust3r
from extern.CUT3R.src.dust3r.model import ARCroco3DStereo

class Surfel:
    def __init__(self, position, normal, radius=1.0, color=None):
        self.position = position
        self.normal = normal
        self.radius = radius
        self.color = color

    def __repr__(self):
        return (f"Surfel(position={self.position}, "
                f"normal={self.normal}, radius={self.radius}, "
                f"color={self.color})")

class KeyFrameMemoryBankGeometryRetrieval:
    def __init__(self,
                 n_sample_points: int = 10000,
                 sample_radius: float = 30.0,
                 image_size: Tuple[int, int] = (352, 640),
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = torch.device('cuda'),
                 use_surfel: bool = False,
                 surfel_model_path: str = "liguang0115/cut3r",
                 ):
        self.n_sample_points = n_sample_points
        self.sample_radius = sample_radius
        self.H, self.W = image_size
        self.device = device
        self.dtype = dtype      
        self.use_surfel = use_surfel
        
        self.c2w_list = torch.empty((0, 4, 4), device=device)
        self.fxfycxcy_list = torch.empty((0, 4), device=device)
        self.rgb_list = torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=device)
        self.frames_num = 0

        # Surfel related initialization
        self.surfels = []
        self.surfel_Ks = []
        self.surfel_depths = []
        self.surfel_to_timestep = {}
        self.surfel_width = 640
        self.surfel_height = 320
        self.surfel_model = None
        
        if self.use_surfel:
            ckpt_path = hf_hub_download(repo_id=surfel_model_path, filename="cut3r_512_dpt_4_64.pth")
            add_path_to_dust3r(ckpt_path)
            self.surfel_model = ARCroco3DStereo.from_pretrained(ckpt_path).to(device)
            self.surfel_model.eval()
            print(f"Loaded Surfel model from {ckpt_path}")

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
        if self.frames_num >= 2:
            self.update_with_surfels()

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

    def estimate_normal_from_pointmap(self, pointmap: torch.Tensor) -> torch.Tensor:
        h, w = pointmap.shape[:2]
        device = pointmap.device
        dtype = pointmap.dtype
        
        normal_map = torch.zeros((h, w, 3), device=device, dtype=dtype)
        
        for y in range(h):
            for x in range(w):
                if x+1 >= w or y+1 >= h:
                    continue
                
                p_center = pointmap[y, x]
                p_right  = pointmap[y, x+1]
                p_down   = pointmap[y+1, x]
                
                v1 = p_right - p_center
                v2 = p_down - p_center
                
                v1 = v1 / (torch.linalg.norm(v1) + 1e-8)
                v2 = v2 / (torch.linalg.norm(v2) + 1e-8)
                
                n_c = torch.cross(v1, v2)
                norm_len = torch.linalg.norm(n_c)
                
                if norm_len < 1e-8:
                    continue
                
                normal_map[y, x] = n_c / norm_len
        
        return normal_map

    def pointmap_to_surfels(self,
                            pointmap: torch.Tensor,
                            focal_lengths: torch.Tensor,
                            depths: torch.Tensor,
                            confs: torch.Tensor,
                            poses: torch.Tensor,
                            radius_scale: float = 0.5,
                            estimate_normals: bool = True):
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).to(self.device)
        if isinstance(focal_lengths, np.ndarray):
            focal_lengths = torch.from_numpy(focal_lengths).to(self.device)
        if isinstance(depths, np.ndarray):
            depths = torch.from_numpy(depths).to(self.device)
        if isinstance(confs, np.ndarray):
            confs = torch.from_numpy(confs).to(self.device)
            
        pointmap = pointmap.to(self.device)
        focal_lengths = focal_lengths.to(self.device)
        depths = depths.to(self.device)
        confs = confs.to(self.device)
        poses = poses.to(self.device)
            
        if len(focal_lengths) == 2:
            focal_lengths = torch.mean(focal_lengths, dim=0)
            
        if estimate_normals:
            normal_map = self.estimate_normal_from_pointmap(pointmap)
        else:
            normal_map = torch.zeros_like(pointmap)
            
        depth_threshold = torch.quantile(depths, 0.999)
        valid_mask = (depths <= depth_threshold) & (confs >= 1.0) # threshold hardcoded as in vmem_pipeline default or adjusted
        
        positions = pointmap[valid_mask]
        normals = normal_map[valid_mask]
        valid_depths = depths[valid_mask]
        
        camera_pos = poses[0:3, 3]
        view_directions = positions - camera_pos.unsqueeze(0)
        view_directions = F.normalize(view_directions, dim=1)
        
        dot_products = torch.sum(view_directions * normals, dim=1)
        
        flip_mask = dot_products < 0
        normals[flip_mask] = -normals[flip_mask]
        
        dot_products = torch.abs(torch.sum(view_directions * normals, dim=1))
        
        adjustment_values = 0.2 + 0.8 * dot_products
        radii = (radius_scale * valid_depths / focal_lengths / adjustment_values)
        
        positions = positions.detach().cpu().numpy()
        normals = normals.detach().cpu().numpy()
        radii = radii.detach().cpu().numpy()
        
        surfels = [Surfel(pos, norm, rad) for pos, norm, rad in zip(positions, normals, radii)]
        
        return surfels

    def render_surfels_to_image_old(
        self,
        surfels,
        poses,
        focal_lengths,
        principal_points,
        image_width,
        image_height,
        disk_resolution=16
    ):
        if isinstance(focal_lengths, torch.Tensor):
            focal_lengths = focal_lengths.cpu().numpy()
        if isinstance(principal_points, torch.Tensor):
            principal_points = principal_points.cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.cpu().numpy()

        surfel_index_map = np.full((image_height, image_width), -1, dtype=np.int32)
        z_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)
        cos_buffer = np.zeros((image_height, image_width), dtype=np.float32)

        fx, fy = focal_lengths[0], focal_lengths[1]
        cx, cy = principal_points[0], principal_points[1]
        R = poses[0:3, 0:3]
        t = poses[0:3, 3]
        
        near_z = 0.1
        far_z = 100000.0
        
        positions = np.array([s.position for s in surfels])
        positions_h = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        
        extrinsics = np.zeros((4, 4))
        extrinsics[0:3, 0:3] = np.linalg.inv(R)
        extrinsics[0:3, 3] = -np.linalg.inv(R) @ t
        extrinsics[3, 3] = 1
        
        cam_points = (extrinsics @ positions_h.T).T
        cam_points = cam_points[:, :3] / cam_points[:, 3:]
        
        in_front = cam_points[:, 2] > near_z
        behind_far = cam_points[:, 2] < far_z
        
        screen_x = fx * (cam_points[:, 0] / cam_points[:, 2]) + cx
        screen_y = fy * (cam_points[:, 1] / cam_points[:, 2]) + cy
        
        margin = 50
        in_screen_x = (screen_x >= -margin) & (screen_x < image_width + margin)
        in_screen_y = (screen_y >= -margin) & (screen_y < image_height + margin)
        
        visible_mask = in_front & behind_far & in_screen_x & in_screen_y
        visible_indices = np.where(visible_mask)[0]

        def point_in_polygon_2d(px, py, polygon):
            n = len(polygon)
            inside = False
            p1x, p1y = polygon[0]
            for i in range(n + 1):
                p2x, p2y = polygon[i % n]
                if min(p1x, p2x) < px <= max(p1x, p2x):
                    if py <= max(p1y, p2y):
                        if p1x != p2x:
                            xinters = (px - p1x) * (p2y - p1y) / (p2x - p1x) + p1y
                        if p1y == p2y or py <= xinters:
                            inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        angles = np.linspace(0, 2*math.pi, disk_resolution, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        for idx in visible_indices:
            surfel = surfels[idx]
            cp = cam_points[idx]
            
            normal_cam = extrinsics[0:3, 0:3] @ surfel.normal
            
            view_dir = -cp / np.linalg.norm(cp)
            cos_val = np.dot(normal_cam, view_dir)
            
            if cos_val <= 0:
                continue
                
            u0 = np.array([normal_cam[1], -normal_cam[0], 0])
            if np.linalg.norm(u0) < 1e-6:
                u0 = np.array([1, 0, 0])
            u0 = u0 / np.linalg.norm(u0)
            v0 = np.cross(normal_cam, u0)
            
            disk_points_cam = []
            for k in range(disk_resolution):
                pt_offset = surfel.radius * (cos_angles[k] * u0 + sin_angles[k] * v0)
                pt_cam = cp + pt_offset
                disk_points_cam.append(pt_cam)
            
            disk_points_scr = []
            for pt in disk_points_cam:
                if pt[2] <= near_z:
                    continue
                sx = fx * (pt[0] / pt[2]) + cx
                sy = fy * (pt[1] / pt[2]) + cy
                disk_points_scr.append((sx, sy))
                
            if len(disk_points_scr) < 3:
                continue
            
            poly_min_x = max(0, int(min(p[0] for p in disk_points_scr)))
            poly_max_x = min(image_width - 1, int(max(p[0] for p in disk_points_scr)))
            poly_min_y = max(0, int(min(p[1] for p in disk_points_scr)))
            poly_max_y = min(image_height - 1, int(max(p[1] for p in disk_points_scr)))
            
            for y in range(poly_min_y, poly_max_y + 1):
                for x in range(poly_min_x, poly_max_x + 1):
                    if point_in_polygon_2d(x, y, disk_points_scr):
                        dist_sq = cp[0]*cp[0] + cp[1]*cp[1] + cp[2]*cp[2]
                        current_depth = np.sqrt(dist_sq)
                        
                        if current_depth < z_buffer[y, x]:
                            z_buffer[y, x] = current_depth
                            surfel_index_map[y, x] = idx
                            cos_buffer[y, x] = cos_val

        depth = z_buffer
        depth[depth == np.inf] = 0

        return {
            "depth": depth,
            "surfel_index_map": surfel_index_map,
            "cos_value_map": cos_buffer
        }

    def render_surfels_to_image(
        self,
        surfels,
        poses,
        focal_lengths,
        principal_points,
        image_width,
        image_height,
        disk_resolution=16
    ):
        """
        Renders oriented surfels into a 2D RGB image with a simple z-buffer.
        Each surfel is treated as a 2D disk in 3D, oriented by its normal.
        The disk is approximated by a polygon of 'disk_resolution' segments.

        Args:
            surfels (list): List of Surfel objects, each having:
                - position: (x, y, z) in world coords
                - normal:   (nx, ny, nz)
                - radius:   float, radius in world units
            poses (torch.Tensor): Tensor of poses, shape [4, 4]
            focal_lengths (torch.Tensor): Tensor of focal lengths, shape [2]
            principal_points (torch.Tensor): Tensor of principal points, shape [2]
            image_width, image_height (int): output image size
            disk_resolution (int): number of segments for approximating each disk

        Returns:
            Dictionary containing:
            - depth: depth map
            - surfel_index_map: map of surfel indices
            - cos_value_map: map of cosine values between view and normal directions
        """
        if isinstance(focal_lengths, torch.Tensor):
            focal_lengths = focal_lengths.detach().cpu().numpy()
        if isinstance(principal_points, torch.Tensor):
            principal_points = principal_points.detach().cpu().numpy()
        if isinstance(poses, torch.Tensor):
            poses = poses.detach().cpu().numpy()

        # Initialize buffers
        surfel_index_map = np.full((image_height, image_width), -1, dtype=np.int32)
        z_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)
        cos_buffer = np.zeros((image_height, image_width), dtype=np.float32)

        # Unpack camera parameters
        fx, fy, cx, cy = focal_lengths[0], focal_lengths[1], principal_points[0], principal_points[1]
        R = poses[0:3, 0:3]
        t = poses[0:3, 3]
        
        # Compute view frustum planes in world space
        # We'll use 6 planes: near, far, left, right, top, bottom
        near_z = 0.1  # Near plane distance
        far_z = 100000.0  # Far plane distance
        
        # Convert all surfel positions to camera space at once for efficient culling
        positions = np.array([s.position for s in surfels])
        positions_h = np.concatenate([positions, np.ones((len(positions), 1))], axis=1)
        
        # Compute camera matrix
        extrinsics = np.zeros((4, 4))
        extrinsics[0:3, 0:3] = np.linalg.inv(R)
        extrinsics[0:3, 3] = -np.linalg.inv(R) @ t
        extrinsics[3, 3] = 1
        
        # Transform all points to camera space at once
        cam_points = (extrinsics @ positions_h.T).T
        cam_points = cam_points[:, :3] / cam_points[:, 3:]
        
        # Compute view frustum culling mask
        in_front = cam_points[:, 2] > near_z
        behind_far = cam_points[:, 2] < far_z
        
        # Project points to get screen coordinates
        screen_x = fx * (cam_points[:, 0] / cam_points[:, 2]) + cx
        screen_y = fy * (cam_points[:, 1] / cam_points[:, 2]) + cy
        
        # Check which points are within screen bounds (with some margin for surfel radius)
        margin = 50  # Margin in pixels to account for surfel radius
        in_screen_x = (screen_x >= -margin) & (screen_x < image_width + margin)
        in_screen_y = (screen_y >= -margin) & (screen_y < image_height + margin)
        
        # Combine all culling masks
        visible_mask = in_front & behind_far & in_screen_x & in_screen_y
        visible_indices = np.where(visible_mask)[0]

        def point_in_polygon_2d(px, py, polygon):
            """Fast point-in-polygon test using ray casting"""
            inside = False
            n = len(polygon)
            j = n - 1
            for i in range(n):
                if (((polygon[i][1] > py) != (polygon[j][1] > py)) and
                    (px < (polygon[j][0] - polygon[i][0]) * (py - polygon[i][1]) /
                     (polygon[j][1] - polygon[i][1] + 1e-15) + polygon[i][0])):
                    inside = not inside
                j = i
            return inside

        # Pre-compute angle samples for circle approximation
        angles = np.linspace(0, 2*math.pi, disk_resolution, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Process only visible surfels
        for idx in visible_indices:
            surfel = surfels[idx]
            px, py, pz = surfel.position
            nx, ny, nz = surfel.normal
            radius = surfel.radius

            # Skip degenerate normals
            normal = np.array([nx, ny, nz], dtype=float)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                continue
            normal /= norm_len

            # Compute view direction and cosine value
            point_direction = (px, py, pz) - t
            point_direction = point_direction / np.linalg.norm(point_direction)
            cos_value = np.dot(point_direction, normal)

            # Skip backfaces
            if cos_value < 0:
                continue

            # Build local coordinate frame
            up = np.array([0, 0, 1], dtype=float)
            if abs(np.dot(normal, up)) > 0.9:
                up = np.array([0, 1, 0], dtype=float)
            xAxis = np.cross(normal, up)
            xAxis /= np.linalg.norm(xAxis)
            yAxis = np.cross(normal, xAxis)
            yAxis /= np.linalg.norm(yAxis)

            # Generate circle points efficiently
            offsets = radius * (cos_angles[:, None] * xAxis + sin_angles[:, None] * yAxis)
            circle_points = positions[idx] + offsets

            # Project all circle points at once
            circle_points_h = np.concatenate([circle_points, np.ones((len(circle_points), 1))], axis=1)
            cam_circle = (extrinsics @ circle_points_h.T).T
            depths = cam_circle[:, 2]
            valid_mask = depths > 0
            if not np.any(valid_mask):
                continue

            screen_points = np.zeros((len(circle_points), 2))
            screen_points[:, 0] = fx * (cam_circle[:, 0] / depths) + cx
            screen_points[:, 1] = fy * (cam_circle[:, 1] / depths) + cy
            
            # Get bounding box
            valid_points = screen_points[valid_mask]
            if len(valid_points) < 3:
                continue

            min_x = max(0, int(np.floor(np.min(valid_points[:, 0]))))
            max_x = min(image_width - 1, int(np.ceil(np.max(valid_points[:, 0]))))
            min_y = max(0, int(np.floor(np.min(valid_points[:, 1]))))
            max_y = min(image_height - 1, int(np.ceil(np.max(valid_points[:, 1]))))

            # Average depth for z-buffer
            avg_depth = float(np.mean(depths[valid_mask]))

            # Rasterize polygon
            for py_ in range(min_y, max_y + 1):
                for px_ in range(min_x, max_x + 1):
                    if point_in_polygon_2d(px_, py_, valid_points):
                        if avg_depth < z_buffer[py_, px_]:
                            z_buffer[py_, px_] = avg_depth
                            surfel_index_map[py_, px_] = idx
                            cos_buffer[py_, px_] = cos_value

        # Clean up depth buffer
        depth = z_buffer
        depth[depth == np.inf] = 0

        return {
            "depth": depth,
            "surfel_index_map": surfel_index_map,
            "cos_value_map": cos_buffer
        }

    def construct_surfels_for_last_added_frames(self, start_idx=0, end_idx=None):
        if not self.use_surfel or self.surfel_model is None:
            return

        if end_idx is None:
            end_idx = len(self.rgb_list)
        
        if start_idx >= end_idx:
            return

        # Prepare input images
        input_images = []
        for i in range(start_idx, end_idx):
            img_tensor = self.rgb_list[i].permute(1, 2, 0).cpu().numpy()
            img_pil = Image.fromarray((img_tensor * 255).astype(np.uint8))
            
            start_w = 0
            start_h = (352 - 320) // 2
            image = img_pil.crop((start_w, start_h, start_w + 640, start_h + 320))
        
            input_images.append(image)

        # input_images[-1].save(f"./test_geo.png")
            
        c2ws = self.c2w_list[start_idx:end_idx].cpu().numpy()
        # Transform c2ws if needed (flip Y/Z like in vmem_pipeline)
        c2ws_transformed = deepcopy(c2ws)
        # c2ws_transformed[..., :, [1, 2]] *= -1

        scene = run_inference_from_pil(
            input_images,
            self.surfel_model,
            poses=c2ws_transformed,
            depths=torch.from_numpy(np.array(self.surfel_depths)) if len(self.surfel_depths) > 0 else None,
            lr=0.01,
            niter=400,
            visualize=False,
            device=self.device,
        )

        pointcloud = torch.cat(scene['point_clouds'], dim=0)
        confs = torch.cat(scene['confidences'], dim=0)
        depths = torch.cat(scene['depths'], dim=0)
        focal_lengths = scene['camera_info']['focal']
        self.surfel_Ks.extend([focal_lengths[i] for i in range(len(focal_lengths))])
        self.surfel_depths = [depths[i].detach().cpu().numpy() for i in range(len(depths))]
        
        shrink_factor = 0.05 # default in vmem_pipeline config
        
        # Resize pointcloud
        pointcloud = pointcloud.permute(0, 3, 1, 2)
        pointcloud = F.interpolate(pointcloud, scale_factor=shrink_factor, mode='bilinear')
        pointcloud = pointcloud.permute(0, 2, 3, 1)

        depths = depths.unsqueeze(1)
        depths = F.interpolate(depths, scale_factor=shrink_factor, mode='bilinear')
        depths = depths.squeeze(1)

        confs = confs.unsqueeze(1)
        confs = F.interpolate(confs, scale_factor=shrink_factor, mode='bilinear')
        confs = confs.squeeze(1)

        # Extend surfel list
        for rel_idx, frame_idx in enumerate(range(start_idx, end_idx)):
            surfels = self.pointmap_to_surfels(
                pointmap=pointcloud[rel_idx],
                focal_lengths=torch.tensor(focal_lengths[rel_idx]) * shrink_factor,
                depths=depths[rel_idx],
                confs=confs[rel_idx],
                poses=c2ws_transformed[rel_idx],
                estimate_normals=True,
                radius_scale=0.5,
            )
            
            # surfel_start_index = len(self.surfels)
            # self.surfels.extend(surfels)
            
            # # Map new surfels to timesteps (frame indices)
            # for j in range(len(surfels)):
            #     self.surfel_to_timestep[surfel_start_index + j] = [frame_idx]
            
            filtered_surfels, self.surfel_to_timestep = self.merge_surfels(
                surfels, 
                frame_idx,
                self.surfels,
                self.surfel_to_timestep,
            )
            
            surfel_start_index = len(self.surfels)
            self.surfels.extend(filtered_surfels)
            for j in range(len(filtered_surfels)):
                 self.surfel_to_timestep[surfel_start_index + j] = [frame_idx]

    def update_with_surfels(self):
        # Update surfels for newly added frames
        # Assuming frames_num tracks total frames, we check surfel metadata to see where we left off
        # But we don't have a simple tracker for last surfelized frame.
        # We can deduce from surfel_to_timestep.
        
        current_max_frame = -1
        if self.surfel_to_timestep:
            current_max_frame = max([max(v) for v in self.surfel_to_timestep.values()])
        
        start_idx = current_max_frame + 1
        end_idx = self.frames_num
        
        if start_idx < end_idx:
            print(f"Constructing surfels for frames {start_idx} to {end_idx}")
            self.construct_surfels_for_last_added_frames(start_idx, end_idx)

    def merge_surfels(
        self,
        new_surfels: list,
        current_timestep: int,
        existing_surfels: list,
        existing_surfel_to_timestep: dict,
        position_threshold: Union[float, None] = None,  # Now optional
        normal_threshold: float = 0.6,
        max_points_per_node: int = 10 
    ):

        assert len(existing_surfels) == len(existing_surfel_to_timestep), (
            "existing_surfels and existing_surfel_to_timestep should have the same length"
        )
        
        # Automatically calculate position threshold if not provided
        if position_threshold is None:
            # Calculate average radius from both new and existing surfels
            all_radii = np.array([s.radius for s in existing_surfels + new_surfels])
            if len(all_radii) > 0:
                # Use mean radius as base threshold with a scaling factor
                mean_radius = np.mean(all_radii)
                std_radius = np.std(all_radii)
                # Position threshold = mean + 0.5 * std to account for variance
                position_threshold = mean_radius + 0.5 * std_radius
            else:
                # Fallback to default if no surfels available
                position_threshold = 0.025

        positions = np.array([s.position for s in existing_surfels])  # Shape: (N, 3)
        normals = np.array([s.normal for s in existing_surfels])      # Shape: (N, 3)

        if len(positions) > 0:
            octree = Octree(positions, max_points=max_points_per_node)
        else:
            octree = None
        

        filtered_surfels = []
        
        merge_count = 0
        for new_surfel in new_surfels:
            is_merged = False
            if octree is not None:
                neighbor_indices = octree.query_ball_point(new_surfel.position, position_threshold)
            else:
                neighbor_indices = []
            
            for idx in neighbor_indices:
                if np.dot(normals[idx], new_surfel.normal) > normal_threshold:
                    if current_timestep not in existing_surfel_to_timestep[idx]:
                        existing_surfel_to_timestep[idx].append(current_timestep)
                    is_merged = True
                    merge_count += 1
                    break
            
            if not is_merged:
                filtered_surfels.append(new_surfel)
        
        print(f"merge_count: {merge_count}")
        return filtered_surfels, existing_surfel_to_timestep

    def geodesic_distance(self,
                        camera_pose1,
                        camera_pose2,
                        weight_translation=1,):
        """
        Computes the geodesic distance between two camera poses in SE(3).
        """
        # Extract the rotation and translation components
        R1 = camera_pose1[:3, :3]
        t1 = camera_pose1[:3, 3]
        R2 = camera_pose2[:3, :3]
        t2 = camera_pose2[:3, 3]

        # Compute the translation distance (Euclidean distance)
        translation_distance = torch.norm(t1 - t2)
        
        # Compute the relative rotation matrix
        R_relative = torch.matmul(R1.T, R2)
        
        # Compute the angular distance from the trace of the relative rotation matrix
        trace_value = torch.trace(R_relative)
        # Clamp the trace value to avoid numerical issues
        trace_value = torch.clamp(trace_value, -1.0, 3.0)
        angular_distance = torch.acos((trace_value - 1) / 2)
        
        # Combine the two distances
        geodesic_dist = translation_distance*weight_translation + angular_distance
            
        return geodesic_dist

    def retrieve_top_k_surfel(self, query_c2w, query_fxfycxcy, k=4, debug_dir=None, use_non_maximum_suppression=True):
        if self.frames_num < k:
            top_k = list(range(self.frames_num))
        else:
            if not self.surfels:
                return list(range(max(0, self.frames_num - k), self.frames_num))

            if query_c2w.dim() == 2:
                query_c2w = query_c2w.unsqueeze(0)
                query_fxfycxcy = query_fxfycxcy.unsqueeze(0)

            # Use average camera pose for retrieval
            # query_c2ws_np = query_c2w.cpu().numpy()
            c2w_transformed = average_camera_pose(query_c2w)
            
            # Transform average pose (flip Y/Z)
            # c2w_transformed = deepcopy(average_c2w)
            # c2w_transformed[0:3, 1] *= -1
            # c2w_transformed[0:3, 2] *= -1

            # Use average intrinsics
            # fx = torch.mean(query_fxfycxcy[:, 0])
            # fy = torch.mean(query_fxfycxcy[:, 1])
            # cx = torch.mean(query_fxfycxcy[:, 2])
            # cy = torch.mean(query_fxfycxcy[:, 3])
            
            target_K = np.mean(self.surfel_Ks, axis=0)
            # pp = torch.tensor([cx, cy])
            
            rendered = self.render_surfels_to_image(
                self.surfels,
                c2w_transformed,
                [target_K*0.65] * 2,
                principal_points=(int(self.surfel_width/2), int(self.surfel_height/2)),
                image_width=int(self.surfel_width),
                image_height=int(self.surfel_height)
            )
            
            surfel_index_map = rendered["surfel_index_map"]
            cos_value_map = rendered["cos_value_map"]
            depth_map = rendered["depth"] 

            if debug_dir is not None:
                visualize_depth(depth_map,
                                visualization_dir=debug_dir, 
                                file_name=f"retrieved_depth_surfels.png",
                                size=(self.surfel_width, self.surfel_height))
            
            filtered_surfel_indices = surfel_index_map[surfel_index_map >= 0]
            filtered_cos_values = cos_value_map[surfel_index_map >= 0]
            filtered_depth_values = depth_map[surfel_index_map >= 0]
            
            timestep_scores = {}
            for idx_in_filter, s_idx in enumerate(filtered_surfel_indices):
                cos_val = filtered_cos_values[idx_in_filter]
                depth_val = filtered_depth_values[idx_in_filter]
                
                if s_idx in self.surfel_to_timestep:
                    frame_indices = self.surfel_to_timestep[s_idx]
                    score = cos_val / (1 + depth_val)
                    for f_idx in frame_indices:
                        timestep_scores[f_idx] = timestep_scores.get(f_idx, 0) + score
            
            # Sort by score
            sorted_frames_scores = sorted(timestep_scores.items(), key=lambda x: x[1], reverse=True)
            sorted_frames = [f[0] for f in sorted_frames_scores]

            # NMS logic
            translation_distance_weight = 10.0 # Hardcoded for now or add to init

            initial_threshold = 1e8
            if use_non_maximum_suppression:
                # Calculate pairwise distances between existing frames
                pairwise_distances = []
                # Sample a subset if too many frames to speed up
                c2w_indices = list(range(len(self.c2w_list)))
                if len(c2w_indices) > 500:
                    c2w_indices = random.sample(c2w_indices, 500)
                
                for i in range(len(c2w_indices)):
                    for j in range(i+1, len(c2w_indices)):
                        idx1 = c2w_indices[i]
                        idx2 = c2w_indices[j]
                        sim = self.geodesic_distance(
                            self.c2w_list[idx1],
                            self.c2w_list[idx2],
                            weight_translation=translation_distance_weight
                        )
                        pairwise_distances.append(sim.item())
                
                if pairwise_distances:
                    pairwise_distances.sort()
                    percentile_idx = int(len(pairwise_distances) * 0.5)  # 50th percentile (median)
                    initial_threshold = pairwise_distances[percentile_idx]
                else:
                    initial_threshold = 1.0


            selected_indices = []
            current_threshold = initial_threshold
            
            # Always start with the best scoring pose
            if len(sorted_frames) > 0:
                selected_indices.append(sorted_frames[0])
                
            # Try with increasingly relaxed thresholds until we get enough frames
            while len(selected_indices) < k and current_threshold >= 1e-5:
                # Try to add each subsequent pose
                for idx in sorted_frames[1:]:
                    if len(selected_indices) >= k:
                        break
                    
                    # Check if already selected
                    if idx in selected_indices:
                        continue
                        
                    # Check if this candidate is sufficiently different from all selected frames
                    is_too_similar = False
                    for selected_idx in selected_indices:
                        similarity = self.geodesic_distance(
                            self.c2w_list[idx],
                            self.c2w_list[selected_idx],
                            weight_translation=translation_distance_weight
                        )
                        if similarity < current_threshold:
                            is_too_similar = True
                            break
                            
                    # Add to selected frames if not too similar to any existing selection
                    if not is_too_similar:
                        selected_indices.append(idx)
                
                # If we still don't have enough frames, relax the threshold and try again
                if len(selected_indices) < k:
                    current_threshold /= 1.5 # Relax factor
                else:
                    break
            
            top_k = selected_indices

            # If we don't have enough, fill with latest
            if len(top_k) < k:
                remaining = k - len(top_k)
                # Add latest distinct from top_k
                for i in range(self.frames_num - 1, -1, -1):
                    if i not in top_k:
                        top_k.append(i)
                        if len(top_k) == k:
                            break
        
        while len(top_k) < k:
            top_k.append(random.choice(top_k))

        return top_k
    
    def reset(self):
        self.c2w_list = torch.empty((0, 4, 4), device=self.device)
        self.fxfycxcy_list = torch.empty((0, 4), device=self.device)
        self.rgb_list = torch.empty((0, 3, self.H, self.W), dtype=torch.float32, device=self.device)
        self.frames_num = 0

        # Surfel related initialization
        self.surfels = []
        self.surfel_Ks = []
        self.surfel_depths = []
        self.surfel_to_timestep = {}
        self.surfel_width = 640
        self.surfel_height = 320
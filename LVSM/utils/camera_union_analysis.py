# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import torch
import numpy as np
from typing import List, Tuple, Dict
import scipy.spatial


class CameraFrustum:
    """
    表示单个相机的视锥体（Frustum）。
    用于计算视锥体在世界坐标系中的覆盖区域。
    """
    
    def __init__(self, c2w: torch.Tensor, fxfycxcy: torch.Tensor, depth_range: Tuple[float, float] = (0.1, 100.0)):
        """
        初始化相机视锥体。
        
        Args:
            c2w: (4, 4) 相机到世界坐标系的变换矩阵
            fxfycxcy: (4,) 包含 [fx, fy, cx, cy] 的内参
            depth_range: (near, far) 深度范围，用于定义视锥体的前后平面
        """
        self.c2w = c2w.clone().detach().cpu()
        self.fxfycxcy = fxfycxcy.clone().detach().cpu()
        
        fx, fy, cx, cy = self.fxfycxcy
        
        h, w = cy * 2, cx * 2
        
        corners_2d = np.array([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h],
            [w / 2, h / 2]
        ], dtype=np.float32)
        
        self.ray_directions = []
        for px, py in corners_2d:
            dx = (px - cx) / fx
            dy = (py - cy) / fy
            dz = 1.0
            direction = np.array([dx, dy, dz])
            direction = direction / np.linalg.norm(direction)
            self.ray_directions.append(direction)
        
        self.depth_range = depth_range
        
        self._compute_frustum_vertices()
    
    def _compute_frustum_vertices(self):
        """计算视锥体的8个顶点（近平面和远平面）"""
        near_dist, far_dist = self.depth_range
        
        camera_pos = self.c2w[:3, 3].numpy()
        
        R = self.c2w[:3, :3].numpy()
        
        corners_2d = [
            [0, 0],
            [self.fxfycxcy[3].item() * 2, 0],
            [0, self.fxfycxcy[2].item() * 2],
            [self.fxfycxcy[3].item() * 2, self.fxfycxcy[2].item() * 2],
        ]
        
        fx, fy, cx, cy = self.fxfycxcy.numpy()
        
        vertices = []
        for near_far_dist in [near_dist, far_dist]:
            for px, py in corners_2d:
                dx = (px - cx) / fx
                dy = (py - cy) / fy
                dz = 1.0
                
                direction = np.array([dx, dy, dz])
                length = np.linalg.norm(direction)
                direction = direction / length
                
                point_cam = direction * near_far_dist
                
                point_world = R @ point_cam + camera_pos
                vertices.append(point_world)
        
        self.frustum_vertices = np.array(vertices, dtype=np.float32)
    
    def get_frustum_vertices(self) -> np.ndarray:
        """
        获取视锥体的顶点（8个点）
        
        Returns:
            (8, 3) 视锥体顶点数组
        """
        return self.frustum_vertices
    
    def get_projection_area_xz(self) -> float:
        """
        在XZ平面上计算视锥体投影的面积
        
        Returns:
            投影面积（标量）
        """
        vertices_xz = self.frustum_vertices[:, [0, 2]]
        
        try:
            hull = scipy.spatial.ConvexHull(vertices_xz)
            area = hull.volume
        except:
            x_min, x_max = vertices_xz[:, 0].min(), vertices_xz[:, 0].max()
            z_min, z_max = vertices_xz[:, 1].min(), vertices_xz[:, 1].max()
            area = (x_max - x_min) * (z_max - z_min)
        
        return float(area)
    
    def get_center_in_world(self) -> np.ndarray:
        """
        获取相机中心（世界坐标系）
        
        Returns:
            (3,) 相机中心坐标
        """
        return self.c2w[:3, 3].numpy()
    
    def get_volume(self) -> float:
        """
        计算视锥体的体积（3D凸包体积）。
        
        Returns:
            视锥体体积（标量）
        """
        try:
            hull = scipy.spatial.ConvexHull(self.frustum_vertices)
            volume = hull.volume
        except:
            near_vertices = self.frustum_vertices[:4]
            far_vertices = self.frustum_vertices[4:]
            
            try:
                near_hull = scipy.spatial.ConvexHull(near_vertices)
                near_area = near_hull.volume
            except:
                near_area = 1.0
            
            near_z = near_vertices[:, 2].mean()
            far_z = far_vertices[:, 2].mean()
            height = abs(far_z - near_z)
            
            volume = near_area * height
        
        return float(volume)


class CameraUnionAnalyzer:
    """
    分析相机联合的可视区域。
    支持将相邻相机合并为联合视锥体，以及检索与新相机重叠最大的联合。
    """
    
    def __init__(self, camera_list: List[Tuple[torch.Tensor, torch.Tensor]], 
                 union_size: int = 5, depth_range: Tuple[float, float] = (0.1, 100.0)):
        """
        初始化相机联合分析器。
        
        Args:
            camera_list: 包含 (c2w, fxfycxcy) 的相机列表
            union_size: 每个union包含的相邻相机数量（M）
            depth_range: 视锥体的深度范围
        """
        self.camera_list = []
        for c2w, fxfycxcy in camera_list:
            frustum = CameraFrustum(c2w, fxfycxcy, depth_range)
            self.camera_list.append(frustum)
        
        self.union_size = union_size
        self.depth_range = depth_range
        self.camera_unions = []
        self.union_metadata = []
        
        self._create_camera_unions()
    
    def _create_camera_unions(self):
        """
        将相邻的相机合并为联合，每个联合包含M个相邻相机。
        """
        self.camera_unions = []
        self.union_metadata = []
        
        num_cameras = len(self.camera_list)
        
        for start_idx in range(0, num_cameras - self.union_size + 1, self.union_size):
            end_idx = start_idx + self.union_size
            
            union_cameras = self.camera_list[start_idx:end_idx]
            
            union_vertices = []
            for frustum in union_cameras:
                union_vertices.extend(frustum.get_frustum_vertices().tolist())
            
            union_vertices = np.array(union_vertices, dtype=np.float32)
            
            try:
                hull = scipy.spatial.ConvexHull(union_vertices)
                union_hull_vertices = union_vertices[hull.vertices]
                union_area_xz = float(hull.volume)
                union_area_2d_xz = self._compute_2d_projection_area(union_vertices)
            except:
                union_hull_vertices = union_vertices
                union_area_xz = np.inf
                union_area_2d_xz = self._compute_2d_projection_area(union_vertices)
            
            union_volume = self._compute_3d_volume(union_vertices)
            
            union_info = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'vertices': union_vertices,
                'hull_vertices': union_hull_vertices,
                'area_3d': union_area_xz,
                'area_2d_xz': union_area_2d_xz,
                'volume_3d': union_volume,
            }
            
            self.camera_unions.append(union_info)
            self.union_metadata.append({
                'union_idx': len(self.camera_unions) - 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_cameras': self.union_size,
            })
    
    def _compute_2d_projection_area(self, vertices_3d: np.ndarray) -> float:
        """
        计算3D顶点集合在XZ平面上的投影面积。
        
        Args:
            vertices_3d: (N, 3) 顶点数组
        
        Returns:
            投影面积
        """
        vertices_xz = vertices_3d[:, [0, 2]]
        
        try:
            hull = scipy.spatial.ConvexHull(vertices_xz)
            area = hull.volume
        except:
            x_min, x_max = vertices_xz[:, 0].min(), vertices_xz[:, 0].max()
            z_min, z_max = vertices_xz[:, 1].min(), vertices_xz[:, 1].max()
            area = (x_max - x_min) * (z_max - z_min)
        
        return float(area)
    
    def _compute_3d_volume(self, vertices_3d: np.ndarray) -> float:
        """
        计算3D顶点集合的凸包体积。
        
        Args:
            vertices_3d: (N, 3) 顶点数组
        
        Returns:
            凸包体积
        """
        try:
            if len(vertices_3d) < 4:
                return 0.0
            
            unique_vertices = np.unique(vertices_3d, axis=0)
            if len(unique_vertices) < 4:
                return 0.0
            
            hull = scipy.spatial.ConvexHull(unique_vertices)
            volume = hull.volume
        except:
            volume = 0.0
        
        return float(volume)
    
    def compute_union_overlap(self, union_idx: int, query_frustum: CameraFrustum) -> float:
        """
        计算查询相机与某个联合的重叠面积（在XZ平面投影）。
        
        Args:
            union_idx: union索引
            query_frustum: 查询相机的视锥体
        
        Returns:
            重叠面积
        """
        union_vertices_xz = self.camera_unions[union_idx]['vertices'][:, [0, 2]]
        query_vertices_xz = query_frustum.get_frustum_vertices()[:, [0, 2]]
        
        try:
            all_vertices = np.vstack([union_vertices_xz, query_vertices_xz])
            combined_hull = scipy.spatial.ConvexHull(all_vertices)
            
            union_area = combined_hull.volume
            
            try:
                union_hull = scipy.spatial.ConvexHull(union_vertices_xz)
                union_area_single = union_hull.volume
            except:
                union_area_single = self.camera_unions[union_idx]['area_2d_xz']
            
            try:
                query_hull = scipy.spatial.ConvexHull(query_vertices_xz)
                query_area_single = query_hull.volume
            except:
                query_area_single = query_frustum.get_projection_area_xz()
            
            intersection_area = union_area_single + query_area_single - union_area
            overlap_ratio = intersection_area / (min(union_area_single, query_area_single) + 1e-6)
            
            return float(max(0, intersection_area))
        
        except:
            return 0.0
    
    def compute_union_volume_overlap(self, union_idx: int, query_frustum: CameraFrustum) -> float:
        """
        计算查询相机与某个联合的3D体积重叠量。
        
        Args:
            union_idx: union索引
            query_frustum: 查询相机的视锥体
        
        Returns:
            3D体积重叠量（近似值）
        """
        union_vertices = self.camera_unions[union_idx]['vertices']
        query_vertices = query_frustum.get_frustum_vertices()
        
        try:
            all_vertices = np.vstack([union_vertices, query_vertices])
            
            unique_vertices = np.unique(all_vertices, axis=0)
            if len(unique_vertices) < 4:
                return 0.0
            
            combined_hull = scipy.spatial.ConvexHull(unique_vertices)
            combined_volume = combined_hull.volume
            
            union_volume = self.camera_unions[union_idx]['volume_3d']
            query_volume = query_frustum.get_volume()
            
            intersection_volume = union_volume + query_volume - combined_volume
            intersection_volume = max(0, intersection_volume)
            
            return float(intersection_volume)
        
        except:
            return 0.0
    
    def retrieve_top_k_unions(self, c2w: torch.Tensor, fxfycxcy: torch.Tensor, 
                            k: int = 3, metric: str = 'overlap') -> List[Dict]:
        """
        从所有联合中检索Top-K个与查询相机重叠最大的联合。
        
        Args:
            c2w: (4, 4) 查询相机的相机到世界矩阵
            fxfycxcy: (4,) 查询相机的内参
            k: 返回前K个结果
            metric: 排序指标，可选 'overlap'、'area_ratio'、'center_distance'、
                   'volume_overlap'、'volume_ratio'
        
        Returns:
            包含top-k联合信息的列表，每项包含索引、重叠面积/体积等信息
        """
        query_frustum = CameraFrustum(c2w, fxfycxcy, self.depth_range)
        
        overlaps = []
        for union_idx, union_info in enumerate(self.camera_unions):
            if metric == 'overlap':
                score = self.compute_union_overlap(union_idx, query_frustum)
            elif metric == 'area_ratio':
                query_area = query_frustum.get_projection_area_xz()
                union_area = union_info['area_2d_xz']
                overlap = self.compute_union_overlap(union_idx, query_frustum)
                score = overlap / (min(query_area, union_area) + 1e-6)
            elif metric == 'center_distance':
                query_center = query_frustum.get_center_in_world()
                union_centers = [self.camera_list[i].get_center_in_world() 
                               for i in range(union_info['start_idx'], union_info['end_idx'])]
                union_center = np.mean(union_centers, axis=0)
                distance = np.linalg.norm(query_center - union_center)
                score = -distance
            elif metric == 'volume_overlap':
                score = self.compute_union_volume_overlap(union_idx, query_frustum)
            elif metric == 'volume_ratio':
                query_volume = query_frustum.get_volume()
                union_volume = union_info['volume_3d']
                volume_overlap = self.compute_union_volume_overlap(union_idx, query_frustum)
                min_volume = min(query_volume, union_volume)
                if min_volume > 1e-6:
                    score = volume_overlap / min_volume
                else:
                    score = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}. "
                               f"Supported metrics: 'overlap', 'area_ratio', 'center_distance', "
                               f"'volume_overlap', 'volume_ratio'")
            
            # volume_overlap = self.compute_union_volume_overlap(union_idx, query_frustum)
            # query_volume = query_frustum.get_volume()
            # union_volume = union_info['volume_3d']
            # min_volume = min(query_volume, union_volume)
            # volume_ratio = volume_overlap / min_volume if min_volume > 1e-6 else 0.0
            
            overlaps.append({
                'union_idx': union_idx,
                'score': score,
                'metric': metric,
                'start_idx': union_info['start_idx'],
                'end_idx': union_info['end_idx'],
                'union_metadata': self.union_metadata[union_idx],
                # 'area_overlap': self.compute_union_overlap(union_idx, query_frustum),
                # 'area_ratio': (self.compute_union_overlap(union_idx, query_frustum) / 
                #               min(query_frustum.get_projection_area_xz(), union_info['area_2d_xz']) + 1e-6)
                # if min(query_frustum.get_projection_area_xz(), union_info['area_2d_xz']) > 1e-6 else 0.0,
                # 'volume_overlap': volume_overlap,
                # 'volume_ratio': volume_ratio,
                # 'query_volume': query_volume,
                # 'union_volume': union_volume,
            })
        
        overlaps.sort(key=lambda x: x['score'], reverse=True)
        
        return overlaps[:min(k, len(overlaps))]

    
    def get_union_info(self, union_idx: int) -> Dict:
        """
        获取指定联合的详细信息。
        
        Args:
            union_idx: union索引
        
        Returns:
            包含联合详细信息的字典
        """
        return self.camera_unions[union_idx]
    
    def get_all_unions_info(self) -> List[Dict]:
        """
        获取所有联合的信息。
        
        Returns:
            所有联合的列表
        """
        return self.camera_unions
    
    def compute_multi_camera_overlap(self, union_idx: int, query_frustums: List[CameraFrustum]) -> float:
        """
        计算多个查询相机与某个联合的2D投影重叠面积。
        
        Args:
            union_idx: union索引
            query_frustums: 查询相机视锥体列表
        
        Returns:
            所有查询相机与union的总重叠面积
        """
        total_overlap = 0.0
        
        for query_frustum in query_frustums:
            overlap = self.compute_union_overlap(union_idx, query_frustum)
            total_overlap += overlap
        
        return total_overlap
    
    def compute_multi_camera_volume_overlap(self, union_idx: int, query_frustums: List[CameraFrustum]) -> float:
        """
        计算多个查询相机与某个联合的3D体积重叠量。
        
        Args:
            union_idx: union索引
            query_frustums: 查询相机视锥体列表
        
        Returns:
            所有查询相机与union的总体积重叠量
        """
        total_volume_overlap = 0.0
        
        for query_frustum in query_frustums:
            volume_overlap = self.compute_union_volume_overlap(union_idx, query_frustum)
            total_volume_overlap += volume_overlap
        
        return total_volume_overlap
    
    def compute_multi_camera_union_overlap(self, union_idx: int, query_c2w_list: List[torch.Tensor], 
                                          query_fxfycxcy_list: List[torch.Tensor]) -> Tuple[float, float]:
        """
        计算多个查询相机构成的联合与某个union的重叠面积（在XZ平面投影）。
        这种方法考虑查询相机之间可能存在的重叠。
        
        Args:
            union_idx: union索引
            query_c2w_list: 查询相机的相机到世界矩阵列表
            query_fxfycxcy_list: 查询相机的内参列表
        
        Returns:
            (query_union_overlap, union_area) 元组
        """
        query_frustums = [CameraFrustum(c2w, fxfycxcy, self.depth_range) 
                         for c2w, fxfycxcy in zip(query_c2w_list, query_fxfycxcy_list)]
        
        all_query_vertices_xz = []
        for frustum in query_frustums:
            query_vertices_xz = frustum.get_frustum_vertices()[:, [0, 2]]
            all_query_vertices_xz.append(query_vertices_xz)
        
        all_query_vertices_xz = np.vstack(all_query_vertices_xz)
        
        union_vertices_xz = self.camera_unions[union_idx]['vertices'][:, [0, 2]]
        
        try:
            query_union_hull = scipy.spatial.ConvexHull(all_query_vertices_xz)
            query_union_area = query_union_hull.volume
            
            try:
                union_hull = scipy.spatial.ConvexHull(union_vertices_xz)
                union_area = union_hull.volume
            except:
                union_area = self.camera_unions[union_idx]['area_2d_xz']
            
            all_vertices = np.vstack([union_vertices_xz, all_query_vertices_xz])
            combined_hull = scipy.spatial.ConvexHull(all_vertices)
            combined_area = combined_hull.volume
            
            intersection_area = query_union_area + union_area - combined_area
            intersection_area = max(0, intersection_area)
            
            return float(intersection_area), float(union_area)
        
        except:
            return 0.0, self.camera_unions[union_idx]['area_2d_xz']
    
    def compute_multi_camera_union_volume_overlap(self, union_idx: int, query_c2w_list: List[torch.Tensor], 
                                                 query_fxfycxcy_list: List[torch.Tensor]) -> Tuple[float, float]:
        """
        计算多个查询相机构成的联合与某个union的3D体积重叠量。
        
        Args:
            union_idx: union索引
            query_c2w_list: 查询相机的相机到世界矩阵列表
            query_fxfycxcy_list: 查询相机的内参列表
        
        Returns:
            (volume_overlap, union_volume) 元组
        """
        query_frustums = [CameraFrustum(c2w, fxfycxcy, self.depth_range) 
                         for c2w, fxfycxcy in zip(query_c2w_list, query_fxfycxcy_list)]
        
        all_query_vertices = []
        for frustum in query_frustums:
            query_vertices = frustum.get_frustum_vertices()
            all_query_vertices.append(query_vertices)
        
        all_query_vertices = np.vstack(all_query_vertices)
        
        union_vertices = self.camera_unions[union_idx]['vertices']
        
        try:
            unique_query_vertices = np.unique(all_query_vertices, axis=0)
            if len(unique_query_vertices) < 4:
                return 0.0, self.camera_unions[union_idx]['volume_3d']
            
            query_union_hull = scipy.spatial.ConvexHull(unique_query_vertices)
            query_union_volume = query_union_hull.volume
            
            union_volume = self.camera_unions[union_idx]['volume_3d']
            
            all_vertices = np.vstack([union_vertices, all_query_vertices])
            unique_vertices = np.unique(all_vertices, axis=0)
            
            if len(unique_vertices) < 4:
                return 0.0, union_volume
            
            combined_hull = scipy.spatial.ConvexHull(unique_vertices)
            combined_volume = combined_hull.volume
            
            intersection_volume = query_union_volume + union_volume - combined_volume
            intersection_volume = max(0, intersection_volume)
            
            return float(intersection_volume), float(union_volume)
        
        except:
            return 0.0, self.camera_unions[union_idx]['volume_3d']
    
    def retrieve_top_k_unions_multi_cams(self, query_c2w_list: List[torch.Tensor], 
                                        query_fxfycxcy_list: List[torch.Tensor],
                                        k: int = 3, metric: str = 'volume_ratio') -> List[Dict]:
        """
        从所有联合中检索Top-K个与查询相机集合重叠最大的联合。
        
        Args:
            query_c2w_list: 查询相机的相机到世界矩阵列表 (list of (4,4) tensors)
            query_fxfycxcy_list: 查询相机的内参列表 (list of (4,) tensors)
            k: 返回前K个结果
            metric: 排序指标，可选：
                   '2d' - 2D投影重叠面积总和
                   '2d_ratio' - 2D投影重叠比例
                   '3d' - 3D体积重叠总和
                   '3d_ratio' - 3D体积重叠比例
                   'mean_2d' - 平均2D重叠
                   'mean_3d' - 平均3D重叠
        
        Returns:
            包含top-k联合信息的列表
        """
        if not query_c2w_list or not query_fxfycxcy_list:
            raise ValueError("查询相机列表不能为空")
        
        if len(query_c2w_list) != len(query_fxfycxcy_list):
            raise ValueError("查询相机的c2w和fxfycxcy数量必须相同")
        
        num_query_cameras = len(query_c2w_list)
        
        overlaps = []
        for union_idx, union_info in enumerate(self.camera_unions):
            area_overlap, union_area = self.compute_multi_camera_union_overlap(
                union_idx, query_c2w_list, query_fxfycxcy_list
            )
            
            volume_overlap, union_volume = self.compute_multi_camera_union_volume_overlap(
                union_idx, query_c2w_list, query_fxfycxcy_list
            )
            
            query_frustums = [CameraFrustum(c2w, fxfycxcy, self.depth_range) 
                            for c2w, fxfycxcy in zip(query_c2w_list, query_fxfycxcy_list)]
            
            all_query_vertices = []
            for frustum in query_frustums:
                all_query_vertices.append(frustum.get_frustum_vertices())
            
            all_query_vertices = np.vstack(all_query_vertices)
            unique_query_vertices = np.unique(all_query_vertices, axis=0)
            
            if len(unique_query_vertices) >= 4:
                try:
                    query_union_hull = scipy.spatial.ConvexHull(unique_query_vertices)
                    query_union_volume = query_union_hull.volume
                except:
                    query_union_volume = 1.0
            else:
                query_union_volume = 1.0
            
            if metric == '2d':
                score = area_overlap
            elif metric == '2d_ratio':
                score = area_overlap / (min(query_union_volume / 10, union_area) + 1e-6)
            elif metric == '3d':
                score = volume_overlap
            elif metric == '3d_ratio':
                score = volume_overlap / (min(query_union_volume, union_volume) + 1e-6)
            elif metric == 'mean_2d':
                total_2d = 0.0
                for i, (c2w, fxfycxcy) in enumerate(zip(query_c2w_list, query_fxfycxcy_list)):
                    frustum = CameraFrustum(c2w, fxfycxcy, self.depth_range)
                    total_2d += self.compute_union_overlap(union_idx, frustum)
                score = total_2d / num_query_cameras
            elif metric == 'mean_3d':
                total_3d = 0.0
                for i, (c2w, fxfycxcy) in enumerate(zip(query_c2w_list, query_fxfycxcy_list)):
                    frustum = CameraFrustum(c2w, fxfycxcy, self.depth_range)
                    total_3d += self.compute_union_volume_overlap(union_idx, frustum)
                score = total_3d / num_query_cameras
            else:
                raise ValueError(f"Unknown metric: {metric}. "
                               f"Supported metrics: '2d', '2d_ratio', '3d', '3d_ratio', 'mean_2d', 'mean_3d'")
            
            overlaps.append({
                'union_idx': union_idx,
                'score': score,
                'metric': metric,
                'start_idx': union_info['start_idx'],
                'end_idx': union_info['end_idx'],
                'union_metadata': self.union_metadata[union_idx],
                'area_overlap': area_overlap,
                'area_ratio': area_overlap / (union_area + 1e-6) if union_area > 1e-6 else 0.0,
                'volume_overlap': volume_overlap,
                'volume_ratio': volume_overlap / (union_volume + 1e-6) if union_volume > 1e-6 else 0.0,
                'query_union_volume': query_union_volume,
                'union_volume': union_volume,
                'num_query_cameras': num_query_cameras,
            })
        
        overlaps.sort(key=lambda x: x['score'], reverse=True)
        
        return overlaps[:min(k, len(overlaps))]


def create_camera_union_analyzer(c2w_sequence: torch.Tensor, 
                                 fxfycxcy_sequence: torch.Tensor,
                                 union_size: int = 5,
                                 depth_range: Tuple[float, float] = (0.1, 4.0)) -> CameraUnionAnalyzer:
    """
    便捷函数：从相机序列创建CameraUnionAnalyzer。
    
    Args:
        c2w_sequence: (N, 4, 4) 相机到世界矩阵序列
        fxfycxcy_sequence: (N, 4) 相机内参序列
        union_size: 每个union的相机数量
        depth_range: 视锥体深度范围
    
    Returns:
        CameraUnionAnalyzer实例
    """
    camera_list = []
    for i in range(c2w_sequence.shape[0]):
        c2w = c2w_sequence[i]
        fxfycxcy = fxfycxcy_sequence[i]
        camera_list.append((c2w, fxfycxcy))
    
    return CameraUnionAnalyzer(camera_list, union_size, depth_range = depth_range)

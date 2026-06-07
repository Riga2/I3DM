import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class UnifiedDatasetRe10K_Keyframes(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path=None,
        caption_path=None,
        repeat=1,
        config=None,
        **kwargs,
    ):
        self.metadata_path = metadata_path
        self.caption_path = caption_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        self.config = config
        self.image_size_H = 352
        self.image_size_W = 640
        self.num_target_views = 77
        self.num_input_views = 4
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)
        self.extraRate = 0.1
        self.data = []
        self.caption_dict = {}
        self.load_metadata(metadata_path)

    def load_metadata(self, metadata_path):
        if metadata_path is None:
            raise ValueError("metadata_path is required for UnifiedDatasetRe10K_Keyframes.")
        with open(metadata_path, "r") as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]

        if self.caption_path is None:
            raise ValueError("caption_path is required for UnifiedDatasetRe10K_Keyframes.")
        with open(self.caption_path, "r", encoding="utf-8") as f:
            self.caption_dict = json.load(f)

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        target_w_resize = 640
        target_h_resize = 360
        target_h_crop = 352
        images = []
        intrinsics = []

        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            aspect_ratio = original_image_w / original_image_h
            if abs(aspect_ratio - (16 / 9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)

            current_w, current_h = image.size
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            image = image.crop((start_w, start_h, start_w + target_w_resize, start_h + target_h_crop))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = target_w_resize / original_image_w
            resize_ratio_y = target_h_resize / original_image_h
            fxfycxcy *= np.array([resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y])
            fxfycxcy[2] -= start_w
            fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()

            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = torch.from_numpy(np.linalg.inv(w2cs)).float()
        return images, intrinsics, c2ws

    def preprocess_poses(self, in_c2ws: torch.Tensor, scene_scale_factor=1.35):
        """
        Align poses to the average camera frame, then rescale the scene to a fixed range.
        """
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
        avg_down = in_c2ws[:, :3, 1].mean(0)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

        avg_pose = torch.eye(4, device=in_c2ws.device)
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_pose)
        in_c2ws = avg_pose @ in_c2ws

        scene_scale = scene_scale_factor * torch.max(torch.abs(in_c2ws[:, :3, 3]))
        in_c2ws[:, :3, 3] /= scene_scale
        return in_c2ws

    def _sample_another(self):
        return self.__getitem__(random.randint(0, len(self) - 1))

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]
        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            with open(scene_path.strip(), "r") as f:
                target_data_json = json.load(f)
            target_frames = target_data_json["frames"][-self.num_target_views:]
            if len(target_frames) < self.num_target_views:
                return self._sample_another()

            target_image_paths_chosen = [frame["image_path"] for frame in target_frames]
            target_frames_chosen = list(target_frames)
            clip_name = os.path.basename(scene_path).split(".")[0]
            scene_name = clip_name.split("_")[0]

            if self.extraRate > 0 and random.random() < self.extraRate:
                input_image_paths_chosen = [target_image_paths_chosen[0]] * self.num_input_views
                input_frames_chosen = [target_frames_chosen[0]] * self.num_input_views
            else:
                # you need to ensure the clips_metadata_F77 and metadata directories are in the same base path
                dir_name = os.path.dirname(scene_path.replace("clips_metadata_F77", "metadata"))
                all_video_frames_scene_path = os.path.join(dir_name, f"{scene_name}.json")
                with open(all_video_frames_scene_path.strip(), "r") as f:
                    all_data_json = json.load(f)
                all_video_frames = all_data_json["frames"]
                input_indices = random.sample(list(range(len(all_video_frames))), self.num_input_views)
                input_image_paths_chosen = [all_video_frames[idx]["image_path"] for idx in input_indices]
                input_frames_chosen = [all_video_frames[idx] for idx in input_indices]

            selected_image_paths = input_image_paths_chosen + target_image_paths_chosen
            selected_frames = input_frames_chosen + target_frames_chosen
            all_images, all_intrinsics, all_c2ws = self.preprocess_frames(selected_frames, selected_image_paths)
            if all_images is None:
                return self._sample_another()

            target_c2ws_for_nvs = all_c2ws[self.num_input_views:][::4]
            target_intrinsics_for_nvs = all_intrinsics[self.num_input_views:][::4]
            target_img_for_nvs = all_images[self.num_input_views:][::4]

            all_nvs_c2ws = torch.cat((all_c2ws[:self.num_input_views], target_c2ws_for_nvs), dim=0)
            c2ws_for_nvs_processed = self.preprocess_poses(all_nvs_c2ws, self.scene_scale_factor)
            caption = self.caption_dict[clip_name] if clip_name in self.caption_dict else ""

            return {
                "input_nvs_image": all_images[:self.num_input_views],
                "input_nvs_c2w": c2ws_for_nvs_processed[:self.num_input_views],
                "input_nvs_fxfycxcy": all_intrinsics[:self.num_input_views],
                "target_nvs_image": target_img_for_nvs,
                "target_nvs_c2w": c2ws_for_nvs_processed[self.num_input_views:],
                "target_nvs_fxfycxcy": target_intrinsics_for_nvs,
                "scene_name": scene_name,
                "clip_name": clip_name,
                "prompt": caption,
                "target_src_image": all_images[self.num_input_views:],
                "target_src_c2w": all_c2ws[self.num_input_views:],
                "target_src_fxfycxcy": all_intrinsics[self.num_input_views:],
                "ref_img": all_images[self.num_input_views],
            }
        except Exception as exc:
            print(f"Error loading data id {data_id}, path: {scene_path}: {exc}")
            return self._sample_another()

    def __len__(self):
        return len(self.data) * self.repeat

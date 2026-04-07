import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image



class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)

class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True

import random
import numpy as np
import torch.nn.functional as F
import math

class UnifiedDatasetRe10K(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, 
        metadata_path=None,
        repeat=1,
        config=None,
        num_views=5,
        **kwargs
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size = 256
        self.patch_size = 16
        self.num_input_views = 4
        self.num_target_views = num_views
        self.min_frame_dist = kwargs.get("min_frame_dist", 30)
        self.max_frame_dist = kwargs.get("max_frame_dist", 50)
        self.square_crop = kwargs.get("square_crop", True)
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)
        self.num_views = self.num_input_views + self.num_target_views

        self.load_metadata(metadata_path)
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption', 'video_captions_refined.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
            self.caption_dict = json.load(f)

    def view_selector(self, frames):
        total_frames = len(frames)
        if total_frames < self.num_views:
            return None
            
        min_dist = self.min_frame_dist
        max_dist = min(total_frames - 1, self.max_frame_dist)
        
        if max_dist <= min_dist:
            frame_dist = total_frames - 1
        else:
            frame_dist = random.randint(min_dist, max_dist)

        start_frame = random.randint(0, total_frames - frame_dist - 1)
        end_frame = start_frame + frame_dist

        valid_target_start_max = end_frame - self.num_target_views + 1
        target_start_idx = random.randint(start_frame, valid_target_start_max - 1)
        target_index = list(range(target_start_idx, target_start_idx + self.num_target_views))

        # candidates = [v for v in range(start_frame, end_frame) if v not in target_index]
        candidates = [v for v in range(start_frame, end_frame)]
        input_index = sorted(random.sample(candidates, self.num_input_views))

        return input_index + target_index

    def view_selector_skip(self, frames):
        total_frames = len(frames)
        if total_frames < self.num_views:
            return None

        max_skip = min(8, total_frames // self.num_views)
        skip = random.randint(1, max_skip)

        valid_index = list(range(0, len(frames), skip))
        cur_len = len(valid_index)

        min_dist = max(self.num_views, self.min_frame_dist)
        max_dist = min(cur_len - 1, self.max_frame_dist)
        
        if max_dist <= min_dist:
            frame_dist = cur_len - 1
        else:
            frame_dist = random.randint(min_dist, max_dist)

        start_frame = random.randint(0, cur_len - frame_dist - 1)
        end_frame = start_frame + frame_dist

        valid_target_start_max = end_frame - self.num_target_views + 1
        target_start_idx = random.randint(start_frame, valid_target_start_max)
        target_index = list(range(target_start_idx, target_start_idx + self.num_target_views))

        candidates = [v for v in range(start_frame, end_frame)]
        input_index = sorted(random.sample(candidates, self.num_input_views))

        input_frames = [valid_index[i] for i in input_index]
        target_frames = [valid_index[i] for i in target_index]
        return input_frames + target_frames
    
    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.image_size
        patch_size = self.patch_size
        square_crop = self.square_crop

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            if resize_h != original_image_h:
                image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
                if square_crop:
                    min_size = min(resize_h, resize_w)
                    start_h = (resize_h - min_size) // 2
                    start_w = (resize_w - min_size) // 2
                    image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            
            if square_crop and (resize_h != original_image_h):
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws
    
    def preprocess_single_frames(self, image_path_chosen):
        resize_h = self.image_size
        patch_size = self.patch_size
        square_crop = self.square_crop

        image = Image.open(image_path_chosen)
        original_image_w, original_image_h = image.size
            
        resize_w = int(resize_h / original_image_h * original_image_w)
        resize_w = int(round(resize_w / patch_size) * patch_size)

        if resize_h != original_image_h:
            image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image

    def preprocess_poses(self, in_c2ws, scene_scale_factor=1.35):
        # Translation and Rotation
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

        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        if scene_scale > 1e-6:
             scene_scale = scene_scale_factor * scene_scale
             in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            
            frames = data_json["frames"]
            scene_name = data_json["scene_name"]
            image_indices = self.view_selector_skip(frames)
            
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]
            
            input_images, input_intrinsics, input_src_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
            input_c2ws = self.preprocess_poses(input_src_c2ws, self.scene_scale_factor)

            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)
            scene_indices = torch.full_like(image_indices, data_id)
            indices = torch.cat([image_indices, scene_indices], dim=-1)

            caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

            ref_img = self.preprocess_single_frames(frames[0]["image_path"])

            return {
                "image": input_images,
                "c2w": input_c2ws,
                "fxfycxcy": input_intrinsics,
                "index": indices,
                "scene_name": scene_name,
                "src_c2w": input_src_c2ws,
                "prompt": caption,
                "ref_img": ref_img
            }

        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    

class UnifiedDatasetRe10K_Token(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path=None,
        repeat=1,
        config=None,
        precomputed_token=False,
        precomputed_token_dir=None,
        **kwargs
    ):
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.image_size_W = 640
        self.patch_size = 16
        self.num_frames_per_token = 11
        self.num_input_tokens = 1
        self.num_target_tokens = 7
        assert self.num_frames_per_token * self.num_target_tokens % 4 == 1, "num_frames_per_token * num_target_tokens must satisfy 4n+1."
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)

        self.num_target_views = self.num_target_tokens * self.num_frames_per_token
        self.num_input_views = self.num_input_tokens * self.num_frames_per_token
            
        self.load_metadata(metadata_path)

        if precomputed_token:
            self.input_token_paths = {}
            for file_name in os.listdir(precomputed_token_dir):
                if file_name.endswith(".pt"):
                    self.input_token_paths[file_name.split('.')[0]] = os.path.join(precomputed_token_dir, file_name)
            print(f"{len(self.input_token_paths)} precomputed token files found.")
        else:
            self.input_token_paths = None
            
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F121', 'video_captions_refined.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
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
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
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
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        
        return images, intrinsics, c2ws

    def preprocess_frames_only(self, frames_chosen, image_paths_chosen):
        target_w_resize = 640
        target_h_resize = 360
        target_h_crop = 352
        
        images = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            aspect_ratio = original_image_w / original_image_h
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
            image = image.crop((start_w, start_h, start_w + target_w_resize, start_h + target_h_crop))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            images.append(image)

        images = torch.stack(images, dim=0)
        
        return images

    def preprocess_input_poses(
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
        avg_pose_inv = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose_inv @ in_c2ws 

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws, avg_pose, avg_pose_inv
    
    def preprocess_target_poses(
        self,
        avg_pose_inv: torch.Tensor,
        target_c2ws_src: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the target poses using the average pose and scene scale from input poses.
        """

        # Apply the same translation and rotation
        target_c2ws = avg_pose_inv @ target_c2ws_src 

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(target_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        target_c2ws[:, :3, 3] /= scene_scale

        return target_c2ws

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            
            frames = data_json["frames"]
            scene_name = os.path.basename(scene_path).split(".")[0]
                
            if len(frames) < (self.num_input_views + self.num_target_views):
                return self.__getitem__(random.randint(0, len(self) - 1))

            target_indices = list(range(len(frames) - self.num_target_views, len(frames)))
            input_indices = random.sample(list(range(0, len(frames) - self.num_target_views)), self.num_input_views)
    
            image_indices = input_indices + target_indices
            image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]
            
            all_images, all_intrinsics, all_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
            if all_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            scene_scale_factor = 1.35
            # centerize and scale the poses (for unbounded scenes)
            input_c2ws_for_nvs_processed, avg_input_c2w, avg_input_w2c = self.preprocess_input_poses(all_c2ws[:self.num_input_views], scene_scale_factor)

            # choose 4n+1 target c2w and intrinsic for each token, 4n+1=self.num_target_views
            target_c2ws_for_nvs = all_c2ws[self.num_input_views:][::4]
            target_intrinsics_for_nvs = all_intrinsics[self.num_input_views:][::4]
            target_img_for_nvs = all_images[self.num_input_views:][::4]

            target_c2ws_for_nvs_processed = self.preprocess_target_poses(avg_input_w2c, target_c2ws_for_nvs, scene_scale_factor)

            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)
            scene_indices = torch.full_like(image_indices, data_id)
            indices = torch.cat([image_indices, scene_indices], dim=-1)

            caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

            return {
                "input_nvs_image": all_images[:self.num_input_views],                 # [self.num_input_views, 3, H, W]
                "input_nvs_c2w":   input_c2ws_for_nvs_processed,                      # [self.num_input_views, 4, 4]
                "input_nvs_fxfycxcy": all_intrinsics[:self.num_input_views],          # [self.num_input_views, 4]

                "target_nvs_image": target_img_for_nvs,                 # [n, 3, H, W]
                "target_nvs_c2w": target_c2ws_for_nvs_processed,        # [n, 4, 4]
                "target_nvs_fxfycxcy": target_intrinsics_for_nvs,       # [n, 4]

                "index": indices,
                "scene_name": scene_name,
                "prompt": caption,
                "target_src_image": all_images[self.num_input_views:],  # [self.num_target_views, 3, H, W]
                "target_src_c2w": all_c2ws[self.num_input_views:],
                "target_src_fxfycxcy": all_intrinsics[self.num_input_views:],
                "ref_img": all_images[self.num_input_views],
            }
            
        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
            
    def __getitem__token(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            
            frames = data_json["frames"]
            scene_name = os.path.basename(scene_path).split(".")[0]

            if self.input_token_paths is not None and scene_name in self.input_token_paths:
                token_path = self.input_token_paths[scene_name]
                token_data = torch.load(token_path, weights_only=True)

                self.num_target_views = self.num_target_tokens * self.num_frames_per_token

                target_indices = list(range(len(frames) - self.num_target_views, len(frames)))
                image_paths_chosen = [frames[ic]["image_path"] for ic in target_indices]
                frames_chosen = [frames[ic] for ic in target_indices]
    
                target_images, target_intrinsics, target_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
                if target_images is None:
                    return self.__getitem__(random.randint(0, len(self) - 1))
                
                # input_indices = list(range(0, len(frames) - self.num_target_views))
                # input_image_paths_chosen = [frames[ic]["image_path"] for ic in input_indices]
                # input_frames_chosen = [frames[ic] for ic in input_indices]
                # input_images = self.preprocess_frames_only(input_frames_chosen, input_image_paths_chosen)

                scene_scale_factor = 1.35
                # choose 4n+1 target c2w and intrinsic for each token, 4n+1=self.num_target_views
                target_c2ws_for_nvs = target_c2ws[::4]
                target_intrinsics_for_nvs = target_intrinsics[::4]
                target_img_for_nvs = target_images[::4]

                if self.num_input_tokens == 1:
                    selected_token_id = random.randint(0, token_data['token_c2w'].shape[0] - 1)
                    token_w2c = torch.inverse(token_data['token_c2w'][selected_token_id])
                    target_c2ws_for_nvs_processed = self.preprocess_target_poses(token_w2c, target_c2ws_for_nvs, scene_scale_factor)
                    
                    input_scene_token = token_data['scene_tokens'][selected_token_id].unsqueeze(0)
                    input_token_c2w = token_data['token_c2w'][selected_token_id].unsqueeze(0)
                else:
                    target_c2ws_for_nvs_processed = []                    
                    for i in range(self.num_input_tokens):
                        token_w2c = torch.inverse(token_data['token_c2w'][i])
                        cur_token_target_c2w = self.preprocess_target_poses(token_w2c, target_c2ws_for_nvs, scene_scale_factor)
                        target_c2ws_for_nvs_processed.append(cur_token_target_c2w)
                    target_c2ws_for_nvs_processed = torch.cat(target_c2ws_for_nvs_processed, dim=0) # [self.num_input_tokens * n, 4, 4]
                    
                    input_scene_token = token_data['scene_tokens']
                    input_token_c2w = token_data['token_c2w']

                target_indices = torch.tensor(target_indices).long().unsqueeze(-1)
                scene_indices = torch.full_like(target_indices, data_id)
                indices = torch.cat([target_indices, scene_indices], dim=-1)

                caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

                return {
                    "target_nvs_image": target_img_for_nvs,                 # [n, 3, H, W]
                    "target_nvs_c2w": target_c2ws_for_nvs_processed,        # [self.num_input_tokens * n, 4, 4]
                    "target_nvs_fxfycxcy": target_intrinsics_for_nvs,       # [n, 4]

                    # "input_nvs_image": input_images,                             # [self.num_input_views, 3, H, W] only for debug
                    "input_scene_token": input_scene_token,
                    "input_token_c2w": input_token_c2w,                       

                    "index": indices,
                    "scene_name": scene_name,
                    "prompt": caption,
                    "target_src_image": target_images,                      # [self.num_target_views, 3, H, W]
                    "target_src_c2w": target_c2ws,
                    "target_src_fxfycxcy": target_intrinsics,
                    "ref_img": target_images[0],
                }
            else:
                print(f"No precomputed token for scene: {scene_name}")
                return self.__getitem__(random.randint(0, len(self) - 1))        

        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def __getitem__old(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            
            frames = data_json["frames"]
            scene_name = os.path.basename(scene_path).split(".")[0]
                
            if len(frames) < (self.num_input_tokens + self.num_target_tokens) * self.num_frames_per_token:
                return self.__getitem__(random.randint(0, len(self) - 1))

            self.num_target_views = self.num_target_tokens * self.num_frames_per_token
            self.num_input_views = self.num_input_tokens * self.num_frames_per_token

            target_indices = list(range(len(frames) - self.num_target_views, len(frames)))
            total_num_tokens = len(frames) // self.num_frames_per_token
            if total_num_tokens - self.num_target_tokens > self.num_input_tokens:
                selected_input_token_ids = random.sample(range(total_num_tokens - self.num_target_tokens), self.num_input_tokens)
                input_indices = [i * self.num_frames_per_token + j for i in selected_input_token_ids for j in range(self.num_frames_per_token)]
            else:
                input_indices = list(range(0, len(frames) - self.num_target_views))
    
            image_indices = input_indices + target_indices
            # image_indices = list(range(len(frames) - 88,  len(frames)))
            image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]
            
            all_images, all_intrinsics, all_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
            if all_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            # centerize and scale the poses (for unbounded scenes)
            # all_input_c2ws, _, ctx_avg_pose = self.preprocess_input_poses(all_c2ws[:11], scene_scale_factor)
            # target_c2ws_processed = self.preprocess_target_poses(ctx_avg_pose, all_c2ws[11:], scene_scale_factor)
            # all_input_images = all_images[:11] # [self.num_input_views, 3, H, W]
            # target_images = all_images[11:][::4] # [self.num_target_views, 3, H, W]
            # target_c2ws_processed = target_c2ws_processed[::4] # [self.num_target_views, 4, 4]
            # all_input_intrinsics = all_intrinsics[:11] # [self.num_input_views,
            # target_intrinsics = all_intrinsics[11:][::4] # [self.num_target_views, 4]
            
            scene_scale_factor = 1.35
            input_c2ws = []
            input_token_w2cs = []

            input_token_avg_c2ws = []
            input_token_avg_fxfycxcy = []

            all_input_images = []
            all_input_intrinsics = []
            for i in range(self.num_input_tokens):
                # centerize and scale the poses (for unbounded scenes)
                cur_token_c2w = all_c2ws[i * self.num_frames_per_token:(i + 1) * self.num_frames_per_token]
                cur_frames_input_c2ws, cur_token_avg_c2w, cur_token_w2c = self.preprocess_input_poses(cur_token_c2w, scene_scale_factor)

                input_token_w2cs.append(cur_token_w2c)
                input_c2ws.append(cur_frames_input_c2ws)
                all_input_images.append(all_images[i * self.num_frames_per_token:(i + 1) * self.num_frames_per_token])
                all_input_intrinsics.append(all_intrinsics[i * self.num_frames_per_token:(i + 1) * self.num_frames_per_token])

                input_token_avg_c2ws.append(cur_token_avg_c2w)
                input_token_avg_fxfycxcy.append(all_intrinsics[i * self.num_frames_per_token:(i + 1) * self.num_frames_per_token].mean(dim=0))

            # all_input_images = all_images[:self.num_input_views] # [self.num_input_views, 3, H, W]
            # all_input_intrinsics = all_intrinsics[:self.num_input_views] # [self.num_input_views, 4]
            all_input_c2ws = torch.cat(input_c2ws, dim=0)
            all_input_images = torch.cat(all_input_images, dim=0)
            all_input_intrinsics = torch.cat(all_input_intrinsics, dim=0)

            # token infos
            input_token_avg_c2ws = torch.stack(input_token_avg_c2ws, dim=0) # [self.num_input_tokens, 4, 4]
            input_token_avg_fxfycxcy = torch.stack(input_token_avg_fxfycxcy, dim=0) # [self.num_input_tokens, 4]

            # choose 4n+1 target c2w and intrinsic for each token, 4n+1=self.num_target_views
            target_c2ws_for_nvs = all_c2ws[self.num_input_views:][::4]
            target_intrinsics_for_nvs = all_intrinsics[self.num_input_views:][::4]
            target_img_for_nvs = all_images[self.num_input_views:][::4]

            target_c2ws_for_nvs_processed = []
            for i in range(self.num_input_tokens):
                cur_token_target_c2w = self.preprocess_target_poses(input_token_w2cs[i], target_c2ws_for_nvs, scene_scale_factor)
                target_c2ws_for_nvs_processed.append(cur_token_target_c2w)
            
            target_c2ws_for_nvs_processed = torch.cat(target_c2ws_for_nvs_processed, dim=0) # [self.num_input_tokens * n, 4, 4]

            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)
            scene_indices = torch.full_like(image_indices, data_id)
            indices = torch.cat([image_indices, scene_indices], dim=-1)

            caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

            return {
                "input_nvs_image": all_input_images,                    # [self.num_input_views, 3, H, W]
                "input_nvs_c2w": all_input_c2ws,                        # [self.num_input_views, 4, 4]
                "input_nvs_fxfycxcy": all_input_intrinsics,             # [self.num_input_views, 4]

                "target_nvs_image": target_img_for_nvs,                 # [n, 3, H, W]
                "target_nvs_c2w": target_c2ws_for_nvs_processed,        # [self.num_input_tokens * n, 4, 4]
                "target_nvs_fxfycxcy": target_intrinsics_for_nvs,       # [n, 4]

                "token_avg_c2w": input_token_avg_c2ws,                  # [self.num_input_tokens, 4, 4]
                "token_avg_fxfycxcy": input_token_avg_fxfycxcy,         # [self.num_input_tokens, 4]

                "index": indices,
                "scene_name": scene_name,
                "prompt": caption,
                "target_src_image": all_images[-self.num_target_views:],  # [self.num_target_views, 3, H, W]
                "target_src_c2w": all_c2ws[-self.num_target_views:],
                "target_src_fxfycxcy": all_intrinsics[-self.num_target_views:],
                "ref_img": all_images[-self.num_target_views],
            }

            # return {
            #     "image": torch.cat([all_input_images, target_images], dim=0),   # [self.num_input_views + self.num_target_views, 3, H, W]
            #     "c2w": torch.cat([all_input_c2ws, target_c2ws_processed], dim=0), # [self.num_input_views + self.num_input_tokens * n, 4, 4]
            #     "fxfycxcy": torch.cat([all_input_intrinsics, target_intrinsics], dim=0), # [self.num_input_views + self.num_input_tokens * n, 4]
            #     "index": indices,
            #     "scene_name": scene_name,
            #     "prompt": caption,
            #     "target_src_c2w": all_c2ws[-self.num_target_views:],
            #     "ref_img": all_images[-self.num_target_views],
            # }

            # all_input_c2ws = []
            # token_c2ws = []
            # token_intrinsics = []
            # scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
            # for i in range(self.num_input_tokens):
            #     # centerize and scale the poses (for unbounded scenes)
            #     cur_token_c2w = input_c2ws[i * self.num_frames_per_token:(i + 1) * self.num_frames_per_token]
            #     cur_frames_input_c2ws, cur_token_avg_c2w, _ = self.preprocess_input_poses(cur_token_c2w, scene_scale_factor)
                
            #     cur_token_avg_intrinsics = input_intrinsics[i * self.num_frames_per_token:(i + 1) * self.num_frames_per_token].mean(dim=0)
                
            #     token_c2ws.append(cur_token_avg_c2w)
            #     token_intrinsics.append(cur_token_avg_intrinsics)
            #     all_input_c2ws.append(cur_frames_input_c2ws)

            # all_input_images = input_images[:self.num_input_views] # [self.num_input_views, 3, H, W]
            # all_input_intrinsics = input_intrinsics[:self.num_input_views] # [self.num_input_views, 4]
            # all_input_c2ws = torch.cat(all_input_c2ws, dim=0)   # [self.num_input_views, 4, 4]

            # target_images = input_images[self.num_input_views:]         # [self.num_target_frames, 3, H, W]
            # target_intrinsics = input_intrinsics[self.num_input_views:] # [self.num_target_frames, 4]
            # target_c2w = input_c2ws[self.num_input_views:]              # [self.num_target_frames, 4, 4]
            # token_c2ws_processed, avg_token_w2c = self.preprocess_input_poses(torch.stack(token_c2ws, dim=0), 1.0)  # [self.num_tokens, 4, 4], [4,4]
            # token_intrinsics_processed = torch.stack(token_intrinsics, dim=0)                                           # [self.num_tokens, 4]
            # target_c2ws_processed = self.preprocess_target_poses(avg_token_w2c, target_c2w, 1.0)

            # image_indices = torch.tensor(image_indices).long().unsqueeze(-1)
            # scene_indices = torch.full_like(image_indices, data_id)
            # indices = torch.cat([image_indices, scene_indices], dim=-1)

            # caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

            # return {
            #     "image": input_images,
            #     "c2w": input_c2ws, 
            #     "fxfycxcy": input_intrinsics,
            #     "index": indices,
            #     "scene_name": scene_name,
            #     "target_c2w": target_c2ws,
            #     # "target_src_c2w": input_c2ws[-self.num_target_views:],
            #     "prompt": caption,
            #     "ref_img": input_images[-self.num_target_views],
            # }

        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
    def all_getitem(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            
            frames = data_json["frames"]
            scene_name = data_json["scene_name"]
            
            if len(frames) != self.num_views:
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            image_indices = list(range(self.num_views))
            image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]
            
            input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
            if input_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
            ctx_c2ws, ctx_avg_pose = self.preprocess_input_poses(input_c2ws[:self.num_input_views], scene_scale_factor)
            target_c2ws = self.preprocess_target_poses(ctx_avg_pose, input_c2ws[self.num_input_views:], scene_scale_factor)
            input_c2ws = torch.cat([ctx_c2ws, target_c2ws], dim=0)

            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)
            scene_indices = torch.full_like(image_indices, data_id)
            indices = torch.cat([image_indices, scene_indices], dim=-1)

            caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

            return {
                "image": input_images,
                "c2w": input_c2ws, 
                "fxfycxcy": input_intrinsics,
                "index": indices,
                "scene_name": scene_name,
                "target_c2w": target_c2ws,
                # "target_src_c2w": input_c2ws[-self.num_target_views:],
                "prompt": caption,
                "ref_img": input_images[-self.num_target_views],
            }

        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def old_getitem(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            
            frames = data_json["frames"]
            scene_name = data_json["scene_name"]
            
            if len(frames) != self.num_views:
                return self.__getitem__(random.randint(0, len(self) - 1))
            
            image_indices = list(range(self.num_views))
            image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]
            
            input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)
            if input_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)

            all_input_c2ws = []
            token_c2ws = []
            for i in range(self.num_tokens):
                # centerize and scale the poses (for unbounded scenes)
                cur_token_c2w = input_c2ws[i * self.num_frames_per_token:(i + 1) * self.num_frames_per_token]
                cur_frames_input_c2ws, cur_token_avg_pose = self.preprocess_input_poses(cur_token_c2w, scene_scale_factor)
                token_c2ws.append(cur_token_avg_pose)
                all_input_c2ws.append(cur_frames_input_c2ws)
            all_input_images = input_images[:self.num_tokens * self.num_frames_per_token] # [self.num_tokens * self.num_frames_per_token, 3, H, W]
            all_input_intrinsics = input_intrinsics[:self.num_tokens * self.num_frames_per_token] # [self.num_tokens * self.num_frames_per_token, 4]
            all_input_c2ws = torch.cat(all_input_c2ws, dim=0)

            target_c2ws_for_tokens = []
            for i in range(self.num_tokens):
                cur_token_target_c2w = self.preprocess_target_poses(token_c2ws[i], input_c2ws[self.num_tokens * self.num_frames_per_token:], scene_scale_factor)
                target_c2ws_for_tokens.append(cur_token_target_c2w)
            target_images = input_images[self.num_tokens * self.num_frames_per_token:] # [self.num_target_frames, 3, H, W]
            target_intrinsics = input_intrinsics[self.num_tokens * self.num_frames_per_token:].repeat(self.num_tokens, 1) # [self.num_tokens * self.num_target_frames, 4]
            target_c2ws_processed = torch.cat(target_c2ws_for_tokens, dim=0) # [self.num_tokens * self.num_target_frames, 4, 4]

            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)
            scene_indices = torch.full_like(image_indices, data_id)
            indices = torch.cat([image_indices, scene_indices], dim=-1)

            caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

            ref_img = self.preprocess_single_frames(frames[-self.num_target_views]["image_path"])

            return {
                "image": torch.cat([all_input_images, target_images], dim=0),
                "c2w": torch.cat([all_input_c2ws, target_c2ws_processed], dim=0), 
                "fxfycxcy": torch.cat([all_input_intrinsics, target_intrinsics], dim=0),
                "index": indices,
                "scene_name": scene_name,
                "target_src_c2w": input_c2ws[-self.num_target_views:],
                "prompt": caption,
                "ref_img": ref_img
            }

        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True

class UnifiedDatasetRe10K_Keyframes(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path=None,
        repeat=1,
        config=None,
        **kwargs
    ):
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.image_size_W = 640
        self.patch_size = 16
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)

        self.num_target_views = 77
        self.num_input_views = 4
            
        self.load_metadata(metadata_path)

        self.extraRate = 0.1
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]

        # caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F121', 'video_captions_refined.json')
        # caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F121', 'video_captions_refined.json')
        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F77_new', 'video_captions_refined.json')
        # caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F77', 'video_captions.json')
        # print("Remeber to use refined captions for better results!!!!!!!!!!!!!!!!")

        with open(caption_path, 'r', encoding='utf-8') as f:
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
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
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
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        
        return images, intrinsics, c2ws

    def preprocess_frames_only(self, frames_chosen, image_paths_chosen):
        target_w_resize = 640
        target_h_resize = 360
        target_h_crop = 352
        
        images = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            aspect_ratio = original_image_w / original_image_h
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
            image = image.crop((start_w, start_h, start_w + target_w_resize, start_h + target_h_crop))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            images.append(image)

        images = torch.stack(images, dim=0)
        
        return images

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

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            # directly load target frames
            with open(scene_path.strip(), 'r') as f:
                target_data_json = json.load(f)
            target_frames = target_data_json["frames"][-self.num_target_views:]
            if len(target_frames) < self.num_target_views:
                return self.__getitem__(random.randint(0, len(self) - 1))
            target_image_paths_chosen = [target_frames[i]["image_path"] for i in range(len(target_frames))]
            target_frames_chosen = [target_frames[i] for i in range(len(target_frames))]
            
            clip_name = os.path.basename(scene_path).split(".")[0]
            scene_name = clip_name.split("_")[0]

            if self.extraRate > 0 and random.random() < self.extraRate:
                input_image_paths_chosen = [target_image_paths_chosen[0]] * self.num_input_views
                input_frames_chosen = [target_frames_chosen[0]] * self.num_input_views
            else:
                dir_name = os.path.dirname(scene_path.replace("clips_metadata_F77", "metadata"))
                # dir_name = os.path.dirname(scene_path.replace("fixed_metadata_F77", "metadata"))
                all_video_frames_scene_path = os.path.join(dir_name, f"{scene_name}.json")
                
                # select input frames from all video frames
                with open(all_video_frames_scene_path.strip(), 'r') as f:
                    all_data_json = json.load(f)
                all_video_frames = all_data_json["frames"]
                input_indices = random.sample(list(range(0, len(all_video_frames))), self.num_input_views)
                input_image_paths_chosen = [all_video_frames[ic]["image_path"] for ic in input_indices]
                input_frames_chosen = [all_video_frames[ic] for ic in input_indices]
            
            selected_image_paths = input_image_paths_chosen + target_image_paths_chosen
            selected_frames = input_frames_chosen + target_frames_chosen
            
            # load and preprocess input + target frames
            all_images, all_intrinsics, all_c2ws = self.preprocess_frames(selected_frames, selected_image_paths)
            if all_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            # choose 4n+1 target c2w and intrinsic for each token, 4n+1=self.num_target_views
            target_c2ws_for_nvs = all_c2ws[self.num_input_views:][::4]
            target_intrinsics_for_nvs = all_intrinsics[self.num_input_views:][::4]
            target_img_for_nvs = all_images[self.num_input_views:][::4]

            # preprocess nvs poses
            scene_scale_factor = 1.35
            all_nvs_c2ws = torch.cat((all_c2ws[:self.num_input_views], target_c2ws_for_nvs), dim=0)
            c2ws_for_nvs_processed = self.preprocess_poses(all_nvs_c2ws, scene_scale_factor)

            caption = self.caption_dict[clip_name] if clip_name in self.caption_dict else ""

            return {
                "input_nvs_image": all_images[:self.num_input_views],                 # [self.num_input_views, 3, H, W]
                "input_nvs_c2w":   c2ws_for_nvs_processed[:self.num_input_views],     # [self.num_input_views, 4, 4]
                "input_nvs_fxfycxcy": all_intrinsics[:self.num_input_views],          # [self.num_input_views, 4]

                "target_nvs_image": target_img_for_nvs,                                 # [n, 3, H, W]
                "target_nvs_c2w": c2ws_for_nvs_processed[self.num_input_views:],        # [n, 4, 4]
                "target_nvs_fxfycxcy": target_intrinsics_for_nvs,                       # [n, 4]

                "scene_name": scene_name,
                "clip_name": clip_name,
                "prompt": caption,
                "target_src_image": all_images[self.num_input_views:],  # [self.num_target_views, 3, H, W]
                "target_src_c2w": all_c2ws[self.num_input_views:],
                "target_src_fxfycxcy": all_intrinsics[self.num_input_views:],
                "ref_img": all_images[self.num_input_views],
            }
            
        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))  

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True  



class UnifiedDatasetRe10K_Keyframes_memPrecompute(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path=None,
        repeat=1,
        config=None,
        **kwargs
    ):
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.image_size_W = 640
        self.patch_size = 16
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)

        self.num_target_views = 77
        self.num_input_views = 4
            
        self.load_metadata(metadata_path)
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]
    
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
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
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
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        
        return images, intrinsics, c2ws

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            # directly load target frames
            with open(scene_path.strip(), 'r') as f:
                target_data_json = json.load(f)
            target_frames = target_data_json["frames"][-self.num_target_views:]
            if len(target_frames) < self.num_target_views:
                return self.__getitem__(random.randint(0, len(self) - 1))
            target_image_paths_chosen = [target_frames[i]["image_path"] for i in range(len(target_frames))]
            target_frames_chosen = [target_frames[i] for i in range(len(target_frames))]
            
            clip_name = os.path.basename(scene_path).split(".")[0]
            scene_name = clip_name.split("_")[0]

            dir_name = os.path.dirname(scene_path.replace("fixed_metadata_F77", "metadata"))
            all_video_frames_scene_path = os.path.join(dir_name, f"{scene_name}.json")
            
            # select input frames from all video frames
            with open(all_video_frames_scene_path.strip(), 'r') as f:
                all_data_json = json.load(f)
            all_video_frames = all_data_json["frames"]
            input_indices = random.sample(list(range(0, len(all_video_frames))), self.num_input_views)
            input_image_paths_chosen = [all_video_frames[ic]["image_path"] for ic in input_indices]
            input_frames_chosen = [all_video_frames[ic] for ic in input_indices]
            
            selected_image_paths = input_image_paths_chosen + target_image_paths_chosen
            selected_frames = input_frames_chosen + target_frames_chosen
            
            # load and preprocess input + target frames
            all_images, all_intrinsics, all_c2ws = self.preprocess_frames(selected_frames, selected_image_paths)
            if all_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            return {
                "input_nvs_image": all_images[:self.num_input_views],                 # [self.num_input_views, 3, H, W]
                "input_nvs_c2w":   c2ws_for_nvs_processed[:self.num_input_views],     # [self.num_input_views, 4, 4]
                "input_nvs_fxfycxcy": all_intrinsics[:self.num_input_views],          # [self.num_input_views, 4]

                "target_nvs_image": target_img_for_nvs,                                 # [n, 3, H, W]
                "target_nvs_c2w": c2ws_for_nvs_processed[self.num_input_views:],        # [n, 4, 4]
                "target_nvs_fxfycxcy": target_intrinsics_for_nvs,                       # [n, 4]

                "scene_name": scene_name,
                "clip_name": clip_name,
                "prompt": caption,
                "target_src_image": all_images[self.num_input_views:],  # [self.num_target_views, 3, H, W]
                "target_src_c2w": all_c2ws[self.num_input_views:],
                "target_src_fxfycxcy": all_intrinsics[self.num_input_views:],
                "ref_img": all_images[self.num_input_views],
            }
            
        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))  

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True  
    

class UnifiedDatasetRe10K_Keyframes_CTX(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path=None,
        repeat=1,
        config=None,
        **kwargs
    ):
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.image_size_W = 640
        self.patch_size = 16
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)

        self.num_target_views = 77
        self.num_input_views = 20
            
        self.load_metadata(metadata_path)
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F121', 'video_captions_refined.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
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
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
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
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        
        return images, intrinsics, c2ws

    def preprocess_frames_only(self, frames_chosen, image_paths_chosen):
        target_w_resize = 640
        target_h_resize = 360
        target_h_crop = 352
        
        images = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            aspect_ratio = original_image_w / original_image_h
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
            image = image.crop((start_w, start_h, start_w + target_w_resize, start_h + target_h_crop))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            images.append(image)

        images = torch.stack(images, dim=0)
        
        return images

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

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            # directly load target frames
            with open(scene_path.strip(), 'r') as f:
                target_data_json = json.load(f)
            target_frames = target_data_json["frames"][-self.num_target_views:]
            if len(target_frames) < self.num_target_views:
                return self.__getitem__(random.randint(0, len(self) - 1))
            target_image_paths_chosen = [target_frames[i]["image_path"] for i in range(len(target_frames))]
            target_frames_chosen = [target_frames[i] for i in range(len(target_frames))]
            
            clip_name = os.path.basename(scene_path).split(".")[0]
            scene_name = clip_name.split("_")[0]

            dir_name = os.path.dirname(scene_path.replace("clips_metadata_F121", "metadata"))
            all_video_frames_scene_path = os.path.join(dir_name, f"{scene_name}.json")
            
            # select input frames from all video frames
            with open(all_video_frames_scene_path.strip(), 'r') as f:
                all_data_json = json.load(f)
            all_video_frames = all_data_json["frames"]
            input_indices = random.sample(list(range(0, len(all_video_frames))), self.num_input_views)
            input_image_paths_chosen = [all_video_frames[ic]["image_path"] for ic in input_indices]
            input_frames_chosen = [all_video_frames[ic] for ic in input_indices]
            
            selected_image_paths = input_image_paths_chosen + target_image_paths_chosen
            selected_frames = input_frames_chosen + target_frames_chosen
            
            # load and preprocess input + target frames
            all_images, all_intrinsics, all_c2ws = self.preprocess_frames(selected_frames, selected_image_paths)
            if all_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            caption = self.caption_dict[clip_name] if clip_name in self.caption_dict else ""

            return {
                "input_ctx_image": all_images[:self.num_input_views],                 # [self.num_input_views, 3, H, W]
                "input_ctx_c2w":   all_c2ws[:self.num_input_views],     # [self.num_input_views, 4, 4]
                "input_ctx_fxfycxcy": all_intrinsics[:self.num_input_views],          # [self.num_input_views, 4]

                "scene_name": scene_name,
                "clip_name": clip_name,
                "prompt": caption,

                "target_src_image": all_images[self.num_input_views:],  # [self.num_target_views, 3, H, W]
                "target_src_c2w": all_c2ws[self.num_input_views:],
                "target_src_fxfycxcy": all_intrinsics[self.num_input_views:],
                
                "ref_img": all_images[self.num_input_views],
            }
            
        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))  

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True  

class UnifiedDatasetRe10K_Keyframes_NVS(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path=None,
        repeat=1,
        config=None,
        **kwargs
    ):
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.image_size_W = 640
        self.patch_size = 16
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)

        self.num_target_views = 77
        self.num_input_views = 4
            
        self.load_metadata(metadata_path)
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F77_new', 'video_captions_refined.json')

        with open(caption_path, 'r', encoding='utf-8') as f:
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
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
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
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        
        return images, intrinsics, c2ws

    def preprocess_frames_only(self, frames_chosen, image_paths_chosen):
        target_w_resize = 640
        target_h_resize = 360
        target_h_crop = 352
        
        images = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            aspect_ratio = original_image_w / original_image_h
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
            image = image.crop((start_w, start_h, start_w + target_w_resize, start_h + target_h_crop))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            images.append(image)

        images = torch.stack(images, dim=0)
        
        return images

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

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            # directly load target frames
            with open(scene_path.strip(), 'r') as f:
                target_data_json = json.load(f)
            target_frames = target_data_json["frames"]
            if len(target_frames) < self.num_target_views:
                return self.__getitem__(random.randint(0, len(self) - 1))
            target_image_paths_chosen = [target_frames[i]["image_path"] for i in range(len(target_frames))]
            target_frames_chosen = [target_frames[i] for i in range(len(target_frames))]
            
            clip_name = os.path.basename(scene_path).split(".")[0]
            scene_name = clip_name.split("_")[0]

            dir_name = os.path.dirname(scene_path.replace("clips_metadata_F77", "metadata"))
            all_video_frames_scene_path = os.path.join(dir_name, f"{scene_name}.json")
            
            # select input frames from all video frames
            with open(all_video_frames_scene_path.strip(), 'r') as f:
                all_data_json = json.load(f)
            all_video_frames = all_data_json["frames"]
            input_indices = random.sample(list(range(0, len(all_video_frames))), self.num_input_views)
            input_image_paths_chosen = [all_video_frames[ic]["image_path"] for ic in input_indices]
            input_frames_chosen = [all_video_frames[ic] for ic in input_indices]
            
            selected_image_paths = input_image_paths_chosen + target_image_paths_chosen
            selected_frames = input_frames_chosen + target_frames_chosen
            
            # load and preprocess input + target frames
            all_images, all_intrinsics, all_c2ws = self.preprocess_frames(selected_frames, selected_image_paths)
            if all_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            # choose 4n+1 target c2w and intrinsic for each token, 4n+1=self.num_target_views
            target_c2ws_for_nvs = all_c2ws[self.num_input_views:][::4]
            target_intrinsics_for_nvs = all_intrinsics[self.num_input_views:][::4]
            target_img_for_nvs = all_images[self.num_input_views:][::4]

            # preprocess nvs poses
            scene_scale_factor = 1.35
            all_nvs_c2ws = torch.cat((all_c2ws[:self.num_input_views], target_c2ws_for_nvs), dim=0)
            c2ws_for_nvs_processed = self.preprocess_poses(all_nvs_c2ws, scene_scale_factor)

            caption = self.caption_dict[clip_name] if clip_name in self.caption_dict else ""
            
            return {
                "input_nvs_image": all_images[:self.num_input_views],                 # [self.num_input_views, 3, H, W]
                "input_nvs_c2w":   c2ws_for_nvs_processed[:self.num_input_views],     # [self.num_input_views, 4, 4]
                "input_nvs_fxfycxcy": all_intrinsics[:self.num_input_views],          # [self.num_input_views, 4]

                "target_nvs_image": target_img_for_nvs,                                 # [n, 3, H, W]
                "target_nvs_c2w": c2ws_for_nvs_processed[self.num_input_views:],        # [n, 4, 4]
                "target_nvs_fxfycxcy": target_intrinsics_for_nvs,                       # [n, 4]

                "scene_name": scene_name,
                "clip_name": clip_name,
                "target_src_image": all_images[self.num_input_views:],  # [self.num_target_views, 3, H, W]
                "target_src_c2w": all_c2ws[self.num_input_views:],
                "target_src_fxfycxcy": all_intrinsics[self.num_input_views:],
                "ref_img": all_images[self.num_input_views],
                "prompt": caption,
            }
            
        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))  

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True  
    

class UnifiedDatasetRe10K_Keyframes_121(torch.utils.data.Dataset):
    def __init__(
        self,
        metadata_path=None,
        repeat=1,
        config=None,
        **kwargs
    ):
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.image_size_W = 640
        self.patch_size = 16
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)

        self.num_target_views = 77
        self.num_input_views = 4
            
        self.load_metadata(metadata_path)
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F121', 'video_captions_refined.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
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
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
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
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        
        return images, intrinsics, c2ws

    def preprocess_frames_only(self, frames_chosen, image_paths_chosen):
        target_w_resize = 640
        target_h_resize = 360
        target_h_crop = 352
        
        images = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            aspect_ratio = original_image_w / original_image_h
            if abs(aspect_ratio - (16/9)) > 0.01:
                print(image_paths_chosen[0])
                return None, None, None

            if original_image_w != target_w_resize:
                image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)
            
            current_w, current_h = image.size 
            
            start_h = (current_h - target_h_crop) // 2
            start_w = 0
            
            # crop((left, top, right, bottom))
            image = image.crop((start_w, start_h, start_w + target_w_resize, start_h + target_h_crop))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            images.append(image)

        images = torch.stack(images, dim=0)
        
        return images

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

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            # directly load target frames
            with open(scene_path.strip(), 'r') as f:
                target_data_json = json.load(f)
            target_frames = target_data_json["frames"]
            if len(target_frames) < self.num_target_views:
                return self.__getitem__(random.randint(0, len(self) - 1))
            target_image_paths_chosen = [target_frames[i]["image_path"] for i in range(len(target_frames))]
            target_frames_chosen = [target_frames[i] for i in range(len(target_frames))]
            
            clip_name = os.path.basename(scene_path).split(".")[0]
            scene_name = clip_name.split("_")[0]

            dir_name = os.path.dirname(scene_path.replace("clips_metadata_F77", "metadata"))
            all_video_frames_scene_path = os.path.join(dir_name, f"{scene_name}.json")
            
            # select input frames from all video frames
            with open(all_video_frames_scene_path.strip(), 'r') as f:
                all_data_json = json.load(f)
            all_video_frames = all_data_json["frames"]
            input_indices = random.sample(list(range(0, len(all_video_frames))), self.num_input_views)
            input_image_paths_chosen = [all_video_frames[ic]["image_path"] for ic in input_indices]
            input_frames_chosen = [all_video_frames[ic] for ic in input_indices]
            
            selected_image_paths = input_image_paths_chosen + target_image_paths_chosen
            selected_frames = input_frames_chosen + target_frames_chosen
            
            # load and preprocess input + target frames
            all_images, all_intrinsics, all_c2ws = self.preprocess_frames(selected_frames, selected_image_paths)
            if all_images is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            # choose 4n+1 target c2w and intrinsic for each token, 4n+1=self.num_target_views
            target_c2ws_for_nvs = all_c2ws[self.num_input_views:][::4]
            target_intrinsics_for_nvs = all_intrinsics[self.num_input_views:][::4]
            target_img_for_nvs = all_images[self.num_input_views:][::4]

            # preprocess nvs poses
            scene_scale_factor = 1.35
            all_nvs_c2ws = torch.cat((all_c2ws[:self.num_input_views], target_c2ws_for_nvs), dim=0)
            c2ws_for_nvs_processed = self.preprocess_poses(all_nvs_c2ws, scene_scale_factor)

            caption = self.caption_dict[clip_name] if clip_name in self.caption_dict else ""

            return {
                "input_nvs_image": all_images[:self.num_input_views],                 # [self.num_input_views, 3, H, W]
                "input_nvs_c2w":   c2ws_for_nvs_processed[:self.num_input_views],     # [self.num_input_views, 4, 4]
                "input_nvs_fxfycxcy": all_intrinsics[:self.num_input_views],          # [self.num_input_views, 4]

                "target_nvs_image": target_img_for_nvs,                                 # [n, 3, H, W]
                "target_nvs_c2w": c2ws_for_nvs_processed[self.num_input_views:],        # [n, 4, 4]
                "target_nvs_fxfycxcy": target_intrinsics_for_nvs,                       # [n, 4]

                "scene_name": scene_name,
                "clip_name": clip_name,
                "prompt": caption,
                "target_src_image": all_images[self.num_input_views:],  # [self.num_target_views, 3, H, W]
                "target_src_c2w": all_c2ws[self.num_input_views:],
                "target_src_fxfycxcy": all_intrinsics[self.num_input_views:],
                "ref_img": all_images[self.num_input_views],
            }
            
        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))  

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True  


def generate_points_in_sphere(n_points, radius):
    """
    在指定半径的球体内部均匀生成点。
    (N_P, 3)
    """
    # Sample three independent uniform distributions
    samples_r = torch.rand(n_points)       # For radius distribution
    samples_phi = torch.rand(n_points)     # For azimuthal angle phi
    samples_u = torch.rand(n_points)       # For polar angle theta

    # Apply cube root to ensure uniform volumetric distribution
    r = radius * torch.pow(samples_r, 1/3)
    # Azimuthal angle phi uniformly distributed in [0, 2π]
    phi = 2 * math.pi * samples_phi
    # Convert u to theta to ensure cos(theta) is uniformly distributed
    theta = torch.acos(1 - 2 * samples_u)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1) # (N_P, 3)
    return points

def project_points_to_image(points_w, C2W, K_params, H, W):
    """
    将世界坐标系下的点投影到图像平面。
    假设 Batch 维度 B=1。

    :param points_w: (N_P, 3) 世界坐标点
    :param C2W: (T, 4, 4) C2W矩阵
    :param K_params: (T, 4) Intrinsics [fx, fy, cx, cy]
    :param H: 图像高度
    :param W: 图像宽度
    :return: (N_P, T, 2) 投影后的像素坐标 (u, v)
             (N_P, T) 布尔张量，表示点是否在相机前方 (Z_c > 0)
    """
    N_P, _ = points_w.shape
    T = C2W.shape[0]
    device = points_w.device
    
    # C2W: (T, 4, 4)
    W2C = torch.inverse(C2W) # (T, 4, 4)
    
    fx, fy, cx, cy = K_params.unbind(-1)
    
    K_matrix = torch.zeros((T, 3, 3), device=device)
    K_matrix[:, 0, 0] = fx
    K_matrix[:, 1, 1] = fy
    K_matrix[:, 0, 2] = cx
    K_matrix[:, 1, 2] = cy
    K_matrix[:, 2, 2] = 1.0
    
    # points_w: (N_P, 3) -> (1, N_P, 3) -> (T, N_P, 3)
    points_w_expanded = points_w.unsqueeze(0).repeat(T, 1, 1)
    points_w_homo = torch.cat([points_w_expanded, torch.ones_like(points_w_expanded[..., :1])], dim=-1)
    # (T, N_P, 4, 1)
    P_w = points_w_homo.unsqueeze(-1)

    # P_c = W2C @ P_w
    # W2C: (T, 4, 4) -> (T, 1, 4, 4)
    # P_w: (T, N_P, 4, 1)
    # P_c_homo: (T, N_P, 4, 1)
    P_c_homo = W2C.unsqueeze(1) @ P_w
    
    # P_c: (T, N_P, 3)
    P_c = P_c_homo[:, :, :3, 0] 
    
    Z_c = P_c[..., 2] # (T, N_P)
    
    # K_matrix: (T, 3, 3) -> (T, 1, 3, 3)
    # P_c: (T, N_P, 3) -> (T, N_P, 3, 1)
    P_c_3d = P_c.unsqueeze(-1)
    
    # P_img_homo: (T, N_P, 3, 1)
    P_img_homo = K_matrix.unsqueeze(1) @ P_c_3d 
    
    # P_img_homo: (T, N_P, 3, 1) -> (T, N_P, 3)
    P_img_homo_3 = P_img_homo[..., 0]
    
    u_v = P_img_homo_3[..., :2] / P_img_homo_3[..., 2:3] # (T, N_P, 2)
    
    u_v_reshaped = u_v.permute(1, 0, 2)
    
    # Z_c: (T, N_P) -> (N_P, T)
    Z_c_reshaped = Z_c.permute(1, 0)
    
    is_in_front = Z_c_reshaped > 0
    
    return u_v_reshaped, is_in_front


def get_fov_indices(curr_frame, memory_condition_length, H, W, pose_conditions, K, horizon, device):
    """
    Generate indices for condition similarity based on the current frame and C2W/K conditions.
    假设 Batch 维度 B=1。

    :param curr_frame: 当前帧索引
    :param memory_condition_length: 要选择的历史帧数量 M
    :param pose_conditions: (T_total, 4, 4) C2W 姿态矩阵
    :param K: (T_total, 4) 内参 [fx, fy, cx, cy]
    :param horizon: 当前帧到未来帧的数量 H'
    :return: (memory_condition_length) 选取的历史帧索引
    """
    # (H', 4, 4) - Current/Future
    pose_cur = pose_conditions[curr_frame:curr_frame+horizon].to(device) 
    # (H', 4) - Current/Future
    K_curr = K[curr_frame:curr_frame+horizon].to(device) 
    # (T_hist, 4, 4) - History
    pose_history = pose_conditions[:curr_frame].to(device) 
    # (T_hist, 4) - History
    K_history = K[:curr_frame].to(device) 

    num_samples = 10000
    radius = 30
    points = generate_points_in_sphere(num_samples, radius).to(device) # (N_P, 3)
    
    camera_center_w = pose_conditions[curr_frame, :3, 3].to(device) # (3)
    
    points_w = points + camera_center_w.unsqueeze(0) # (N_P, 3)

    # uv_cur: (N_P, H', 2), in_front_cur: (N_P, H')
    uv_cur, in_front_cur = project_points_to_image(points_w, pose_cur, K_curr, H, W) 

    is_inside_pixel_cur = (uv_cur[..., 0] >= 0) & (uv_cur[..., 0] < W) & \
                          (uv_cur[..., 1] >= 0) & (uv_cur[..., 1] < H) # (N_P, H')

    in_fov_cur_per_frame = in_front_cur & is_inside_pixel_cur # (N_P, H')

    in_fov1 = torch.sum(in_fov_cur_per_frame, dim=-1) > 0 # (N_P)

    # History: T_hist = curr_frame
    T_hist = curr_frame
    # uv_hist: (N_P, T_hist, 2), in_front_hist: (N_P, T_hist)
    uv_hist, in_front_hist = project_points_to_image(points_w, pose_history, K_history, H, W) 

    is_inside_pixel_hist = (uv_hist[..., 0] >= 0) & (uv_hist[..., 0] < W) & \
                           (uv_hist[..., 1] >= 0) & (uv_hist[..., 1] < H) # (N_P, T_hist)

    in_fov_list = in_front_hist & is_inside_pixel_hist # (N_P, T_hist)
    in_fov_list = in_fov_list.permute(1, 0) 
    
    
    random_idx_b = []
        
    if in_fov1.sum() == 0:
        return torch.zeros(memory_condition_length, dtype=torch.long)
    
    current_in_fov_list = in_fov_list.clone() 

    for _ in range(memory_condition_length):
        # ((T_hist, N_P) & (N_P)) / (N_P).sum() -> (T_hist)
        overlap_ratio = (current_in_fov_list.bool() & in_fov1.unsqueeze(0)).sum(1).float() / in_fov1.sum().float()
        
        confidence = overlap_ratio 

        if len(random_idx_b) > 0:
            confidence[torch.tensor(random_idx_b, device=device)] = -1e10
            
        _, r_idx = torch.topk(confidence, k=1, dim=0)
        
        selected_idx = r_idx.item()
        random_idx_b.append(selected_idx)

        # current_in_fov_list[selected_idx, :] = False

    # random_idx = torch.tensor(random_idx_b, dtype=torch.long).cpu()

    return random_idx_b

class UnifiedDatasetRe10K_2(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, 
        metadata_path=None,
        repeat=1,
        config=None,
        **kwargs
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.patch_size = 16
        self.num_global_views = 8
        self.num_local_views = 13 # 4n+1
        self.num_target_views = 21
        self.skip = 5
        self.square_crop = kwargs.get("square_crop", True)
        self.num_views = self.num_global_views + self.num_local_views + self.num_target_views

        self.load_metadata(metadata_path)
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]
        filtered_data = []
        for scene_path in self.data:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            if len(data_json["frames"]) >= self.num_views * self.skip:
                filtered_data.append(scene_path)
        self.data = filtered_data

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption', 'video_captions_refined.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
            self.caption_dict = json.load(f)

    def view_selector(self, frames):
        total_frames = len(frames)
        if total_frames < self.num_views:
            return None
        
        index_start = random.randint(0, total_frames - (self.num_local_views+self.num_target_views))
        local_target_index = list(range(index_start, index_start+(self.num_local_views+self.num_target_views)))

        global_candidates = [i for i in range(0, total_frames) if i not in local_target_index]

        return global_candidates + local_target_index
    
    
    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.image_size_H
        patch_size = self.patch_size

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            if resize_h != original_image_h:
                start_h = (original_image_h - resize_h) // 2
                start_w = (original_image_w - resize_w) // 2

                end_h = start_h + resize_h
                end_w = start_w + resize_w

                image = image.crop((start_w, start_h, end_w, end_h))
                # image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
                # if square_crop:
                #     min_size = min(resize_h, resize_w)
                #     start_h = (resize_h - min_size) // 2
                #     start_w = (resize_w - min_size) // 2
                #     image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            fxfycxcy = np.array(cur_frame["fxfycxcy"])

            if resize_h != original_image_h:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
                
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(self, in_c2ws, scene_scale_factor=1.35):
        # Translation and Rotation
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

        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        if scene_scale > 1e-6:
             scene_scale = scene_scale_factor * scene_scale
             in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def preprocess_single_frames(self, image_path_chosen):
        resize_h = self.image_size_H
        patch_size = self.patch_size

        image = Image.open(image_path_chosen)
        original_image_w, original_image_h = image.size
            
        resize_w = int(resize_h / original_image_h * original_image_w)
        resize_w = int(round(resize_w / patch_size) * patch_size)

        if resize_h != original_image_h:
            start_h = (original_image_h - resize_h) // 2
            start_w = (original_image_w - resize_w) // 2

            end_h = start_h + resize_h
            end_w = start_w + resize_w

            image = image.crop((start_w, start_h, end_w, end_h))

        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image
    
    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)


        with open(scene_path.strip(), 'r') as f:
            data_json = json.load(f)
        
        frames = data_json["frames"][::self.skip]
        scene_name = data_json["scene_name"]
        image_indices = self.view_selector(frames) # candidates + local + target
        
        if image_indices is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
        frames_chosen = [frames[ic] for ic in image_indices]
        
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen) # candidates + local + target

        all_index = list(range(len(image_indices)))
        target_index = all_index[-self.num_target_views:]
        local_index = all_index[-(self.num_local_views+self.num_target_views):-self.num_target_views]
        candidate_index = all_index[:-(self.num_local_views+self.num_target_views)]
        H, W = input_images.shape[-2:]
        global_index = get_fov_indices(
            curr_frame=len(candidate_index), 
            memory_condition_length=self.num_global_views, 
            H=H, W=W, 
            pose_conditions=input_c2ws[candidate_index+target_index], 
            K=input_intrinsics[candidate_index+target_index], 
            horizon=self.num_target_views,
            device=input_images.device
        )

        caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

        ref_img = self.preprocess_single_frames(frames[0]["image_path"])
        
        return {
            "mem_frames": input_images[global_index+local_index],
            "mem_c2w": input_c2ws[global_index+local_index],
            "mem_intric": input_intrinsics[global_index+local_index],
            "num_global": self.num_global_views,
            "num_local": self.num_local_views,

            "target_frames": input_images[target_index],
            "target_c2w": input_c2ws[target_index],
            "target_intric": input_intrinsics[target_index],

            "scene_name": scene_name,
            "prompt": caption,

            "ref_img": ref_img,
        }

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """
    将 (3, H, W) 范围 0-1 的 PyTorch tensor 转换为 PIL Image。
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
        
    np_array = tensor.permute(1, 2, 0).numpy()

    np_array = (np_array * 255).astype(np.uint8)

    pil_image = Image.fromarray(np_array)
    
    return pil_image

class UnifiedDatasetCam(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, 
        metadata_path=None,
        num_views=81,
        repeat=1,
        config=None,
        time_division_factor=4,
        time_division_remainder=1,
        **kwargs
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.patch_size = 16
        self.num_views = num_views
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.square_crop = kwargs.get("square_crop", False)
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)

        self.load_metadata(metadata_path)
    
    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption', 'video_captions_refined.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
            self.caption_dict = json.load(f)
    
    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.image_size_H
        patch_size = self.patch_size
        square_crop = self.square_crop

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = Image.open(cur_image_path)
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            if resize_h != original_image_h:
                start_h = (original_image_h - resize_h) // 2
                start_w = (original_image_w - resize_w) // 2

                end_h = start_h + resize_h
                end_w = start_w + resize_w

                image = image.crop((start_w, start_h, end_w, end_h))
                # image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
                # if square_crop:
                #     min_size = min(resize_h, resize_w)
                #     start_h = (resize_h - min_size) // 2
                #     start_w = (resize_w - min_size) // 2
                #     image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            
            if square_crop and (resize_h != original_image_h):
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws
    
    def preprocess_single_frames(self, image_path_chosen):
        resize_h = self.image_size_H
        patch_size = self.patch_size
        square_crop = self.square_crop

        image = Image.open(image_path_chosen)
        original_image_w, original_image_h = image.size
            
        resize_w = int(resize_h / original_image_h * original_image_w)
        resize_w = int(round(resize_w / patch_size) * patch_size)

        if resize_h != original_image_h:
            image = image.resize((resize_w, resize_h), resample=Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image

    def view_selector(self, frames):
        total_frames = len(frames)
        
        if total_frames < self.num_views:
            num_frames = max(1, total_frames)

            factor = self.time_division_factor
            remainder = self.time_division_remainder
            current_remainder = num_frames % factor

            delta = current_remainder - remainder
            if delta < 0:
                delta += factor
                
            num_frames -= delta
            num_frames = max(1, num_frames)
            
            return list(range(num_frames))
        else:
            segment_length = self.num_views
            max_start_index = total_frames - segment_length
            
            start_index = random.randint(0, max_start_index)
            end_index = start_index + segment_length
            
            return list(range(start_index, end_index))

    def __getitem__(self, data_id):
        scene_path_or_dict = self.data[data_id % len(self.data)]

        if isinstance(scene_path_or_dict, str):
            scene_path = scene_path_or_dict
        elif isinstance(scene_path_or_dict, dict) and "scene_path" in scene_path_or_dict:
            scene_path = scene_path_or_dict["scene_path"]
        else:
            scene_path = str(scene_path_or_dict)

        try:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            
            frames = data_json["frames"]
            scene_name = data_json["scene_name"]
            image_indices = self.view_selector(frames)
            
            if image_indices is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]
            
            input_images, input_intrinsics, input_src_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)

            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)
            scene_indices = torch.full_like(image_indices, data_id)
            indices = torch.cat([image_indices, scene_indices], dim=-1)

            caption = self.caption_dict[scene_name] if scene_name in self.caption_dict else ""

            # ref_img = self.preprocess_single_frames(frames[0]["image_path"])
            ref_img = tensor_to_pil_image(input_images[0])

            return {
                "image": input_images,
                "fxfycxcy": input_intrinsics,
                "index": indices,
                "scene_name": scene_name,
                "src_c2w": input_src_c2ws,
                "prompt": caption,
                "ref_img": ref_img
            }

        except Exception as e:
            print(f"Error loading data id {data_id}, path: {scene_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def __len__(self):
        return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


class UnifiedDatasetMultiTokens(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, 
        metadata_path=None,
        num_views=77,
        repeat=1,
        config=None,
        **kwargs
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.load_from_cache = metadata_path is None
        
        self.data = []
        
        self.config = config
        self.image_size_H = 352
        self.patch_size = 16
        self.num_views = num_views
        self.square_crop = kwargs.get("square_crop", False)
        self.scene_scale_factor = kwargs.get("scene_scale_factor", 1.35)

        self.load_metadata(metadata_path)

    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.data = [path.strip() for path in lines if path.strip()]
        filtered_data = []
        for scene_path in self.data:
            with open(scene_path.strip(), 'r') as f:
                data_json = json.load(f)
            if len(data_json["frames"]) >= self.num_views * self.skip:
                filtered_data.append(scene_path)
        self.data = filtered_data

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption', 'video_captions_refined.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
            self.caption_dict = json.load(f)

        
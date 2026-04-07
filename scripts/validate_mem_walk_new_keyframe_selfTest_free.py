import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from base_diffsynth import save_video, VideoData, load_state_dict
# from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from base_diffsynth.pipelines.wan_video_mem_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import json
from frame_memory_retrieval import KeyFrameMemoryBank, KeyFrameMemoryBankLearnedOccRetrieval

from torchvision.utils import save_image
from tqdm import tqdm
import cv2
import numpy as np
import math
import torch.nn.functional as F
from generate_custom_trajectory import generate_camera_trajectory_local, pose_string_to_json


import matplotlib.pyplot as plt
import numpy as np
from torch.amp import autocast

def preprocess_frames(frames_chosen, image_paths_chosen):
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

def preprocess_poses(
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



class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.all_scene_paths = [path.strip() for path in lines if path.strip()][-1:]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'captions', 'video_captions_refined.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
            self.caption_dict = json.load(f)

    def __len__(self):
        return len(self.all_scene_paths)

    def __getitem__(self, idx):
        scene_path = self.all_scene_paths[idx].strip()
        data_json = json.load(open(scene_path, 'r'))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]

        image_indices= list(range(len(frames)))

        image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
        frames_chosen = [frames[ic] for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = preprocess_frames(frames_chosen, image_paths_chosen)

        # caption_clips = []
        # for i in range(len(frames)//76):
        #     clip_name = f"{scene_name}_clip_{i}"
        #     if clip_name in self.caption_dict:
        #         caption_clips.append(self.caption_dict[clip_name])

        caption = self.caption_dict[scene_name]
        
        return {
            "input_images": input_images,
            "input_c2ws": input_c2ws,
            "input_intrinsics": input_intrinsics,
            # "captions": caption_clips,  # Note: collate_fn handling might be needed if lengths vary
            "caption": caption,
            "scene_name": scene_name
        }

def test_data_loader(metadata_path):
    dataset = ValidationDataset(metadata_path)
    # Using a simple collate function or batch size 1 to handle variable length captions/clips easily
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=16, prefetch_factor=2)
    return dataloader

def tensor2rgb(image_tensor):
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    image_numpy = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    if image_numpy.shape[0] == 1:
        image_numpy = image_numpy[0]
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return image_numpy

def rgb2tensor(image_numpy):
    if isinstance(image_numpy, torch.Tensor):
        return image_numpy
    if isinstance(image_numpy, Image.Image):
        image_numpy = np.array(image_numpy)
    if image_numpy.ndim == 2:
        image_numpy = np.expand_dims(image_numpy, axis=0)
    else:
        image_numpy = np.transpose(image_numpy, (2, 0, 1))
    image_tensor = torch.from_numpy(image_numpy).float() / 255.0
    return image_tensor

model_paths = 'configs/model_conf_wan_fun_i2v_1.3b.json'
model_configs = []
with open(model_paths, 'r', encoding='utf-8') as f:
    model_path_loaded = json.load(f)
model_configs += [ModelConfig(path=path) for path in model_path_loaded]

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=torch.device("cuda:0"),
    model_configs=model_configs,
)

saved_model_path = './trained_models/Keyframes_SceneDec_ft/step-11000.safetensors'
state_dict = load_state_dict(saved_model_path, device=pipe.device, torch_dtype=pipe.torch_dtype)

lora_state_dict = {k: v for k, v in state_dict.items() if 'lora' in k}
pipe.load_lora(pipe.dit, state_dict=lora_state_dict, alpha=1)

other_state_dict = {k.removeprefix("pipe."): v for k, v in state_dict.items() if 'lora' not in k}
pipe.load_state_dict(other_state_dict, strict=False)

pipe.enable_vram_management()

base_results_dir = './results/action_ctrl'
os.makedirs(base_results_dir, exist_ok=True)

test_dataloader = test_data_loader(
                './self_captured/full_list_gaoda.txt'
                )

ctx_len = 4
num_per_clip = 77
mem_bank = KeyFrameMemoryBankLearnedOccRetrieval(n_sample_points=100000)
for data in tqdm(test_dataloader):
    test_input_imgs = data["input_images"]
    test_input_c2ws = data["input_c2ws"][:4].clone()
    ref_w2c = torch.linalg.inv(test_input_c2ws[-1])
    test_input_c2ws = ref_w2c @ test_input_c2ws
    test_input_intrinsics = data["input_intrinsics"][:4]
    # captions = data["captions"]
    caption = data["caption"]
    scene_name = data["scene_name"]
    
    print(f"Scene {scene_name} begin generation...")
    save_results_dir = f"{base_results_dir}/{scene_name}"
    os.makedirs(save_results_dir, exist_ok=True)

    mem_bank.reset()
    mem_bank.add_key_frames(
        frames_c2w=test_input_c2ws[:4],
        frames_fxfycxcy=test_input_intrinsics[:4],
        frames_rgb=test_input_imgs[:4],
    )

    count = 0

    all_videos = [Image.fromarray(tensor2rgb(test_input_imgs[3]))]  # Start with the first frame
    all_extrinsics = []
    all_intrinsics = []
    scene_logs = []
    
    # Track the last pose to use as initial pose for the next clip
    # Start with the last input c2w as the initial pose
    current_initial_pose = test_input_c2ws[-1].cpu().numpy()

    while True:
        action_string = input("Please enter action_string (or 'c' to break/continue): ")
        if action_string.strip().lower() == 'c':
            break

        pose_json = pose_string_to_json(action_string, initial_pose=current_initial_pose)
        if len(pose_json) != 77:
            print(f"Pose JSON length should be 77 for one clip (including keyframe), but got {len(pose_json)}. Please try again.")
            continue

        # prepare clips to be generated
        c2w_list = []
        fxfycxcy_list = []
        for i in range(77):
            pose_dict = pose_json[str(i)]
            c2w_list.append(pose_dict["extrinsic"])
            K = pose_dict["K"]
            fxfycxcy_list.append([K[0][0], K[1][1], K[0][2], K[1][2]])

        current_initial_pose = c2w_list[-1]
        
        clips_trajs = torch.tensor(c2w_list).float()
        clips_fxfycxcys = test_input_intrinsics[-1].unsqueeze(0).repeat(clips_trajs.shape[0], 1)
        # clips_fxfycxcys = torch.tensor(fxfycxcy_list).float()

        ref_img = rgb2tensor(all_videos[-1]).to('cuda')
        cur_traj = clips_trajs.to('cuda')
        cur_fxfycxcy = clips_fxfycxcys.to('cuda')
        if count == 0:
            cur_caption = caption
            print(cur_caption)
        else:
            cur_caption = input("Please enter caption for current clip: ")

        # Save camera parameters
        extrinsics_data = cur_traj.cpu().numpy().tolist()
        intrinsics_data = []
        for intr in cur_fxfycxcy.cpu().numpy():
            fx, fy, cx, cy = intr
            intrinsics_data.append([
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0]
            ])
        if count > 0:
            all_extrinsics.extend(extrinsics_data[1:])
            all_intrinsics.extend(intrinsics_data[1:])          
        else:
            all_extrinsics.extend(extrinsics_data)
            all_intrinsics.extend(intrinsics_data)

        retrieved_result = mem_bank.retrieve_top_k_cameras_learned_use(
            query_c2w=cur_traj[::4],
            query_fxfycxcy=cur_fxfycxcy[::4],
            k=ctx_len,
        )
        print("Retrieved frames indices:", retrieved_result)

        scene_logs.append({
            "clip_id": count,
            "retrieved_indices": retrieved_result,
        })
        with open(os.path.join(save_results_dir, "retrieval_logs.json"), "w") as f:
            json.dump(scene_logs, f, indent=4)
        
        retrieved_data = mem_bank.get_retrieved_data_frames(retrieved_result)
        union_c2ws = torch.cat([retrieved_data['frames_c2ws'], cur_traj[::4]], dim=0)
        union_c2ws_processed = preprocess_poses(union_c2ws)

        video = pipe(
            height=352,
            width=640,
            num_frames=num_per_clip,

            input_nvs_image=retrieved_data['frames_rgb'],
            input_nvs_c2w=union_c2ws_processed[:ctx_len],
            input_nvs_fxfycxcy=retrieved_data['frames_fxfycxcy'],

            target_nvs_c2w=union_c2ws_processed[ctx_len:],
            target_nvs_fxfycxcy=cur_fxfycxcy[::4],
            target_nvs_image=None,
            debug_save_dir=f"{save_results_dir}/debug_clip_{count}",

            cam_c2w=cur_traj,
            cam_intric=cur_fxfycxcy,
            reference_image=ref_img,
            prompt=cur_caption,
            negative_prompt="色调艳丽，过曝，细节模糊不清，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，杂乱的背景,不完整的,扭曲的墙",
            seed=1, tiled=True,
            return_weights=False,
        )

        all_videos.extend(video[1:])
        save_video(all_videos[-num_per_clip:], f"{save_results_dir}/video_Wan2.1-Fun-{count}.mp4", fps=16, quality=5)
        save_video(all_videos, f"{save_results_dir}/video_Wan2.1-Fun-combined.mp4", fps=16, quality=5)

        video_tensor = torch.stack([torch.from_numpy(np.array(im) / 255.0).permute(2,0,1).float() for im in video], dim=0)

        for i in range(len(video_tensor)):
            os.makedirs(f"{save_results_dir}/video_frames_{count}", exist_ok=True)
            save_image(video_tensor[i], f"{save_results_dir}/video_frames_{count}/video_frame_{i}.png")

        count+=1

        mem_bank.add_key_frames(
            frames_c2w=cur_traj.reshape(-1,4,4)[::4],
            frames_fxfycxcy=cur_fxfycxcy.reshape(-1,4)[::4],
            frames_rgb=video_tensor.reshape(-1,video_tensor.shape[1],video_tensor.shape[2],video_tensor.shape[3])[::4],
        )
        print(f"Memory bank size: {mem_bank.frames_num} frames.")

    
    assert len(all_extrinsics) == 1+76*count, f"{len(all_extrinsics)} vs {1+76*count}"
    with open(f"{save_results_dir}/gt_cam.json", "w") as f:
        json.dump({
            "extrinsics": all_extrinsics,
            "intrinsics": all_intrinsics
        }, f, indent=4)
    # save_video(all_videos, f"{save_results_dir}/video_Wan2.1-Fun-combined.mp4", fps=16, quality=5)
    print(f"Generation completed for scene {scene_name}.")


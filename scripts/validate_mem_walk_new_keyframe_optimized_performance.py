import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from base_diffsynth import save_video, VideoData, load_state_dict
# from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from base_diffsynth.pipelines.wan_video_mem_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import json
from frame_memory_retrieval_optimized import KeyFrameMemoryBankLearnedOccRetrieval
from frame_memory_retrieval_learned_fast import KeyFrameMemoryBankLearnedOccRetrievalFast

from torchvision.utils import save_image
from tqdm import tqdm
import cv2
import numpy as np
import math
import torch.nn.functional as F
from accelerate import Accelerator
import time

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


def preprocess_input_poses(
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


def preprocess_input_poses_batch(
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
    """
    Batch version of preprocess_input_poses.
    Args:
        in_c2ws: (B, F, 4, 4)
    Returns:
        processed_c2ws: (B, F, 4, 4)
        avg_poses: (B, 4, 4)
        avg_poses_inv: (B, 4, 4)
    """
    B = in_c2ws.shape[0]
    processed_c2ws_list = []
    avg_poses_list = []
    avg_poses_inv_list = []

    for b in range(B):
        processed_c2w, avg_pose, avg_pose_inv = preprocess_input_poses(in_c2ws[b], scene_scale_factor=scene_scale_factor)
        processed_c2ws_list.append(processed_c2w)
        avg_poses_list.append(avg_pose)
        avg_poses_inv_list.append(avg_pose_inv)
    
    return torch.stack(processed_c2ws_list), torch.stack(avg_poses_list), torch.stack(avg_poses_inv_list)


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
    def __init__(self, metadata_path, rank=0, world_size=1):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        all_lines = [path.strip() for path in lines if path.strip()][:2]
        self.all_scene_paths = all_lines[rank::world_size]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_F239', 'video_captions_refined.json')
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

        caption_clips = []
        for i in range(len(frames)//76):
            clip_name = f"{scene_name}_clip_{i}"
            if clip_name in self.caption_dict:
                caption_clips.append(self.caption_dict[clip_name])
        
        return {
            "input_images": input_images,
            "input_c2ws": input_c2ws,
            "input_intrinsics": input_intrinsics,
            "captions": caption_clips,  # Note: collate_fn handling might be needed if lengths vary
            "scene_name": scene_name
        }

def test_data_loader(metadata_path, rank=0, world_size=1):
    dataset = ValidationDataset(metadata_path, rank=rank, world_size=world_size)
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


accelerator = Accelerator()

device = accelerator.device
if device.type == 'cuda' and device.index is None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

model_paths = 'configs/model_conf_wan_fun_i2v_1.3b.json'
model_configs = []
with open(model_paths, 'r', encoding='utf-8') as f:
    model_path_loaded = json.load(f)
model_configs += [ModelConfig(path=path) for path in model_path_loaded]

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=model_configs,
)

saved_model_path = './trained_models/Keyframes_SceneDec_ft/step-11000.safetensors'
state_dict = load_state_dict(saved_model_path, device=pipe.device, torch_dtype=pipe.torch_dtype)

lora_state_dict = {k: v for k, v in state_dict.items() if 'lora' in k}
pipe.load_lora(pipe.dit, state_dict=lora_state_dict, alpha=1)

other_state_dict = {k.removeprefix("pipe."): v for k, v in state_dict.items() if 'lora' not in k}
# other_state_dict = {k.removeprefix("pipe."): v for k, v in state_dict.items() if (('lora' not in k) and ('nvs_model' not in k))}
pipe.load_state_dict(other_state_dict, strict=False)

pipe.enable_vram_management()


base_results_dir = './results/performance_test_fulloptimized'
if accelerator.is_main_process:
    os.makedirs(base_results_dir, exist_ok=True)

test_dataloader = test_data_loader(
                '/mnt/afse2/DATASET/re10k_preprocessed/test/full_list_F239.txt',
                rank=accelerator.process_index,
                world_size=accelerator.num_processes
                )

ctx_st, ctx_ed = 7, 11
ctx_len = ctx_ed - ctx_st
num_per_clip = 77
retrieval_impl = os.environ.get("I3DM_RETRIEVAL_IMPL", "fast").lower()
if retrieval_impl in {"fast", "full_optimized"}:
    mem_bank = KeyFrameMemoryBankLearnedOccRetrievalFast(
        n_sample_points=100000,
        candidate_pool_size=int(os.environ.get("I3DM_RETRIEVAL_CANDIDATE_POOL", "64")),
        proposal_points=int(os.environ.get("I3DM_RETRIEVAL_PROPOSAL_POINTS", "8192")),
        always_include_recent=int(os.environ.get("I3DM_RETRIEVAL_RECENT", "8")),
        learned_batch_size=int(os.environ.get("I3DM_RETRIEVAL_BATCH_SIZE", "32")),
    )
else:
    mem_bank = KeyFrameMemoryBankLearnedOccRetrieval(n_sample_points=100000)
# all_scene_acc_retrieval_time = 0.0
# all_scene_gen_time = 0.0
all_scene_avg_clip_retrieval_time = 0.0
all_scene_avg_clip_dit_time = 0.0
all_scene_avg_clip_add_time = 0.0
all_scene_avg_clip_gen_time = 0.0
first_scene = True

for data in tqdm(test_dataloader):
    test_input_imgs = data["input_images"]
    test_input_c2ws = data["input_c2ws"]
    test_input_intrinsics = data["input_intrinsics"]
    captions = data["captions"]
    scene_name = data["scene_name"]

    print(f"Scene {scene_name} begin generation...")
    save_results_dir = f"{base_results_dir}/{scene_name}"
    os.makedirs(save_results_dir, exist_ok=True)

    # mem_bank = KeyFrameMemoryBank(n_sample_points=100000)

    mem_bank.reset()
    mem_bank.add_key_frames(
        frames_c2w=test_input_c2ws[ctx_st:ctx_ed],
        frames_fxfycxcy=test_input_intrinsics[ctx_st:ctx_ed],
        frames_rgb=test_input_imgs[ctx_st:ctx_ed],
    )

    # prepare clips to be generated
    num_clip = (len(test_input_imgs) - ctx_ed) // 76
    candidate_c2ws = test_input_c2ws[ctx_ed-1:ctx_ed+num_clip*76]
    candidate_fxfycxcy = test_input_intrinsics[ctx_ed-1:ctx_ed+num_clip*76]
    candidate_rgbs = test_input_imgs[ctx_ed-1:ctx_ed+num_clip*76]

    clips_trajs = []
    clips_fxfycxcys = []
    clips_rgbs = []
    # Process Clips
    for i in range(num_clip):
        clips_trajs.append(candidate_c2ws[i*76 : i*76+77])
        clips_fxfycxcys.append(candidate_fxfycxcy[i*76 : i*76+77])
        clips_rgbs.append(candidate_rgbs[i*76 : i*76+77])
    # Process Reverse Clips
    for i in range(num_clip - 1, -1, -1):
        clips_trajs.append(torch.flip(candidate_c2ws[i*76 : i*76+77], [0]))
        clips_fxfycxcys.append(torch.flip(candidate_fxfycxcy[i*76 : i*76+77], [0]))
        clips_rgbs.append(torch.flip(candidate_rgbs[i*76 : i*76+77], [0]))
    clips_captions = captions + captions[::-1]
    ref_img = test_input_imgs[ctx_ed-1]

    count = 0
    all_videos = []
    scene_logs = []
    replace_first_frame = True
    per_clip_dit_time = 0.0
    per_clip_retrive_time = 0.0
    per_clip_add_time = 0.0
    per_clip_gen_time = 0.0
    for ind in tqdm(range(len(clips_captions))):
        count += 1

        cur_traj = clips_trajs[ind].to(accelerator.device)
        cur_fxfycxcy = clips_fxfycxcys[ind].to(accelerator.device)
        cur_rgb = clips_rgbs[ind].to(accelerator.device)
        cur_caption = clips_captions[ind]

        # retrieved_result, retrieved_ratios = mem_bank.retrieve_top_k_cameras_keep_last(
        #     query_c2w=cur_traj[::4],
        #     query_fxfycxcy=cur_fxfycxcy[::4],
        #     k=ctx_len,
        # )
        # retrieved_result, retrieved_ratios = mem_bank.retrieve_top_k_cameras(
        #     query_c2w=cur_traj[::4],
        #     query_fxfycxcy=cur_fxfycxcy[::4],
        #     k=ctx_len,
        # )

        start_time = time.time()
        if retrieval_impl == "fast":
            retrieved_result, retrieval_profile = mem_bank.retrieve_camera_occ_fast(
                query_c2w=cur_traj[::4],
                query_fxfycxcy=cur_fxfycxcy[::4],
                k=ctx_len,
                return_profile=True,
            )
        elif retrieval_impl == "full_optimized":
            retrieved_result, retrieval_profile = mem_bank.retrieve_camera_occ_full_optimized(
                query_c2w=cur_traj[::4],
                query_fxfycxcy=cur_fxfycxcy[::4],
                k=ctx_len,
                return_profile=True,
            )
        else:
            retrieved_result = mem_bank.retrieve_camera_occ_perf_test(
            # retrieved_result, retrieved_ratios = mem_bank.retrieve_top_k_cameras(
            # retrieved_result = mem_bank.random_retrieve(
            # retrieved_result = mem_bank.latest_retrieve(
            # retrieved_result = mem_bank.retrieve_top_k_cameras_fov_optimized(
            # retrieved_result = mem_bank.retrieve_top_k_cameras(
                query_c2w=cur_traj[::4],
                query_fxfycxcy=cur_fxfycxcy[::4],
                k=ctx_len,
            )
            retrieval_profile = {
                "memory_size": mem_bank.frames_num,
                "proposal_candidates": mem_bank.frames_num - 1,
                "proposal_time": 0.0,
                "learned_time": 0.0,
                "selection_time": 0.0,
            }
        
        retrieved_data = mem_bank.get_retrieved_data_frames(retrieved_result)
        torch.cuda.synchronize()
        retrieve_time_cost = time.time() - start_time
        print(f"Clip {count} retrieval time cost: {retrieve_time_cost:.4f} seconds.")
        print(
            "Retrieval profile: "
            f"memory={retrieval_profile.get('memory_size', mem_bank.frames_num)}, "
            f"candidates={retrieval_profile.get('proposal_candidates', 0)}, "
            f"proposal={retrieval_profile.get('proposal_time', 0.0):.4f}s, "
            f"learned={retrieval_profile.get('learned_time', 0.0):.4f}s, "
            f"selection={retrieval_profile.get('selection_time', 0.0):.4f}s"
        )


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
            target_nvs_image=cur_rgb[::4],
            debug_save_dir=f"{save_results_dir}/debug_clip_{count}",
    
            cam_c2w=cur_traj,
            cam_intric=cur_fxfycxcy,
            reference_image=ref_img,
            prompt=cur_caption,
            negative_prompt="色调艳丽，过曝，细节模糊不清，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，杂乱的背景",
            seed=1, tiled=True,
            return_weights=False,
        )
        if replace_first_frame:
            video[0] = tensor2rgb(ref_img)

        all_videos.extend(video)

        torch.cuda.synchronize()
        dit_time_cost = time.time() - start_time - retrieve_time_cost
        print(f"Clip {count} generation time cost: {dit_time_cost:.4f} seconds.")

        # save_video(video, f"{save_results_dir}/video_Wan2.1-Fun-{count}.mp4", fps=16, quality=5)
        # save_video([tensor2rgb(v) for v in cur_rgb], f"{save_results_dir}/gt_video_Wan2.1-Fun-{count}.mp4", fps=16, quality=5)

        video_tensor = torch.stack([torch.from_numpy(np.array(im) / 255.0).permute(2,0,1).float() for im in video], dim=0)
        ref_img = video_tensor[-1]

        # for i in range(len(video_tensor)):
        #     os.makedirs(f"{save_results_dir}/video_frames_{count}", exist_ok=True)
        #     os.makedirs(f"{save_results_dir}/gt_frames_{count}", exist_ok=True)
        #     save_image(video_tensor[i], f"{save_results_dir}/video_frames_{count}/video_frame_{i}.png")
        #     save_image(cur_rgb[i], f"{save_results_dir}/gt_frames_{count}/video_frame_{i}.png")

        mem_bank.add_key_frames(
            frames_c2w=cur_traj.reshape(-1,4,4)[::4],
            frames_fxfycxcy=cur_fxfycxcy.reshape(-1,4)[::4],
            frames_rgb=video_tensor.reshape(-1,video_tensor.shape[1],video_tensor.shape[2],video_tensor.shape[3])[::4],
        )

        torch.cuda.synchronize()
        add_time_cost = time.time() - start_time - retrieve_time_cost - dit_time_cost
        print(f"Clip {count} add time cost: {add_time_cost:.4f} seconds.")


        per_clip_gen_time += time.time() - start_time
        per_clip_retrive_time += retrieve_time_cost
        per_clip_dit_time += dit_time_cost
        per_clip_add_time += add_time_cost

        print(f"Clip {count} total time cost: {time.time() - start_time:.4f} seconds.")

        # print(f"Memory bank size: {mem_bank.frames_num} frames.")

    if not first_scene:
        per_clip_dit_time = per_clip_dit_time / len(clips_captions)
        per_clip_retrive_time = per_clip_retrive_time / len(clips_captions)
        per_clip_add_time = per_clip_add_time / len(clips_captions)
        per_clip_gen_time = per_clip_gen_time / len(clips_captions)

        all_scene_avg_clip_dit_time += per_clip_dit_time
        all_scene_avg_clip_retrieval_time += per_clip_retrive_time
        all_scene_avg_clip_add_time += per_clip_add_time
        all_scene_avg_clip_gen_time += per_clip_gen_time

    first_scene = False

    save_video(all_videos, f"{save_results_dir}/video_Wan2.1-Fun-combined.mp4", fps=16, quality=5)
      
    print(f"Generation completed for scene {scene_name}.")

json.dump({
    "average_clip_retrieval_time": all_scene_avg_clip_retrieval_time / (len(test_dataloader) - 1),
    "average_clip_dit_time": all_scene_avg_clip_dit_time / (len(test_dataloader) - 1),
    "average_clip_add_time": all_scene_avg_clip_add_time / (len(test_dataloader) - 1),
    "average_clip_gen_time": all_scene_avg_clip_gen_time / (len(test_dataloader) - 1),
}, open(f"{base_results_dir}/performance_log.json", 'w'), indent=4)

# print(f"Average retrieval time across all scenes: {all_scene_acc_retrieval_time / len(test_dataloader):.4f} seconds per clip.")
# print(f"Average generation time across all scenes: {all_scene_gen_time / len(test_dataloader):.4f} seconds per clip.")

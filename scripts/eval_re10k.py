import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from base_diffsynth import save_video, load_state_dict
from base_diffsynth.pipelines.wan_video_mem_new import WanVideoPipeline, ModelConfig
import json
from frame_memory_retrieval import KeyFrameMemoryBankLearnedOccRetrieval

from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from accelerate import Accelerator

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
    def __init__(self, metadata_path, caption_path, rank=0, world_size=1, test_num=200):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        all_lines = [path.strip() for path in lines if path.strip()][:test_num]
        self.all_scene_paths = all_lines[rank::world_size]

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
            "captions": caption_clips,
            "scene_name": scene_name
        }

def test_data_loader(metadata_path, caption_path, rank=0, world_size=1, test_num=200):
    dataset = ValidationDataset(metadata_path, caption_path=caption_path, rank=rank, world_size=world_size, test_num=test_num)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config-path",
        default="configs/model_conf_wan_fun_i2v_1.3b.json",
        help="Path to the JSON file that lists WanVideo model config paths.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="./trained_models/step-11000.safetensors",
        help="Path to the trained checkpoint/safetensors file.",
    )
    parser.add_argument(
        "--retrieval-ckpt-path",
        default="./trained_models/ckpt_0000000000016000.pt",
        help="Path to the trained checkpoint/safetensors file.",
    )
    parser.add_argument(
        "--metadata-path",
        default="/mnt/data/jiali/DATASETS/re10k_processed/test/re10k_test_list.txt",
        help="Path to the test metadata list.",
    )
    parser.add_argument(
        "--caption-path",
        default='configs/re10k_test_captions.json',
        help="Path to the caption JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/i3dm_re10K_eval_200",
        help="Directory where generated videos, frames, and logs will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_name_template = "video_{count}.mp4"
    gt_video_name_template = "gt_video_{count}.mp4"
    video_frames_dir_template = "video_frames_{count}"
    gt_frames_dir_template = "gt_frames_{count}"
    frame_name_template = "video_frame_{index}.png"
    combined_video_name = "video_combined.mp4"
    combined_gt_video_name = "gt_video_combined.mp4"

    accelerator = Accelerator()

    device = accelerator.device
    if device.type == 'cuda' and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    model_configs = []
    with open(args.model_config_path, 'r', encoding='utf-8') as f:
        model_path_loaded = json.load(f)
    model_configs += [ModelConfig(path=path) for path in model_path_loaded]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )

    state_dict = load_state_dict(args.checkpoint_path, device=pipe.device, torch_dtype=pipe.torch_dtype)

    lora_state_dict = {k: v for k, v in state_dict.items() if 'lora' in k}
    pipe.load_lora(pipe.dit, state_dict=lora_state_dict, alpha=1)

    other_state_dict = {k.removeprefix("pipe."): v for k, v in state_dict.items() if 'lora' not in k}
    pipe.load_state_dict(other_state_dict, strict=False)

    pipe.enable_vram_management()

    base_results_dir = args.output_dir
    if accelerator.is_main_process:
        os.makedirs(base_results_dir, exist_ok=True)

    test_dataloader = test_data_loader(
                    args.metadata_path,
                    caption_path=args.caption_path,
                    rank=accelerator.process_index,
                    world_size=accelerator.num_processes,
                    )

    ctx_st, ctx_ed = 7, 11
    ctx_len = 4
    num_per_clip = 77
    mem_bank = KeyFrameMemoryBankLearnedOccRetrieval(retrieval_ckpt_path=args.retrieval_ckpt_path)
    for data in tqdm(test_dataloader):
        test_input_imgs = data["input_images"]
        test_input_c2ws = data["input_c2ws"]
        test_input_intrinsics = data["input_intrinsics"]
        captions = data["captions"]
        scene_name = data["scene_name"]

        print(f"Scene {scene_name} begin generation...")
        save_results_dir = os.path.join(base_results_dir, scene_name)
        os.makedirs(save_results_dir, exist_ok=True)

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
        replace_first_frame = True
        for ind in tqdm(range(len(clips_captions))):
            count += 1

            cur_traj = clips_trajs[ind].to(accelerator.device)
            cur_fxfycxcy = clips_fxfycxcys[ind].to(accelerator.device)
            cur_rgb = clips_rgbs[ind].to(accelerator.device)
            cur_caption = clips_captions[ind]

            retrieved_result = mem_bank.retrieve_top_k_cameras_learned_use(
                query_c2w=cur_traj[::4],
                query_fxfycxcy=cur_fxfycxcy[::4],
                k=ctx_len,
            )
            print("Retrieved frames indices:", retrieved_result)

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
                target_nvs_image=cur_rgb[::4],

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

            if all_videos:
                all_videos.extend(video[1:])
            else:
                all_videos.extend(video)
            save_video(video, os.path.join(save_results_dir, video_name_template.format(count=count)), fps=16, quality=5)
            save_video([tensor2rgb(v) for v in cur_rgb], os.path.join(save_results_dir, gt_video_name_template.format(count=count)), fps=16, quality=5)

            video_tensor = torch.stack([torch.from_numpy(np.array(im) / 255.0).permute(2,0,1).float() for im in video], dim=0)
            ref_img = video_tensor[-1]

            video_frames_dir = os.path.join(save_results_dir, video_frames_dir_template.format(count=count))
            gt_frames_dir = os.path.join(save_results_dir, gt_frames_dir_template.format(count=count))
            os.makedirs(video_frames_dir, exist_ok=True)
            os.makedirs(gt_frames_dir, exist_ok=True)
            for i in range(len(video_tensor)):
                save_image(video_tensor[i], os.path.join(video_frames_dir, frame_name_template.format(index=i)))
                save_image(cur_rgb[i], os.path.join(gt_frames_dir, frame_name_template.format(index=i)))

            mem_bank.add_key_frames(
                frames_c2w=cur_traj.reshape(-1,4,4)[::4],
                frames_fxfycxcy=cur_fxfycxcy.reshape(-1,4)[::4],
                frames_rgb=video_tensor.reshape(-1,video_tensor.shape[1],video_tensor.shape[2],video_tensor.shape[3])[::4],
            )
            print(f"Memory bank size: {mem_bank.frames_num} frames.")

        save_video(all_videos, os.path.join(save_results_dir, combined_video_name), fps=16, quality=5)

        all_gt_frames = []
        for clip in clips_rgbs:
            gt_clip_frames = [tensor2rgb(f) for f in clip]
            if all_gt_frames:
                all_gt_frames.extend(gt_clip_frames[1:])
            else:
                all_gt_frames.extend(gt_clip_frames)
        save_video(all_gt_frames, os.path.join(save_results_dir, combined_gt_video_name), fps=16, quality=5)

        print(f"Generation completed for scene {scene_name}.")


if __name__ == "__main__":
    main()

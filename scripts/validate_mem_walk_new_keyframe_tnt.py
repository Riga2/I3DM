import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def interpolate_trajectory(c2ws, intrinsics, multiplier=4):
    """
    Interpolate camera trajectory and intrinsics.
    Args:
        c2ws: (N, 4, 4) torch.Tensor
        intrinsics: (N, 4) torch.Tensor, [fx, fy, cx, cy]
        multiplier: int, number of intervals between frames (multiplier-1 new frames)
    """
    if multiplier <= 1:
        return c2ws, intrinsics

    device = c2ws.device
    dtype = c2ws.dtype
    
    c2ws_np = c2ws.detach().cpu().numpy()
    intrinsics_np = intrinsics.detach().cpu().numpy()
    
    N = c2ws.shape[0]
    
    new_c2ws_list = []
    new_intrinsics_list = []
    
    times = [0, 1]
    interp_times = np.linspace(0, 1, multiplier, endpoint=False)
    
    for i in range(N - 1):
        c2w_start = c2ws_np[i]
        c2w_end = c2ws_np[i+1]
        
        # Rotation
        rot_start = c2w_start[:3, :3]
        rot_end = c2w_end[:3, :3]
        
        key_rots = R.from_matrix([rot_start, rot_end])
        slerp = Slerp(times, key_rots)
        interp_rots = slerp(interp_times).as_matrix()
        
        # Translation
        trans_start = c2w_start[:3, 3]
        trans_end = c2w_end[:3, 3]
        
        # Intrinsics
        k_start = intrinsics_np[i]
        k_end = intrinsics_np[i+1]
        
        for j, t in enumerate(interp_times):
            interp_t = trans_start + (trans_end - trans_start) * t
            interp_k = k_start + (k_end - k_start) * t
            
            new_c2w = np.eye(4)
            new_c2w[:3, :3] = interp_rots[j]
            new_c2w[:3, 3] = interp_t
            
            new_c2ws_list.append(new_c2w)
            new_intrinsics_list.append(interp_k)
            
    # Add last frame
    new_c2ws_list.append(c2ws_np[-1])
    new_intrinsics_list.append(intrinsics_np[-1])
    
    new_c2ws = torch.tensor(np.stack(new_c2ws_list), device=device, dtype=dtype)
    new_intrinsics = torch.tensor(np.stack(new_intrinsics_list), device=device, dtype=dtype)
    
    return new_c2ws, new_intrinsics

class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, 'r') as f:
            lines = f.read().splitlines()
        self.all_scene_paths = [path.strip() for path in lines if path.strip()][-1:]

        # self.check_scene_names = [
        #     # '0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3',
        #     '0a485338bbdaf19ba9090b874bb36ef0599a9c9a475a382c22903cf5981c6ea6',
        #     '0bfdd020cf475b9c68e4b469d1d1a2d0cad303eefe8b78fb2307855afdaac8be',
        #     '032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7',
        #     '0569e83fdc248a51fc0ab082ce5e2baff15755c53c207f545e6d02d91f01d166',
        #     '06da796666297fe4c683c231edf56ec00148a6a52ab5bb159fe1be31f53a58df',
        # ]

        # base_dir = '/mnt/afse1/DATASET/DL3DV/metadata'
        # self.all_scene_paths = [os.path.join(base_dir, f'{name}.json') for name in self.check_scene_names]

        caption_path = os.path.join(os.path.dirname(metadata_path), 'caption_test', 'video_captions_refined.json')
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
# other_state_dict = {k.removeprefix("pipe."): v for k, v in state_dict.items() if (('lora' not in k) and ('nvs_model' not in k))}
pipe.load_state_dict(other_state_dict, strict=False)

pipe.enable_vram_management()


base_results_dir = './results/Keyframes_SceneDec_ft_s11000_occSelection_loop'
os.makedirs(base_results_dir, exist_ok=True)

test_dataloader = test_data_loader(
                '/mnt/afse2/DATASET/tnt_data/full_list.txt'
                )

ctx_st, ctx_ed = 1, 5
ctx_len = ctx_ed - ctx_st
num_per_clip = 77
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
    mem_bank = KeyFrameMemoryBankLearnedOccRetrieval(n_sample_points=100000)
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

    # Interpolate trajectory for smoother movement
    interpolation_multiplier = 15
    candidate_c2ws, candidate_fxfycxcy = interpolate_trajectory(candidate_c2ws, candidate_fxfycxcy, multiplier=interpolation_multiplier)
    candidate_rgbs = candidate_rgbs.repeat_interleave(interpolation_multiplier, dim=0)[:candidate_c2ws.shape[0]] # Dummy RGBs matching length
        
    # explore frames: 310/390
    candidate_c2ws = candidate_c2ws[:310]
    candidate_fxfycxcy = candidate_fxfycxcy[:310]
    candidate_rgbs = candidate_rgbs[:310]

    # Recalculate num_clip based on interpolated length
    forward_num_clip = (candidate_c2ws.shape[0] - 1) // 76

    if len(captions) < forward_num_clip:
        captions.extend([captions[-1]] * (forward_num_clip - len(captions)))
    else:
        captions = captions[:forward_num_clip]

    clips_trajs = []
    clips_fxfycxcys = []
    # clips_rgbs = []
    # Process Clips
    for i in range(forward_num_clip):
        clips_trajs.append(candidate_c2ws[i*76 : i*76+77])
        clips_fxfycxcys.append(candidate_fxfycxcy[i*76 : i*76+77])
        # clips_rgbs.append(candidate_rgbs[i*76 : i*76+77])
    # Process Reverse Clips
    for i in range(forward_num_clip - 1, -1, -1):
        clips_trajs.append(torch.flip(candidate_c2ws[i*76 : i*76+77], [0]))
        clips_fxfycxcys.append(torch.flip(candidate_fxfycxcy[i*76 : i*76+77], [0]))
        # clips_rgbs.append(torch.flip(candidate_rgbs[i*76 : i*76+77], [0]))
    clips_captions = captions + captions[::-1]

    ref_img = test_input_imgs[ctx_ed-1]

    num_clip_total = len(clips_captions)

    count = 0
    all_videos = []
    scene_logs = []
    replace_first_frame = True
    all_extrinsics = []
    all_intrinsics = []
    for ind in tqdm(range(num_clip_total)):
        count += 1

        cur_traj = clips_trajs[ind].to('cuda')
        cur_fxfycxcy = clips_fxfycxcys[ind].to('cuda')
        # cur_rgb = clips_rgbs[ind].to('cuda')
        cur_caption = clips_captions[ind]

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

        if count > 1:
            all_extrinsics.extend(extrinsics_data[1:])
            all_intrinsics.extend(intrinsics_data[1:])          
        else:
            all_extrinsics.extend(extrinsics_data)
            all_intrinsics.extend(intrinsics_data)

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
        # retrieved_result, retrieved_ratios = mem_bank.retrieve_top_k_cameras(
        retrieved_result, retrieved_ratios = mem_bank.retrieve_top_k_cameras_learned_use(
            query_c2w=cur_traj[::4],
            query_fxfycxcy=cur_fxfycxcy[::4],
            k=ctx_len,
        )
        print("Retrieved frames indices:", retrieved_result)
        print("Retrieved ratios:", retrieved_ratios)

        covered_ratios, max_single_ratios, best_mem_indices = mem_bank.calculate_query_coverage(
            query_c2w=cur_traj[::4],
            query_fxfycxcy=cur_fxfycxcy[::4],
            selected_mem_indices=retrieved_result
        )
        print("Covered ratios:", covered_ratios)
        print("Max single covered:", max_single_ratios)
        print("Best memory indices:", best_mem_indices)

        scene_logs.append({
            "clip_id": count,
            "retrieved_indices": retrieved_result,
            "retrieved_ratios": [r.item() if hasattr(r, 'item') else r for r in retrieved_ratios],
            "covered_ratios": covered_ratios.tolist() if hasattr(covered_ratios, "tolist") else covered_ratios,
            "max_single_ratios": max_single_ratios.tolist() if hasattr(max_single_ratios, "tolist") else max_single_ratios,
            "best_mem_indices": best_mem_indices
        })
        with open(os.path.join(save_results_dir, "retrieval_logs.json"), "w") as f:
            json.dump(scene_logs, f, indent=4)
        
        retrieved_data = mem_bank.get_retrieved_data_frames(retrieved_result)
        # save_image(retrieved_data['frames_rgb'], f"{save_results_dir}/retrieved_frame_example_{count}.png")

        union_c2ws = torch.cat([retrieved_data['frames_c2ws'], cur_traj[::4]], dim=0)
        union_c2ws_processed = preprocess_poses(union_c2ws)

        # data_input_batch = {
        #     "image": retrieved_data['frames_rgb'][None].to('cuda', dtype=torch.bfloat16),
        #     "c2w": union_c2ws_processed[:ctx_len][None].to('cuda', dtype=torch.bfloat16),
        #     "fxfycxcy": retrieved_data['frames_fxfycxcy'][None].to('cuda', dtype=torch.bfloat16),
        # }

        # data_target_batch = {
        #     "c2w": union_c2ws_processed[ctx_len:][None].to('cuda', dtype=torch.bfloat16),
        #     "fxfycxcy": cur_fxfycxcy[::4][None].to('cuda', dtype=torch.bfloat16),
        # }
        # with autocast(device_type="cuda", dtype=torch.bfloat16):
        #     nvs_ret, input_ret = nvs_model(data_input_batch, data_target_batch)

        # os.makedirs(f"{save_results_dir}/debug_clip_{count}", exist_ok=True)
        # save_image(retrieved_data['frames_rgb'], f"{save_results_dir}/debug_clip_{count}/nvs_input_debug.png")
        # save_image(nvs_ret[0], f"{save_results_dir}/debug_clip_{count}/nvs_res_debug.png")
        # save_image(cur_rgb[::4], f"{save_results_dir}/debug_clip_{count}/gt_res_debug.png")

        # for t_i in range(input_ret.shape[0]):
        #     save_image(input_ret[t_i], f"{save_results_dir}/debug_clip_{count}/nvs_input_frame_{t_i}.png")

        video = pipe(
            height=352,
            width=640,
            num_frames=num_per_clip,

            input_nvs_image=retrieved_data['frames_rgb'],
            input_nvs_c2w=union_c2ws_processed[:ctx_len],
            input_nvs_fxfycxcy=retrieved_data['frames_fxfycxcy'],
    
            target_nvs_c2w=union_c2ws_processed[ctx_len:],
            target_nvs_fxfycxcy=cur_fxfycxcy[::4],
            target_nvs_image=None, # cur_rgb[::4],
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
        save_video(video, f"{save_results_dir}/video_Wan2.1-Fun-{count}.mp4", fps=16, quality=5)
        # save_video([tensor2rgb(v) for v in cur_rgb], f"{save_results_dir}/gt_video_Wan2.1-Fun-{count}.mp4", fps=16, quality=5)

        video_tensor = torch.stack([torch.from_numpy(np.array(im) / 255.0).permute(2,0,1).float() for im in video], dim=0)
        ref_img = video_tensor[-1]

        for i in range(len(video_tensor)):
            os.makedirs(f"{save_results_dir}/video_frames_{count}", exist_ok=True)
            os.makedirs(f"{save_results_dir}/gt_frames_{count}", exist_ok=True)
            save_image(video_tensor[i], f"{save_results_dir}/video_frames_{count}/video_frame_{i}.png")
            # save_image(cur_rgb[i], f"{save_results_dir}/gt_frames_{count}/video_frame_{i}.png")

        mem_bank.add_key_frames(
            frames_c2w=cur_traj.reshape(-1,4,4)[::4],
            frames_fxfycxcy=cur_fxfycxcy.reshape(-1,4)[::4],
            # frames_rgb=cur_rgb.reshape(-1,3,352,640)[::4],
            frames_rgb=video_tensor.reshape(-1,video_tensor.shape[1],video_tensor.shape[2],video_tensor.shape[3])[::4],
        )
        print(f"Memory bank size: {mem_bank.frames_num} frames.")


    assert len(all_extrinsics) == 1+76*num_clip_total, f"{len(all_extrinsics)} vs {1+76*num_clip_total}"
    with open(f"{save_results_dir}/gt_cam.json", "w") as f:
        json.dump({
            "extrinsics": all_extrinsics,
            "intrinsics": all_intrinsics
        }, f, indent=4)
    
    save_video(all_videos, f"{save_results_dir}/video_Wan2.1-Fun-combined.mp4", fps=16, quality=5)
    
    # all_gt_frames = []
    # for clip in clips_rgbs:
    #     all_gt_frames.extend([tensor2rgb(f) for f in clip])
    # save_video(all_gt_frames, f"{save_results_dir}/gt_video_Wan2.1-Fun-combined.mp4", fps=16, quality=5)

    print(f"Generation completed for scene {scene_name}.")


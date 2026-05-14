import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from frame_memory_retrieval_learned_fast import KeyFrameMemoryBankLearnedOccRetrievalFast
from frame_memory_retrieval_optimized import KeyFrameMemoryBankLearnedOccRetrieval


NEGATIVE_PROMPT = "色调艳丽，过曝，细节模糊不清，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，畸形的，杂乱的背景"


def sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


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
        if abs(aspect_ratio - (16 / 9)) > 0.01:
            raise ValueError(f"Unexpected aspect ratio for {cur_image_path}")

        if original_image_w != target_w_resize:
            image = image.resize((target_w_resize, target_h_resize), resample=Image.LANCZOS)

        start_h = (target_h_resize - target_h_crop) // 2
        image = image.crop((0, start_h, target_w_resize, start_h + target_h_crop))
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        fxfycxcy = np.array(cur_frame["fxfycxcy"])
        resize_ratio_x = target_w_resize / original_image_w
        resize_ratio_y = target_h_resize / original_image_h
        fxfycxcy *= np.array([resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y])
        fxfycxcy[3] -= start_h
        intrinsics.append(torch.from_numpy(fxfycxcy).float())
        images.append(image)

    w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
    c2ws = torch.from_numpy(np.linalg.inv(w2cs)).float()
    return torch.stack(images, dim=0), torch.stack(intrinsics, dim=0), c2ws


def preprocess_poses(in_c2ws: torch.Tensor, scene_scale_factor=1.35):
    center = in_c2ws[:, :3, 3].mean(0)
    avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
    avg_down = in_c2ws[:, :3, 1].mean(0)
    avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
    avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

    avg_pose = torch.eye(4, device=in_c2ws.device)
    avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    in_c2ws = torch.linalg.inv(avg_pose) @ in_c2ws

    scene_scale = scene_scale_factor * torch.max(torch.abs(in_c2ws[:, :3, 3]))
    in_c2ws[:, :3, 3] /= scene_scale
    return in_c2ws


def tensor2rgb(image_tensor):
    image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
    image_numpy = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    return np.transpose(image_numpy, (1, 2, 0))


def load_scenes(metadata_path: str, max_scenes: int):
    with open(metadata_path, "r") as f:
        scene_paths = [line.strip() for line in f.read().splitlines() if line.strip()]
    if max_scenes > 0:
        scene_paths = scene_paths[:max_scenes]

    scenes = []
    for scene_path in scene_paths:
        data_json = json.load(open(scene_path, "r"))
        frames = data_json["frames"]
        image_paths = [frame["image_path"] for frame in frames]
        images, intrinsics, c2ws = preprocess_frames(frames, image_paths)
        scenes.append(
            {
                "scene_name": data_json["scene_name"],
                "images": images,
                "intrinsics": intrinsics,
                "c2ws": c2ws,
            }
        )
    return scenes


def build_clips(images, intrinsics, c2ws, ctx_ed: int, clip_len: int, include_reverse: bool):
    num_clip = (len(images) - ctx_ed) // (clip_len - 1)
    candidate_c2ws = c2ws[ctx_ed - 1 : ctx_ed + num_clip * (clip_len - 1)]
    candidate_fxfycxcy = intrinsics[ctx_ed - 1 : ctx_ed + num_clip * (clip_len - 1)]
    candidate_rgbs = images[ctx_ed - 1 : ctx_ed + num_clip * (clip_len - 1)]

    clips = []
    for i in range(num_clip):
        s = i * (clip_len - 1)
        e = s + clip_len
        clips.append((candidate_c2ws[s:e], candidate_fxfycxcy[s:e], candidate_rgbs[s:e], "forward"))
    if include_reverse:
        for i in range(num_clip - 1, -1, -1):
            s = i * (clip_len - 1)
            e = s + clip_len
            clips.append(
                (
                    torch.flip(candidate_c2ws[s:e], [0]),
                    torch.flip(candidate_fxfycxcy[s:e], [0]),
                    torch.flip(candidate_rgbs[s:e], [0]),
                    "reverse",
                )
            )
    return clips


def expand_clips_for_target_length(clips, clip_len: int, target_generated_frames: int, repeat_clip_cycles: int):
    if not clips:
        return clips

    repeats = max(repeat_clip_cycles, 1)
    if target_generated_frames > 0:
        repeats = max(repeats, int(np.ceil(target_generated_frames / float(len(clips) * clip_len))))

    expanded = []
    for cycle_id in range(repeats):
        for cur_traj, cur_fxfycxcy, cur_rgb, direction in clips:
            expanded.append((cur_traj, cur_fxfycxcy, cur_rgb, f"{direction}_cycle_{cycle_id + 1}"))
            if target_generated_frames > 0 and len(expanded) * clip_len >= target_generated_frames:
                return expanded
    return expanded


def build_mem_bank(args, device: torch.device):
    common = {
        "n_sample_points": args.n_sample_points,
        "device": device,
    }
    if args.method in {"fast", "full_optimized"}:
        return KeyFrameMemoryBankLearnedOccRetrievalFast(
            **common,
            candidate_pool_size=args.candidate_pool_size,
            proposal_points=args.proposal_points,
            always_include_recent=args.always_include_recent,
            learned_batch_size=args.batch_size,
        )
    if args.method == "original":
        return KeyFrameMemoryBankLearnedOccRetrieval(**common)
    raise ValueError(f"Unsupported method: {args.method}")


def build_pipe(args, device: torch.device):
    from base_diffsynth import load_state_dict
    from base_diffsynth.pipelines.wan_video_mem_new import ModelConfig, WanVideoPipeline

    with open(args.model_config, "r", encoding="utf-8") as f:
        model_paths = json.load(f)
    model_configs = [ModelConfig(path=path) for path in model_paths]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
    )
    state_dict = load_state_dict(args.generator_ckpt, device=pipe.device, torch_dtype=pipe.torch_dtype)
    lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
    pipe.load_lora(pipe.dit, state_dict=lora_state_dict, alpha=1)
    other_state_dict = {k.removeprefix("pipe."): v for k, v in state_dict.items() if "lora" not in k}
    pipe.load_state_dict(other_state_dict, strict=False)
    pipe.enable_vram_management()
    return pipe


def retrieve(mem_bank, args, query_c2w, query_fxfycxcy):
    if args.method == "fast":
        return mem_bank.retrieve_camera_occ_fast(
            query_c2w=query_c2w,
            query_fxfycxcy=query_fxfycxcy,
            k=args.k,
            batch_size=args.batch_size,
            candidate_pool_size=args.candidate_pool_size,
            return_profile=True,
        )
    if args.method == "full_optimized":
        return mem_bank.retrieve_camera_occ_full_optimized(
            query_c2w=query_c2w,
            query_fxfycxcy=query_fxfycxcy,
            k=args.k,
            batch_size=args.batch_size,
            disable_checkpoint=not args.keep_checkpoint,
            return_profile=True,
        )

    t0 = mem_bank._profile_now() if hasattr(mem_bank, "_profile_now") else time.perf_counter()
    indices = mem_bank.retrieve_camera_occ_perf_test(
        query_c2w=query_c2w,
        query_fxfycxcy=query_fxfycxcy,
        k=args.k,
        batch_size=args.batch_size,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    profile = {
        "memory_size": mem_bank.frames_num,
        "proposal_candidates": mem_bank.frames_num - 1,
        "proposal_time": 0.0,
        "learned_time": 0.0,
        "selection_time": 0.0,
        "total_time": time.perf_counter() - t0,
    }
    return indices, profile


def write_rows(csv_path: str, rows: List[Dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_summary(csv_rows: List[Dict], summary_path: str, exclude_first_scene: bool = True):
    if exclude_first_scene and csv_rows:
        first_scene = csv_rows[0]["scene"]
        csv_rows = [row for row in csv_rows if row["scene"] != first_scene]

    grouped = defaultdict(list)
    for row in csv_rows:
        grouped[int(row["memory_size_before"])].append(row)

    summary = []
    metric_names = [
        "retrieval_time",
        "proposal_time",
        "learned_time",
        "selection_time",
        "dit_time",
        "add_time",
        "total_clip_time",
    ]
    for memory_size in sorted(grouped):
        rows = grouped[memory_size]
        item = {
            "memory_size_before": memory_size,
            "num_samples": len(rows),
            "excluded_first_scene": exclude_first_scene,
        }
        for metric in metric_names:
            values = np.array([float(row[metric]) for row in rows], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std())
        summary.append(item)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"memory_scaling_{args.method}.csv")
    json_path = os.path.join(args.output_dir, f"memory_scaling_{args.method}.json")
    summary_path = os.path.join(args.output_dir, f"memory_scaling_{args.method}_summary.json")

    scenes = load_scenes(args.metadata_path, args.max_scenes)
    pipe = build_pipe(args, device) if args.run_generation else None
    rows = []

    for scene in tqdm(scenes, desc="scenes"):
        mem_bank = build_mem_bank(args, device)
        mem_bank.reset()
        images = scene["images"]
        c2ws = scene["c2ws"]
        intrinsics = scene["intrinsics"]
        mem_bank.add_key_frames(
            frames_c2w=c2ws[args.ctx_st : args.ctx_ed],
            frames_fxfycxcy=intrinsics[args.ctx_st : args.ctx_ed],
            frames_rgb=images[args.ctx_st : args.ctx_ed],
        )

        ref_img = images[args.ctx_ed - 1].to(device)
        clips = build_clips(images, intrinsics, c2ws, args.ctx_ed, args.clip_len, args.include_reverse)
        clips = expand_clips_for_target_length(
            clips=clips,
            clip_len=args.clip_len,
            target_generated_frames=args.target_generated_frames,
            repeat_clip_cycles=args.repeat_clip_cycles,
        )
        if args.max_clips > 0:
            clips = clips[: args.max_clips]

        for clip_id, (cur_traj, cur_fxfycxcy, cur_rgb, direction) in enumerate(tqdm(clips, desc=scene["scene_name"]), 1):
            cur_traj = cur_traj.to(device)
            cur_fxfycxcy = cur_fxfycxcy.to(device)
            cur_rgb = cur_rgb.to(device)
            memory_size_before = mem_bank.frames_num

            t_clip = sync_time(device)
            retrieved_indices, profile = retrieve(mem_bank, args, cur_traj[:: args.query_stride], cur_fxfycxcy[:: args.query_stride])
            retrieved_data = mem_bank.get_retrieved_data_frames(retrieved_indices)
            retrieval_time = sync_time(device) - t_clip

            dit_time = 0.0
            add_time = 0.0
            if args.run_generation:
                union_c2ws = torch.cat([retrieved_data["frames_c2ws"], cur_traj[:: args.query_stride]], dim=0)
                union_c2ws_processed = preprocess_poses(union_c2ws)
                t_dit = sync_time(device)
                video = pipe(
                    height=352,
                    width=640,
                    num_frames=args.clip_len,
                    input_nvs_image=retrieved_data["frames_rgb"],
                    input_nvs_c2w=union_c2ws_processed[: args.k],
                    input_nvs_fxfycxcy=retrieved_data["frames_fxfycxcy"],
                    target_nvs_c2w=union_c2ws_processed[args.k :],
                    target_nvs_fxfycxcy=cur_fxfycxcy[:: args.query_stride],
                    target_nvs_image=cur_rgb[:: args.query_stride],
                    cam_c2w=cur_traj,
                    cam_intric=cur_fxfycxcy,
                    reference_image=ref_img,
                    prompt=args.prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    seed=args.seed,
                    tiled=True,
                    return_weights=False,
                )
                dit_time = sync_time(device) - t_dit
                video[0] = tensor2rgb(ref_img)
                video_tensor = torch.stack(
                    [torch.from_numpy(np.array(im) / 255.0).permute(2, 0, 1).float() for im in video],
                    dim=0,
                ).to(device)
                ref_img = video_tensor[-1]
                frames_to_add = video_tensor[:: args.query_stride]
            else:
                frames_to_add = cur_rgb[:: args.query_stride]

            t_add = sync_time(device)
            mem_bank.add_key_frames(
                frames_c2w=cur_traj.reshape(-1, 4, 4)[:: args.query_stride],
                frames_fxfycxcy=cur_fxfycxcy.reshape(-1, 4)[:: args.query_stride],
                frames_rgb=frames_to_add,
            )
            add_time = sync_time(device) - t_add
            total_clip_time = sync_time(device) - t_clip

            row = {
                "scene": scene["scene_name"],
                "clip_id": clip_id,
                "direction": direction,
                "method": args.method,
                "memory_size_before": memory_size_before,
                "memory_size_after": mem_bank.frames_num,
                "candidate_pool_size": args.candidate_pool_size if args.method == "fast" else memory_size_before - 1,
                "proposal_candidates": profile.get("proposal_candidates", 0),
                "retrieval_time": retrieval_time,
                "profile_total_time": profile.get("total_time", retrieval_time),
                "proposal_time": profile.get("proposal_time", 0.0),
                "learned_time": profile.get("learned_time", 0.0),
                "selection_time": profile.get("selection_time", 0.0),
                "dit_time": dit_time,
                "add_time": add_time,
                "total_clip_time": total_clip_time,
                "retrieved_indices": json.dumps(retrieved_indices),
            }
            rows.append(row)

            if len(rows) >= args.flush_every:
                write_rows(csv_path, rows)
                rows.clear()

    write_rows(csv_path, rows)
    with open(csv_path, "r") as f:
        csv_rows = list(csv.DictReader(f))
    with open(json_path, "w") as f:
        json.dump(csv_rows, f, indent=2)
    write_summary(csv_rows, summary_path, exclude_first_scene=not args.include_first_scene_in_summary)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark memory-bank scaling for learned retrieval.")
    parser.add_argument("--metadata-path", default="/mnt/afse2/DATASET/re10k_preprocessed/test/full_list_F239.txt")
    parser.add_argument("--output-dir", default="./results/memory_retrieval_scaling")
    parser.add_argument("--method", choices=["fast", "full_optimized", "original"], default="full_optimized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-scenes", type=int, default=5)
    parser.add_argument("--max-clips", type=int, default=-1)
    parser.add_argument("--ctx-st", type=int, default=7)
    parser.add_argument("--ctx-ed", type=int, default=11)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--clip-len", type=int, default=77)
    parser.add_argument("--query-stride", type=int, default=4)
    parser.add_argument("--include-reverse", action="store_true", default=True)
    parser.add_argument("--target-generated-frames", type=int, default=1000)
    parser.add_argument("--repeat-clip-cycles", type=int, default=1)
    parser.add_argument("--n-sample-points", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-pool-size", type=int, default=64)
    parser.add_argument("--proposal-points", type=int, default=8192)
    parser.add_argument("--always-include-recent", type=int, default=8)
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--model-config", default="configs/model_conf_wan_fun_i2v_1.3b.json")
    parser.add_argument("--generator-ckpt", default="./trained_models/Keyframes_SceneDec_ft/step-11000.safetensors")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--include-first-scene-in-summary", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

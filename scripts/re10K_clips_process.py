import json
import os
import argparse
from tqdm import tqdm

def frame_sort_key(frame):
    image_path = frame.get("image_path", "")
    stem = os.path.splitext(os.path.basename(image_path))[0]
    try:
        return int(stem)
    except ValueError:
        return stem


def resolve_dataset_paths(base_dataset_dir, split, clip_length):
    base_dataset_dir = os.path.abspath(base_dataset_dir)
    if os.path.exists(os.path.join(base_dataset_dir, "full_list.txt")):
        split_dir = base_dataset_dir
    else:
        split_dir = os.path.join(base_dataset_dir, split)

    clip_tag = f"F{clip_length}"
    input_list_path = os.path.join(split_dir, "full_list.txt")
    output_list_path = os.path.join(split_dir, f"full_list_{clip_tag}_clips.txt")
    output_metadata_dir = os.path.join(split_dir, f"clips_metadata_{clip_tag}")
    return input_list_path, output_list_path, output_metadata_dir

def process_scenes(input_list_path, output_list_path, output_metadata_dir, clip_length):
    """
    Reads a list of scene JSON paths.
    For scenes with > clip_length frames, splits them into clips of clip_length.
    Saves new clip JSONs to output_metadata_dir.
    Writes paths of new clips to output_list_path.
    """
    print(f"Reading scene list from {input_list_path}")
    try:
        with open(input_list_path, 'r') as f:
            scene_paths = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_list_path}")
        return

    # Create output metadata directory
    os.makedirs(output_metadata_dir, exist_ok=True)

    new_scene_paths = []
    print(f"Processing scenes with > {clip_length} frames...")
    
    count_processed = 0
    
    for scene_path in tqdm(scene_paths):
        target_path = scene_path
        if not os.path.exists(target_path):
             # Try relative to input list path
             rel_path = os.path.join(os.path.dirname(input_list_path), scene_path)
             if os.path.exists(rel_path):
                 target_path = rel_path
             else:
                 print(f"Warning: File not found: {scene_path}")
                 continue
        
        try:
            with open(target_path, 'r') as f:
                data = json.load(f)
            
            frames = []
            is_dict = False
            if isinstance(data, list):
                frames = data
            elif isinstance(data, dict) and 'frames' in data:
                frames = data['frames']
                is_dict = True
            else:
                continue

            frames = sorted(frames, key=frame_sort_key)

            num_frames = len(frames)
            if num_frames <= clip_length:
                continue

            count_processed += 1

            # Generate clips
            # Strategy: Stride = clip_length. Ensure last clip covers the end.
            # This creates non-overlapping clips for the most part, 
            # and one overlapping clip at the end if there's a remainder.
            start_indices = list(range(0, num_frames, clip_length))
            
            valid_start_indices = [i for i in start_indices if i + clip_length <= num_frames]
            
            # If the last valid clip doesn't reach the end, add one more clip ending at num_frames
            if not valid_start_indices or valid_start_indices[-1] + clip_length < num_frames:
                valid_start_indices.append(num_frames - clip_length)
            
            valid_start_indices = sorted(list(set(valid_start_indices)))

            # Use original filename stem for new files
            base_name = os.path.splitext(os.path.basename(scene_path))[0]
            
            for i, start_idx in enumerate(valid_start_indices):
                end_idx = start_idx + clip_length
                clip_frames = frames[start_idx:end_idx]
                
                if len(clip_frames) != clip_length:
                    continue
                
                if is_dict:
                    new_data = data.copy()
                    new_data['frames'] = clip_frames
                else:
                    new_data = clip_frames
                
                new_filename = f"{base_name}_clip{i}.json"
                new_file_path = os.path.join(output_metadata_dir, new_filename)
                
                with open(new_file_path, 'w') as f:
                    json.dump(new_data, f, indent=4)
                
                new_scene_paths.append(new_file_path)

        except Exception as e:
            print(f"Error processing {scene_path}: {e}")

    print(f"Processed {count_processed} scenes.")
    print(f"Generated {len(new_scene_paths)} clips.")
    
    print(f"Writing new list to {output_list_path}")
    with open(output_list_path, 'w') as f:
        for p in new_scene_paths:
            f.write(p + '\n')
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Re10K scenes into fixed-length clips.")
    parser.add_argument(
        "--base_dataset_dir",
        type=str,
        help="Processed Re10K root directory, or the split directory that contains full_list.txt.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process when base_dataset_dir is the dataset root.")
    parser.add_argument("--clip_length", type=int, default=77, help="Length of each clip in frames.")

    args = parser.parse_args()
    input_list_path, output_list_path, output_metadata_dir = resolve_dataset_paths(
        args.base_dataset_dir,
        args.split,
        args.clip_length,
    )

    print(f"Base dataset dir: {os.path.abspath(args.base_dataset_dir)}")
    print(f"Input list: {input_list_path}")
    print(f"Output list: {output_list_path}")
    print(f"Output metadata dir: {output_metadata_dir}")
    process_scenes(input_list_path, output_list_path, output_metadata_dir, args.clip_length)

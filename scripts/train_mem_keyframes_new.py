import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from base_diffsynth import load_state_dict
from base_diffsynth.pipelines.wan_video_mem_new import WanVideoPipeline, ModelConfig
from base_diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, WandBModelLogger, launch_training_task, wan_parser
from base_diffsynth.trainers.unified_dataset import UnifiedDatasetRe10K_Keyframes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
from einops import rearrange
from easydict import EasyDict as edict
# from external.lvsm_model import LVSM_model_latent, LVSM_model_IMG
from torchvision.utils import save_image

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        extra_trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models, extra_trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

        # self.nvs_model = LVSM_model_IMG()

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}

        nvs_target_len = len(data["target_nvs_image"])

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters
            "input_video": data["target_src_image"],
            "num_frames": data["target_src_image"].shape[0],
            "height": data["target_src_image"].shape[2],
            "width": data["target_src_image"].shape[3],
            
            # context encoder inputs
            "input_nvs_image": data['input_nvs_image'],
            "input_nvs_c2w": data['input_nvs_c2w'],
            "input_nvs_fxfycxcy": data['input_nvs_fxfycxcy'],
            "debug_save_dir": None,
            # "debug_save_dir": "./models/train/Wan2.1-FUN-1.3B_mem_lora_debug/nvs_debug",

            # context decoder inputs
            "target_nvs_c2w": data['target_nvs_c2w'],
            "target_nvs_fxfycxcy": data['target_nvs_fxfycxcy'],
            "target_nvs_image": data['target_nvs_image'],

            "cam_c2w": data['target_src_c2w'],
            "cam_intric": data['target_src_fxfycxcy'],
            # "input_image": data['ref_img'],
            "reference_image": data['ref_img'],

            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        # for extra_input in self.extra_inputs:
        #     if extra_input == "input_image":
        #         inputs_shared["input_image"] = data["video"][0]
        #     elif extra_input == "end_image":
        #         inputs_shared["end_image"] = data["video"][-1]
        #     elif extra_input == "reference_image" or extra_input == "vace_reference_image":
        #         inputs_shared[extra_input] = data[extra_input][0]
        #     else:
        #         inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    dataset = UnifiedDatasetRe10K_Keyframes(
        metadata_path=args.dataset_metadata_path,
        caption_path=args.dataset_caption_path,
        repeat=args.dataset_repeat,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        extra_trainable_models=args.extra_trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    launch_training_task(dataset, model, args=args)
import os
from typing import List

import torch

from .utils import hash_state_dict_keys, init_weights_on_device, load_state_dict, split_state_dict_with_prefix
from .wan_video_animate_adapter import WanAnimateAdapter
from .wan_video_dit import WanModel
from .wan_video_dit_s2v import WanS2VModel
from .wan_video_image_encoder import WanImageEncoder
from .wan_video_mot import MotWanModel
from .wan_video_motion_controller import WanMotionControllerModel
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vace import VaceWanModel
from .wan_video_vae import WanVideoVAE, WanVideoVAE38
from .wav2vec import WanS2VAudioEncoder


# Wan-only model signatures used by the I3DM training and Re10K evaluation scripts.
# Format: (keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource)
MODEL_LOADER_CONFIGS = [
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5ec04e02b42d2580483ad69f4e76346a", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),
    (None, "5f90e66a0672219f12d9a626c8c21f61", ["wan_video_dit", "wan_video_vap"], [WanModel, MotWanModel], "diffusers"),
    (None, "a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "7a513e1f257a861512b1afd387a8ecd9", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    (None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    (None, "966cffdcc52f9c46c391768b27637614", ["wan_video_dit"], [WanS2VModel], "civitai"),
    (None, "9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    (None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    (None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
    (None, "dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    (None, "06be60f3a4526586d8431cd038a71486", ["wans2v_audio_encoder"], [WanS2VAudioEncoder], "civitai"),
    (None, "31fa352acb8a1b1d33cd8764273d80a2", ["wan_video_dit", "wan_video_animate_adapter"], [WanModel, WanAnimateAdapter], "civitai"),
]


def load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        else:
            raise ValueError(f"Unsupported model resource: {model_resource}")

        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(f"        This model is initialized with extra kwargs: {extra_kwargs}")
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}

        model_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()
        model.load_state_dict(model_state_dict, assign=True)
        model = model.to(dtype=model_dtype, device=device)
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs=None):
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}
        for metadata in model_loader_configs or []:
            self.add_model_metadata(*metadata)

    def add_model_metadata(self, keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource):
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_names, model_classes, model_resource)
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (model_names, model_classes, model_resource)

    def match(self, file_path="", state_dict=None):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        state_dict = load_state_dict(file_path) if not state_dict else state_dict
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        return keys_hash in self.keys_hash_dict

    def load(self, file_path="", state_dict=None, device="cuda", torch_dtype=torch.float16, **kwargs):
        state_dict = load_state_dict(file_path) if not state_dict else state_dict
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[keys_hash_with_shape]
            return load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)

        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            return load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)
        return [], []


class ModelDetectorFromSplitedSingleFile(ModelDetectorFromSingleFile):
    def match(self, file_path="", state_dict=None):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        state_dict = load_state_dict(file_path) if not state_dict else state_dict
        return any(super(ModelDetectorFromSplitedSingleFile, self).match(file_path, item) for item in split_state_dict_with_prefix(state_dict))

    def load(self, file_path="", state_dict=None, device="cuda", torch_dtype=torch.float16, **kwargs):
        state_dict = load_state_dict(file_path) if not state_dict else state_dict
        loaded_model_names, loaded_models = [], []
        for sub_state_dict in split_state_dict_with_prefix(state_dict):
            if super().match(file_path, sub_state_dict):
                names, models = super().load(file_path, sub_state_dict, device=device, torch_dtype=torch_dtype)
                loaded_model_names += names
                loaded_models += models
        return loaded_model_names, loaded_models


class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        model_id_list: List[str] = None,
        downloading_priority: List[str] = None,
        file_path_list: List[str] = None,
    ):
        if model_id_list:
            raise ValueError("Preset model downloads were removed from this Wan-only ModelManager. Pass local file paths via ModelConfig instead.")
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        self.model_detector = [
            ModelDetectorFromSingleFile(MODEL_LOADER_CONFIGS),
            ModelDetectorFromSplitedSingleFile(MODEL_LOADER_CONFIGS),
        ]
        self.load_models(file_path_list or [])

    def load_model_from_single_file(self, file_path="", state_dict=None, model_names=None, model_classes=None, model_resource=None):
        print(f"Loading models from file: {file_path}")
        state_dict = load_state_dict(file_path) if not state_dict else state_dict
        names, models = load_model_from_single_file(
            state_dict,
            model_names or [],
            model_classes or [],
            model_resource,
            self.torch_dtype,
            self.device,
        )
        for model_name, model in zip(names, models):
            self.model.append(model)
            self.model_path.append(file_path)
            self.model_name.append(model_name)
        print(f"    The following models are loaded: {names}.")

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"Loading models from: {file_path}")
        device = self.device if device is None else device
        torch_dtype = self.torch_dtype if torch_dtype is None else torch_dtype
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            state_dict = None

        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                names, models = model_detector.load(file_path, state_dict, device=device, torch_dtype=torch_dtype)
                for loaded_name, model in zip(names, models):
                    if model_names is not None and loaded_name not in model_names:
                        continue
                    self.model.append(model)
                    self.model_path.append(file_path)
                    self.model_name.append(loaded_name)
                print(f"    The following models are loaded: {names}.")
                break
        else:
            print("    We cannot detect the model type. No models are loaded.")

    def load_models(self, file_path_list, model_names=None, device=None, torch_dtype=None):
        for file_path in file_path_list:
            self.load_model(file_path, model_names=model_names, device=device, torch_dtype=torch_dtype)

    def fetch_model(self, model_name, file_path=None, require_model_path=False, index=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, loaded_model_name in zip(self.model, self.model_path, self.model_name):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == loaded_model_name:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        elif index is None:
            model = fetched_models[0]
            path = fetched_model_paths[0]
            print(f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {path}.")
        elif isinstance(index, int):
            model = fetched_models[:index]
            path = fetched_model_paths[:index]
            print(f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {path}.")
        else:
            model = fetched_models
            path = fetched_model_paths
            print(f"More than one {model_name} models are loaded in model manager: {fetched_model_paths}. Using {model_name} from {path}.")
        return (model, path) if require_model_path else model

    def to(self, device):
        for model in self.model:
            model.to(device)

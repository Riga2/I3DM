def load_state_dict(*args, **kwargs):
    from .models.utils import load_state_dict as _load_state_dict

    return _load_state_dict(*args, **kwargs)


def save_video(*args, **kwargs):
    from .data.video import save_video as _save_video

    return _save_video(*args, **kwargs)


def save_frames(*args, **kwargs):
    from .data.video import save_frames as _save_frames

    return _save_frames(*args, **kwargs)


def save_video_with_audio(*args, **kwargs):
    from .data.video import save_video_with_audio as _save_video_with_audio

    return _save_video_with_audio(*args, **kwargs)


def merge_video_audio(*args, **kwargs):
    from .data.video import merge_video_audio as _merge_video_audio

    return _merge_video_audio(*args, **kwargs)

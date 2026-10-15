import torch
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_preprocess = None

def get_model():
    global _model, _preprocess
    if _model is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="laion2b_s32b_b82k"
        )
        model = model.to(device)
        model.eval()
        _model = model
        _preprocess = preprocess
    return _model, _preprocess
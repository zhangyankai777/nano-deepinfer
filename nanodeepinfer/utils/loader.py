import glob
import os

import torch
from safetensors import safe_open


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """DeepSeek FP8 权重为 128x128 分块量化，按 scale 反量化回浮点。"""
    out = weight.float()
    block = 128
    s = scale.float()
    out = out.view(s.size(0), block, s.size(1), block) * s[:, None, :, None]
    return out.view(weight.shape)


def load_weights(model: torch.nn.Module, path: str) -> None:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    else:
        from huggingface_hub import snapshot_download

        local = snapshot_download(path, allow_patterns=["*.safetensors"])
        files = sorted(glob.glob(os.path.join(local, "*.safetensors")))
    assert files, f"no safetensors found in {path}"
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded = set()
    for file in files:
        with safe_open(file, framework="pt") as f:
            keys = set(f.keys())
            for name in keys:
                if name.endswith("weight_scale_inv") or name not in params and name not in buffers:
                    continue
                tensor = f.get_tensor(name)
                if tensor.dtype == torch.float8_e4m3fn and f"{name}_scale_inv" in keys:
                    tensor = _dequant_fp8(tensor, f.get_tensor(f"{name}_scale_inv"))
                target = params.get(name, buffers.get(name))
                assert target is not None and target.shape == tensor.shape, f"shape mismatch: {name}"
                with torch.no_grad():
                    target.copy_(tensor.to(target.dtype))
                loaded.add(name)
    missing = [n for n in params if n not in loaded and "e_score_correction_bias" not in n]
    if model.config.tie_word_embeddings:
        missing = [n for n in missing if n != "lm_head.weight"]
    assert not missing, f"missing weights: {missing[:8]}"

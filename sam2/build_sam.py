# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

import sam2

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    # If the user has "sam2/sam2" in their path, they are likey importing the repo itself
    # as "sam2" rather than importing the "sam2" python package (i.e. "sam2/sam2" directory).
    # This typically happens because the user is running Python from the parent directory
    # that contains the sam2 repo they cloned.
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let sam2_base handle this
        ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def build_sam2_moe(
    config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    """
    Build SAM2 model with Mixture of Prompt Experts (MoPE).

    This function builds a SAM2 model with MoE-LoRA adapters in the
    Memory Attention projection layers. It can optionally load base
    weights from a pre-trained checkpoint (the LoRA adapters will be
    randomly initialized).

    Args:
        config_file: Path to MoE configuration file
        ckpt_path: Path to pre-trained SAM2 checkpoint (optional)
        device: Device to load model on
        mode: 'eval' or 'train'
        hydra_overrides_extra: Additional Hydra config overrides
        apply_postprocessing: Apply post-processing settings
        **kwargs: Additional arguments

    Returns:
        SAM2 model with MoE-LoRA adapters
    """
    model = build_sam2(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
        mode=mode,
        hydra_overrides_extra=hydra_overrides_extra,
        apply_postprocessing=apply_postprocessing,
        **kwargs,
    )

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only MoE adapters (LoRA and gating networks)
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in ['lora', 'gate', 'experts']):
            param.requires_grad = True

    return model


def build_sam2_video_predictor_moe(
    config_file="configs/sam2.1/sam2.1_hiera_b+_moe.yaml",
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    """
    Build SAM2 video predictor with Mixture of Prompt Experts (MoPE).

    This function builds a SAM2 video predictor with MoE-LoRA adapters
    in the Memory Attention projection layers.

    Args:
        config_file: Path to MoE configuration file
        ckpt_path: Path to pre-trained SAM2 checkpoint (optional)
        device: Device to load model on
        mode: 'eval' or 'train'
        hydra_overrides_extra: Additional Hydra config overrides
        apply_postprocessing: Apply post-processing settings
        vos_optimized: Use VOS-optimized predictor
        **kwargs: Additional arguments

    Returns:
        SAM2VideoPredictor with MoE-LoRA adapters
    """
    model = build_sam2_video_predictor(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device=device,
        mode=mode,
        hydra_overrides_extra=hydra_overrides_extra,
        apply_postprocessing=apply_postprocessing,
        vos_optimized=vos_optimized,
        **kwargs,
    )

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only MoE adapters (LoRA and gating networks)
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in ['lora', 'gate', 'experts']):
            param.requires_grad = True

    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]

        # Map checkpoint keys to MoE model keys (if MoE model is used)
        # For MoE models, projection layers are wrapped: proj -> proj.base_layer
        model_keys = set(model.state_dict().keys())

        # Check if this is a MoE model
        is_moe_model = any('base_layer' in k for k in model_keys)

        if is_moe_model:
            # Remap checkpoint keys for MoE models
            sd_remapped = {}
            for key, value in sd.items():
                # If key is a projection layer in memory attention, remap to base_layer
                if any(pattern in key for pattern in [
                    '.q_proj.weight', '.q_proj.bias',
                    '.k_proj.weight', '.k_proj.bias',
                    '.v_proj.weight', '.v_proj.bias',
                    '.out_proj.weight', '.out_proj.bias'
                ]) and 'memory_attention' in key:
                    # Remap: memory_attention.layers.X.Y.proj -> memory_attention.layers.X.Y.proj.base_layer
                    new_key = key.replace('.weight', '.base_layer.weight').replace('.bias', '.base_layer.bias')
                    sd_remapped[new_key] = value
                else:
                    sd_remapped[key] = value
            sd = sd_remapped

        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)
        if missing_keys:
            # Filter out expected missing keys for MoE models (LoRA adapters)
            moe_keys = [k for k in missing_keys if 'lora' in k or 'gate' in k or 'experts' in k]
            non_moe_keys = [k for k in missing_keys if k not in moe_keys]

            if moe_keys:
                logging.info(f"MoE adapter keys not loaded (expected): {len(moe_keys)} keys")
            if non_moe_keys:
                logging.warning(f"Missing non-MoE keys: {non_moe_keys}")
                # Only raise error for non-MoE missing keys if they exist
                if non_moe_keys:
                    raise RuntimeError(f"Missing keys: {non_moe_keys}")
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

"""ComfyUI nodes for Idol: a single image to a 3D Gaussian splat (https://github.com/speridlabs/idol-sdk)."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .idol_mm import load_to_gpu, make_patcher, module_of, unload_patchers

try:
    from comfy_api.latest import Types
    File3D = Types.File3D
except (ImportError, AttributeError):
    File3D = None

log = logging.getLogger("ComfyUI-Idol")

GiB = 1024 ** 3
# Working-memory hints (on top of the weights) handed to ComfyUI so it can evict other models first.
ACT_BIREFNET = int(2.5 * GiB)
ACT_ENCODERS = int(3.0 * GiB)
ACT_DENOISER = int(2.0 * GiB)
ACT_DENOISER_BATCH_CFG = int(3.0 * GiB)

HF_REPO = "speridlabs/idol"
HF_CHOICE = "speridlabs/idol (auto-download)"
AUTO_DIRNAME = "speridlabs_idol"

# Default image encoders, from repos that are not gated, stored under models/idol/encoders.
# timm's DINOv3 ViT-H+ is bit-identical to facebook/dinov3-vith16plus-pretrain-lvd1689m after the key renaming
# in _dinov3_from_timm (Meta's q/v biases and mask token are all zero; timm leaves them out).
# FLUX.2 klein ships the FLUX.2-dev VAE in bf16, the dtype Idol runs it in.
DINOV3_TIMM_REPO = "timm/vit_huge_plus_patch16_dinov3.lvd1689m"
FLUX2_VAE_REPO = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_VAE_FILES = ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
DINOV3_CONFIG = {
    "architectures": ["DINOv3ViTModel"], "model_type": "dinov3_vit", "attention_dropout": 0.0,
    "drop_path_rate": 0.0, "hidden_act": "silu", "hidden_size": 1280, "image_size": 224,
    "initializer_range": 0.02, "intermediate_size": 5120, "key_bias": False, "layer_norm_eps": 1e-05,
    "layerscale_value": 1.0, "mlp_bias": True, "num_attention_heads": 20, "num_channels": 3,
    "num_hidden_layers": 32, "num_register_tokens": 4, "patch_size": 16, "pos_embed_jitter": None,
    "pos_embed_rescale": 2.0, "pos_embed_shift": None, "proj_bias": True, "query_bias": True,
    "rope_theta": 100.0, "torch_dtype": "float32", "use_gated_mlp": True, "value_bias": True,
}

# Quality presets of Idol Image to 3D Gaussians. web/idol_presets.js reads this table from the node definition.
PRESETS = {
    "lowest": {"steps": 15, "num_gaussians": 65536},
    "low": {"steps": 25, "num_gaussians": 131072},
    "medium": {"steps": 50, "num_gaussians": 262144},
    "high": {"steps": 75, "num_gaussians": 524288},
    "ultra": {"steps": 100, "num_gaussians": 1048576},
}

IDOL_DIR = os.path.join(folder_paths.models_dir, "idol")
ENCODERS_DIR = os.path.join(IDOL_DIR, "encoders")
os.makedirs(IDOL_DIR, exist_ok=True)
folder_paths.add_model_folder_path("idol", IDOL_DIR)


# --------------------------------------------------------------------------------------------------------------
# idol-sdk import (lazy, so a missing install gives a readable error instead of a failed custom-node import)
# --------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _idol() -> SimpleNamespace:
    try:
        from tensordict import TensorDict
        from idol.camera import decode_camera
        from idol.conditioning import EncoderAccessError, ImageEncoder, require_encoder_access
        from idol.models import (BiRefNet, GaussianFixedlenDecoder, LatentSeqMMFlowModel, layers,
                                 OctreeProbabilityFixedlenDecoder)
        from idol.pipeline import WEIGHT_FILES
        from idol.preprocess import CROP_MARGIN, preprocess_image
    except ImportError as error:
        raise ImportError(
            "ComfyUI-Idol needs the idol-sdk package and its dependencies. Install them into ComfyUI's Python:\n"
            "  pip install -r custom_nodes/ComfyUI-Idol/requirements.txt\n"
            f"(original error: {error})") from error
    return SimpleNamespace(
        TensorDict=TensorDict, decode_camera=decode_camera, EncoderAccessError=EncoderAccessError,
        ImageEncoder=ImageEncoder, require_encoder_access=require_encoder_access, BiRefNet=BiRefNet,
        GaussianFixedlenDecoder=GaussianFixedlenDecoder, LatentSeqMMFlowModel=LatentSeqMMFlowModel,
        OctreeProbabilityFixedlenDecoder=OctreeProbabilityFixedlenDecoder, layers=layers,
        WEIGHT_FILES=WEIGHT_FILES, CROP_MARGIN=CROP_MARGIN, preprocess_image=preprocess_image)


# --------------------------------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------------------------------
def _local_weight_dirs() -> list[str]:
    names = set()
    for base in folder_paths.get_folder_paths("idol"):
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if name != AUTO_DIRNAME and os.path.isfile(os.path.join(base, name, "config.json")):
                names.add(name)
    return sorted(names)


def _resolve_source(value: str, base: str | None = None):
    """'' -> None. A folder (absolute, or relative to `base`, models/idol or models/) -> its path.
    Anything else is passed through as a Hugging Face repo id."""
    value = (value or "").strip()
    if not value:
        return None
    candidates = [os.path.join(base, value)] if base else []
    candidates += [value, os.path.join(IDOL_DIR, value), os.path.join(folder_paths.models_dir, value)]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return value


def _download_custom_repo(repo_id: str) -> str:
    """A repo made with tools/convert_idol.py (Idol weights + encoders in one folder)."""
    from huggingface_hub import snapshot_download
    target = os.path.join(IDOL_DIR, repo_id.strip().replace("/", "_"))
    try:
        snapshot_download(repo_id.strip(), local_dir=target)
    except Exception as error:
        if not os.path.isfile(os.path.join(target, "config.json")):
            raise
        log.warning("Could not reach %s (%s). Using the local copy.", repo_id, error)
    return target


def _dinov3_from_timm(state: dict) -> dict:
    """timm vit_huge_plus_patch16_dinov3 state dict -> transformers DINOv3ViTModel state dict."""
    hidden = state["cls_token"].shape[-1]
    out = {
        "embeddings.cls_token": state["cls_token"],
        "embeddings.register_tokens": state["reg_token"],
        "embeddings.mask_token": torch.zeros(1, 1, hidden),
        "embeddings.patch_embeddings.weight": state["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.bias": state["patch_embed.proj.bias"],
        "norm.weight": state["norm.weight"],
        "norm.bias": state["norm.bias"],
    }
    renames = [("attn.proj", "attention.o_proj"), ("mlp.fc1_g", "mlp.gate_proj"), ("mlp.fc1_x", "mlp.up_proj"),
               ("mlp.fc2", "mlp.down_proj"), ("norm1", "norm1"), ("norm2", "norm2")]
    blocks = 1 + max(int(key.split(".")[1]) for key in state if key.startswith("blocks."))
    for i in range(blocks):
        src, dst = f"blocks.{i}.", f"layer.{i}."
        q, k, v = state[src + "attn.qkv.weight"].chunk(3)
        out[dst + "attention.q_proj.weight"] = q.contiguous()
        out[dst + "attention.k_proj.weight"] = k.contiguous()
        out[dst + "attention.v_proj.weight"] = v.contiguous()
        out[dst + "attention.q_proj.bias"] = torch.zeros(hidden)
        out[dst + "attention.v_proj.bias"] = torch.zeros(hidden)
        for old, new in renames:
            for param in ("weight", "bias"):
                out[f"{dst}{new}.{param}"] = state[f"{src}{old}.{param}"]
        out[dst + "layer_scale1.lambda1"] = state[src + "gamma_1"]
        out[dst + "layer_scale2.lambda1"] = state[src + "gamma_2"]
    return out


def _ensure_dinov3() -> str:
    """models/idol/encoders/dinov3-vith16plus, downloaded from timm (not gated) and converted on first use."""
    target = os.path.join(ENCODERS_DIR, "dinov3-vith16plus")
    weights = os.path.join(target, "model.safetensors")
    if os.path.isfile(weights) and os.path.isfile(os.path.join(target, "config.json")):
        return target
    import shutil
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file

    staging = os.path.join(IDOL_DIR, ".download")
    log.info("Downloading DINOv3 ViT-H+ (3.4 GB) from %s to %s", DINOV3_TIMM_REPO, target)
    source = hf_hub_download(DINOV3_TIMM_REPO, "model.safetensors", local_dir=staging)
    license_file = hf_hub_download(DINOV3_TIMM_REPO, "LICENSE.md", local_dir=staging)
    state = _dinov3_from_timm(load_file(source))
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "config.json"), "w") as f:
        json.dump(DINOV3_CONFIG, f, indent=2)
    shutil.copyfile(license_file, os.path.join(target, "LICENSE.md"))
    save_file(state, weights + ".tmp", metadata={"format": "pt"})
    os.replace(weights + ".tmp", weights)
    del state
    shutil.rmtree(staging, ignore_errors=True)
    return target


def _ensure_flux2_vae() -> str:
    """models/idol/encoders/flux2-vae (with a vae/ subfolder), downloaded from FLUX.2 klein (not gated)."""
    from huggingface_hub import hf_hub_download
    target = os.path.join(ENCODERS_DIR, "flux2-vae")
    for name in FLUX2_VAE_FILES:
        if not os.path.isfile(os.path.join(target, name)):
            log.info("Downloading the FLUX.2 VAE from %s to %s", FLUX2_VAE_REPO, target)
            hf_hub_download(FLUX2_VAE_REPO, name, local_dir=target)
    return target


def _encoder_source(user_value: str, config_value: str, weights_dir: str):
    """The user's folder or repo id; else a local folder named in config.json (a converted repo); else None (auto)."""
    chosen = _resolve_source(user_value, weights_dir)
    if chosen:
        return chosen
    from_config = _resolve_source(config_value, weights_dir)
    return from_config if from_config and os.path.isdir(from_config) else None


def _prepare_weights(idol, choice: str, dinov3_source: str, flux2_vae_source: str, custom_repo: str = ""):
    """Find (and if needed download) the Idol weights and the image encoders. A gated encoder repo is only used
    when the user names one, and its access is checked *before* any big download."""
    need_download = False
    if (custom_repo or "").strip():
        weights_dir = _download_custom_repo(custom_repo)
    elif choice == HF_CHOICE:
        from huggingface_hub import hf_hub_download
        weights_dir = os.path.join(IDOL_DIR, AUTO_DIRNAME)
        os.makedirs(weights_dir, exist_ok=True)
        hf_hub_download(HF_REPO, "config.json", local_dir=weights_dir)
        need_download = not all(os.path.isfile(os.path.join(weights_dir, f)) for f in idol.WEIGHT_FILES)
    else:
        weights_dir = next((os.path.join(base, choice) for base in folder_paths.get_folder_paths("idol")
                            if os.path.isfile(os.path.join(base, choice, "config.json"))), None)
        if weights_dir is None:
            raise FileNotFoundError(f"Idol weights folder '{choice}' not found under {IDOL_DIR}")

    with open(os.path.join(weights_dir, "config.json")) as f:
        config = json.load(f)
    encoders = config["image_encoders"]
    if (flux2_vae_source or "").strip() == FLUX2_VAE_REPO:  # older workflows typed the repo of the automatic VAE
        flux2_vae_source = ""
    dinov3_repo = _encoder_source(dinov3_source, encoders["dinov3"], weights_dir)
    flux2_repo = _encoder_source(flux2_vae_source, encoders["flux2_vae"], weights_dir)

    remote = [repo for repo in (dinov3_repo, flux2_repo) if repo and not os.path.isdir(repo)]
    try:
        idol.require_encoder_access(remote)
    except idol.EncoderAccessError as error:
        raise RuntimeError(str(error)) from error

    if need_download:
        from huggingface_hub import snapshot_download
        log.info("Downloading Idol weights to %s", weights_dir)
        snapshot_download(HF_REPO, allow_patterns=idol.WEIGHT_FILES, local_dir=weights_dir)
    return weights_dir, config, dinov3_repo or _ensure_dinov3(), flux2_repo or _ensure_flux2_vae()


class IdolBundle:
    """Everything the generator needs. Each model is a ComfyUI-managed ModelPatcher (see idol_mm.py), so VRAM is
    shared with the rest of the workflow and the weights are released when ComfyUI drops this object from its
    node cache."""

    def __init__(self, config, patchers, encoder, load_device, offload_device):
        self.config = config
        self.patchers = patchers
        self.encoder = encoder
        self.load_device = load_device
        self.offload_device = offload_device

    def all_patchers(self):
        return list(self.patchers.values())


# --------------------------------------------------------------------------------------------------------------
# node 1: loader
# --------------------------------------------------------------------------------------------------------------
class IdolModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weights": ([HF_CHOICE] + _local_weight_dirs(),),
                "dinov3_dtype": (["bf16", "fp32"], {"default": "bf16",
                                                   "tooltip": "fp32 runs DINOv3 in full precision (+1.7 GB)."}),
            },
            "optional": {
                "custom_hf_repo": ("STRING", {"default": "", "tooltip":
                                              "Hugging Face repo made with tools/convert_idol.py, for example "
                                              "your-name/idol-comfyui. Overrides 'weights'. A private repo needs "
                                              "'hf auth login' or HF_TOKEN."}),
                "dinov3_source": ("STRING", {"default": "", "tooltip":
                                             "Empty = automatic: DINOv3 ViT-H+ in models/idol/encoders, downloaded "
                                             "once from timm (no Hugging Face login needed, same weights as "
                                             "Meta's gated repo). Or a local folder, or a HF repo id (HF cache)."}),
                "flux2_vae_source": ("STRING", {"default": "", "tooltip":
                                                "Empty = automatic: the FLUX.2 VAE in models/idol/encoders, "
                                                "downloaded once from FLUX.2 klein (no login needed, same VAE as "
                                                "FLUX.2-dev). Or a local folder containing 'vae/', or a HF repo id "
                                                "(HF cache)."}),
            },
        }

    RETURN_TYPES = ("IDOL_MODEL",)
    RETURN_NAMES = ("idol_model",)
    FUNCTION = "load"
    CATEGORY = "Idol"

    def load(self, weights, dinov3_dtype, custom_hf_repo="", dinov3_source="", flux2_vae_source=""):
        from safetensors.torch import load_file

        idol = _idol()
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        weights_dir, config, dinov3_repo, flux2_repo = _prepare_weights(
            idol, weights, dinov3_source, flux2_vae_source, custom_hf_repo)

        def build(cls, args, name):
            model = cls(**args)
            state = load_file(os.path.join(weights_dir, f"{name}.safetensors"))
            model.load_state_dict(state, strict=True)
            del state
            return model.eval().requires_grad_(False)

        # Everything is built on the CPU; ComfyUI moves it to the GPU on demand (see idol_mm.py).
        denoiser = build(idol.LatentSeqMMFlowModel, config["denoiser"]["args"], "denoiser")
        decoder = build(idol.OctreeProbabilityFixedlenDecoder, config["decoder"]["args"], "decoder")
        decoder_gs = build(idol.GaussianFixedlenDecoder, config["decoder_gs"]["args"], "decoder_gs")
        birefnet = build(idol.BiRefNet, {}, "birefnet")
        if load_device.type != "cpu":
            birefnet = birefnet.half()
        encoder = idol.ImageEncoder(dinov3_repo, flux2_repo, dinov3_dtype=dinov3_dtype, device="cpu")

        patchers = {name: make_patcher(module, load_device, offload_device) for name, module in (
            ("denoiser", denoiser), ("decoder", decoder), ("decoder_gs", decoder_gs), ("birefnet", birefnet),
            ("dinov3", encoder.dinov3), ("flux2_vae", encoder.flux2_vae))}
        bundle = IdolBundle(config, patchers, encoder, load_device, offload_device)
        log.info("Idol loaded: %.2f GB of weights registered with ComfyUI's model manager",
                 sum(p.model_size() for p in bundle.all_patchers()) / GiB)
        return (bundle,)


# --------------------------------------------------------------------------------------------------------------
# node 2: generate
# --------------------------------------------------------------------------------------------------------------
class _LazyBackgroundRemover:
    """BiRefNet is only loaded to the GPU if the image really has no alpha (idol's preprocess_image asks for it)."""

    def __init__(self, patcher):
        self.patcher = patcher

    def remove_background(self, image: Image.Image) -> Image.Image:
        mm.throw_exception_if_processing_interrupted()
        load_to_gpu([self.patcher], ACT_BIREFNET)
        return module_of(self.patcher).remove_background(image)


@contextmanager
def _seeded(seed: int):
    """Seed torch for one generation and restore the global RNG afterwards (ComfyUI's other nodes are unaffected)."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
        yield


@contextmanager
def _sampling_settings(idol, enabled: bool):
    """cuDNN attention + TF32 matmuls for the denoiser only (what idol's optimize=True does, minus torch.compile)."""
    if not enabled:
        yield
        return
    backend, tf32 = idol.layers.ATTN_BACKEND, torch.backends.cuda.matmul.allow_tf32
    idol.layers.ATTN_BACKEND = "sdpa_cudnn"
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        idol.layers.ATTN_BACKEND = backend
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _fast_path_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


@torch.no_grad()
def _sample_flow(denoiser, noise, cond, steps, rescale_t, cfg_strength, batch_cfg, pbar):
    """idol.sampler.sample_flow, with ComfyUI's progress bar and cancel button."""
    neg_cond = cond.apply(torch.zeros_like)
    device = noise["latent"].device

    def velocity(x, t, c):
        stamp = torch.full((x.shape[0],), 1000.0 * t, device=device, dtype=torch.float32)
        return denoiser(x, stamp, c)

    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    sample = noise
    for i in range(steps):
        mm.throw_exception_if_processing_interrupted()
        t, t_prev = float(t_seq[i]), float(t_seq[i + 1])
        if cfg_strength > 0 and batch_cfg:
            n = sample.shape[0]
            both = velocity(torch.cat([sample, sample]), t, torch.cat([cond, neg_cond]))
            v = (1 + cfg_strength) * both[:n] - cfg_strength * both[n:]
        else:
            v = velocity(sample, t, cond)
            if cfg_strength > 0:
                v = (1 + cfg_strength) * v - cfg_strength * velocity(sample, t, neg_cond)
        sample = sample - v * (t - t_prev)
        pbar.update(1)
    return sample


def _to_pil(image: torch.Tensor, mask) -> Image.Image:
    """image [H, W, 3] in 0..1, optional mask [H, W] (1 = object) -> RGB or RGBA PIL image."""
    rgb = Image.fromarray((image[..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())
    if mask is None:
        return rgb
    if mask.shape != image.shape[:2]:
        mask = F.interpolate(mask[None, None].float(), size=tuple(image.shape[:2]), mode="bilinear")[0, 0]
    rgb.putalpha(Image.fromarray((mask.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()))
    return rgb


class IdolGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "idol_model": ("IDOL_MODEL",),
                "image": ("IMAGE",),
                "preset": (["custom"] + list(PRESETS), {"default": "medium", "idol_presets": PRESETS, "tooltip":
                                                        "Sets steps and num_gaussians. " + "; ".join(
                                                            f"{name}: {p['steps']} steps, {p['num_gaussians']:,} "
                                                            f"gaussians" for name, p in PRESETS.items()) +
                                                        ". medium is the reference setting. custom uses the "
                                                        "values below. Editing them switches to custom."}),
                "mode": (["synthetic", "real"], {"default": "synthetic", "tooltip":
                                                 "synthetic: renders / generated images (crop 1.2). "
                                                 "real: photographs (crop 1.7)."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "rescale_t": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip":
                                        "Time shift: higher spends more steps at high noise."}),
                "num_gaussians": ("INT", {"default": 262144, "min": 32768, "max": 1048576, "step": 32}),
                "crop_margin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip":
                                          "0 = use the mode's default. Crop side = object bounding box x this."}),
                "fast_sampling": ("BOOLEAN", {"default": True, "tooltip":
                                              "Batched CFG + cuDNN attention + TF32 (CUDA, Ampere or newer). "
                                              "Faster, but needs more VRAM while sampling."}),
                "offload_after_run": ("BOOLEAN", {"default": False, "tooltip":
                                                  "Immediately move all Idol models back to CPU RAM when done. "
                                                  "Off = ComfyUI keeps them on the GPU until the space is needed."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Object mask. Skips background removal. An empty mask is ignored "
                                             "(BiRefNet runs instead)."}),
                "invert_mask": ("BOOLEAN", {"default": True, "tooltip":
                                            "On: the mask is 1 = transparent, as Load Image outputs it. "
                                            "Off: the mask is 1 = object, as most background-removal nodes "
                                            "output it."}),
            },
        }

    RETURN_TYPES = ("IDOL_GAUSSIAN", "IMAGE", "STRING")
    RETURN_NAMES = ("gaussian", "preprocessed_image", "camera_json")
    OUTPUT_IS_LIST = (True, False, True)
    FUNCTION = "generate"
    CATEGORY = "Idol"

    def generate(self, idol_model, image, mode, seed, steps, cfg_strength, rescale_t, num_gaussians, crop_margin,
                 fast_sampling, offload_after_run, preset="custom", mask=None, invert_mask=True):
        if preset in PRESETS:
            steps, num_gaussians = PRESETS[preset]["steps"], PRESETS[preset]["num_gaussians"]
        idol = _idol()
        bundle: IdolBundle = idol_model
        batch = image.shape[0]
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[None]
            if invert_mask:
                mask = 1.0 - mask
            # Load Image outputs an all-zero 64x64 mask for images without alpha
            if float(mask.max()) <= 0.0:
                log.warning("Idol: the mask is empty, so it is ignored and BiRefNet removes the background.")
                mask = None
        pbar = comfy.utils.ProgressBar(batch * steps)

        gaussians, previews, cameras = [], [], []
        try:
            for index in range(batch):
                frame_mask = None if mask is None else mask[min(index, mask.shape[0] - 1)]
                started = time.time()
                gaussian, preview, camera = self._run_one(
                    idol, bundle, _to_pil(image[index], frame_mask), mode, seed, steps, cfg_strength, rescale_t,
                    num_gaussians, crop_margin if crop_margin > 0 else None, fast_sampling, pbar)
                log.info("Idol image %d/%d done in %.1fs", index + 1, batch, time.time() - started)
                gaussians.append(gaussian)
                previews.append(preview)
                cameras.append(json.dumps(camera, indent=2))
        finally:
            if offload_after_run:
                unload_patchers(bundle.all_patchers())

        return (gaussians, torch.stack(previews).to(mm.intermediate_device()), cameras)

    @torch.no_grad()
    def _run_one(self, idol, bundle, pil, mode, seed, steps, cfg_strength, rescale_t, num_gaussians, crop_margin,
                 fast_sampling, pbar):
        patchers, device, config = bundle.patchers, bundle.load_device, bundle.config

        # 1. background removal + crop (BiRefNet is loaded only when the image has no alpha)
        margin = crop_margin if crop_margin is not None else idol.CROP_MARGIN[mode]
        prepared = idol.preprocess_image(pil, background_remover=_LazyBackgroundRemover(patchers["birefnet"]),
                                         image_size=config["image_size"], crop_margin=margin)
        pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None]

        # Same RNG order as idol's pipeline (seed -> VAE posterior sample -> noise -> sampling -> decoding),
        # so a given seed reproduces the reference implementation.
        with _seeded(seed):
            # 2. image encoders (DINOv3 + FLUX.2 VAE)
            load_to_gpu([patchers["dinov3"], patchers["flux2_vae"]], ACT_ENCODERS)
            bundle.encoder.device = device
            cond = bundle.encoder(pixels)

            # 3. flow-matching sampler (the denoiser)
            denoiser = module_of(patchers["denoiser"])
            noise = idol.TensorDict({
                "latent": torch.randn(1, denoiser.q_token_length, denoiser.in_channels, device=device),
                "camera": torch.randn(1, 1, denoiser.cam_channels, device=device),
            }, batch_size=1, device=device)
            fast = fast_sampling and _fast_path_supported(device)
            load_to_gpu([patchers["denoiser"]], ACT_DENOISER_BATCH_CFG if fast else ACT_DENOISER)
            with _sampling_settings(idol, fast):
                sample = _sample_flow(denoiser, noise, cond, steps, rescale_t, cfg_strength, fast, pbar)
            del cond, noise

            # 4. octree + gaussian decoders
            extra = int(GiB + (num_gaussians / 262144) * 0.5 * GiB)
            load_to_gpu([patchers["decoder"], patchers["decoder_gs"]], extra)
            decoder, decoder_gs = module_of(patchers["decoder"]), module_of(patchers["decoder_gs"])
            per_anchor = decoder_gs.rep_config["num_gaussians"]
            anchors = idol.OctreeProbabilityFixedlenDecoder.sample(
                decoder, sample["latent"], num_points=max(1, int(num_gaussians) // per_anchor),
                level=config["max_voxel_level"])
            gaussian = decoder_gs.to_representation(anchors, decoder_gs(anchors, sample["latent"]))[0]
            camera = idol.decode_camera(sample["camera"][0, 0])
            del anchors, sample

        # Park the result in ordinary memory so it does not pin VRAM while the rest of the workflow runs.
        store = mm.intermediate_device()
        for name in ("xyz", "features_dc", "scaling", "rotation", "opacity"):
            setattr(gaussian, name, getattr(gaussian, name).detach().to(store))
        preview = torch.from_numpy(np.array(prepared)).float().div(255.0)
        return gaussian, preview, camera


# --------------------------------------------------------------------------------------------------------------
# node 3: save PLY
# --------------------------------------------------------------------------------------------------------------
class IdolSavePLY:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gaussian": ("IDOL_GAUSSIAN",),
            "filename_prefix": ("STRING", {"default": "idol/gaussian"}),
        }}

    # model_3d plugs into the core splat nodes (Preview Splat, Save Gaussian Splat, Get Splat).
    RETURN_TYPES = ("STRING",) + (("FILE_3D_PLY",) if File3D is not None else ())
    RETURN_NAMES = ("ply_path",) + (("model_3d",) if File3D is not None else ())
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Idol"

    def save(self, gaussian, filename_prefix):
        # Exact path: prefix ending in ".ply" writes that file under output/ with no counter.
        # Example: "VGH_3d/chair.ply" -> output/VGH_3d/chair.ply
        prefix = filename_prefix.replace("\\", "/")
        if prefix.lower().endswith(".ply"):
            output_dir = folder_paths.get_output_directory()
            path = os.path.abspath(os.path.join(output_dir, prefix))
            if not folder_paths.is_within_directory(output_dir, path):
                raise ValueError(f"Invalid filename_prefix: {filename_prefix!r}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            gaussian.save_ply(path)
            rel = os.path.relpath(path, output_dir)
            result = (path,) + ((File3D(path),) if File3D is not None else ())
            return {"ui": {"text": [rel]}, "result": result}

        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 0, 0)
        filename = f"{name}_{counter:05}_.ply"
        path = os.path.join(folder, filename)
        gaussian.save_ply(path)
        result = (path,) + ((File3D(path),) if File3D is not None else ())
        return {"ui": {"text": [os.path.join(subfolder, filename)]}, "result": result}


NODE_CLASS_MAPPINGS = {
    "IdolModelLoader": IdolModelLoader,
    "IdolGenerate": IdolGenerate,
    "IdolSavePLY": IdolSavePLY,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdolModelLoader": "Idol Model Loader",
    "IdolGenerate": "Idol Image to 3D Gaussians",
    "IdolSavePLY": "Idol Save PLY",
}

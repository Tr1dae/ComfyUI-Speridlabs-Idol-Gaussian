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

IDOL_DIR = os.path.join(folder_paths.models_dir, "idol")
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


def _prepare_weights(idol, choice: str, dinov3_source: str, flux2_vae_source: str, custom_repo: str = ""):
    """Find (and if needed download) the Idol weights; check access to the gated encoders *before* any big download."""
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
    dinov3_repo = _resolve_source(dinov3_source, weights_dir) or _resolve_source(encoders["dinov3"], weights_dir)
    flux2_repo = (_resolve_source(flux2_vae_source, weights_dir)
                  or _resolve_source(encoders["flux2_vae"], weights_dir))

    remote = [repo for repo in (dinov3_repo, flux2_repo) if not os.path.isdir(repo)]
    try:
        idol.require_encoder_access(remote)
    except idol.EncoderAccessError as error:
        raise RuntimeError(str(error)) from error

    if need_download:
        from huggingface_hub import snapshot_download
        log.info("Downloading Idol weights to %s", weights_dir)
        snapshot_download(HF_REPO, allow_patterns=idol.WEIGHT_FILES, local_dir=weights_dir)
    return weights_dir, config, dinov3_repo, flux2_repo


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
                                             "Optional local folder (or HF repo id) for DINOv3 ViT-H+. "
                                             "Empty = the gated facebook repo from config.json."}),
                "flux2_vae_source": ("STRING", {"default": "", "tooltip":
                                                "Optional local folder (containing a 'vae' subfolder) or HF repo "
                                                "id for FLUX.2. Empty = black-forest-labs/FLUX.2-dev."}),
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
                "mask": ("MASK", {"tooltip": "Object mask (1 = object). Skips background removal."}),
            },
        }

    RETURN_TYPES = ("IDOL_GAUSSIAN", "IMAGE", "STRING")
    RETURN_NAMES = ("gaussian", "preprocessed_image", "camera_json")
    OUTPUT_IS_LIST = (True, False, True)
    FUNCTION = "generate"
    CATEGORY = "Idol"

    def generate(self, idol_model, image, mode, seed, steps, cfg_strength, rescale_t, num_gaussians, crop_margin,
                 fast_sampling, offload_after_run, mask=None):
        idol = _idol()
        bundle: IdolBundle = idol_model
        batch = image.shape[0]
        if mask is not None and mask.dim() == 2:
            mask = mask[None]
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Idol"

    def save(self, gaussian, filename_prefix):
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), 0, 0)
        filename = f"{name}_{counter:05}_.ply"
        path = os.path.join(folder, filename)
        gaussian.save_ply(path)
        return {"ui": {"text": [os.path.join(subfolder, filename)]}, "result": (path,)}


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

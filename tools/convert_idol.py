#!/usr/bin/env python3
"""Repack Idol and its two image encoders into one folder for ComfyUI-Idol, and optionally upload it.

What this does (the weights do NOT need a conversion for ComfyUI-Idol to work):
  * Idol weights: every tensor is stored in the dtype the model uses at run time. BiRefNet is stored in fp16,
    which is what the node uses on a GPU.
  * DINOv3 and the FLUX.2 VAE: saved in bf16 (what the node casts them to when it loads them) as local
    Hugging Face folders. The download is about half the size, and the node needs no access to the gated repos.
  * config.json points to the local encoder folders.
  * Everything is checked after saving. A Hugging Face upload is optional.

Examples:
  python convert_idol.py --out converted
  python convert_idol.py --out converted --upload your-name/idol-comfyui
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

DEFAULT_REPO = "speridlabs/idol"
WEIGHT_FILES = ["config.json", "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors",
                "birefnet.safetensors"]
# file name -> (class in idol.models, key of its constructor arguments in config.json, or None)
COMPONENTS = {
    "denoiser": ("LatentSeqMMFlowModel", "denoiser"),
    "decoder": ("OctreeProbabilityFixedlenDecoder", "decoder"),
    "decoder_gs": ("GaussianFixedlenDecoder", "decoder_gs"),
    "birefnet": ("BiRefNet", None),
}


def log(message: str) -> None:
    print(message, flush=True)


def megabytes(path: Path) -> float:
    return sum(f.stat().st_size for f in ([path] if path.is_file() else path.rglob("*")) if f.is_file()) / 1e6


def fetch_idol(source: str, work: Path, token) -> Path:
    if os.path.isdir(source):
        return Path(source)
    from huggingface_hub import snapshot_download
    log(f"Downloading {source} ...")
    return Path(snapshot_download(source, allow_patterns=WEIGHT_FILES, local_dir=work / "idol-original", token=token))


def convert_idol_weights(src: Path, dst: Path, config: dict, birefnet_dtype: str) -> None:
    import idol.models as idol_models

    for name, (class_name, key) in COMPONENTS.items():
        model = getattr(idol_models, class_name)(**(config[key]["args"] if key else {}))
        runtime_dtype = {k: v.dtype for k, v in model.state_dict().items()}
        state = load_file(str(src / f"{name}.safetensors"))
        model.load_state_dict(state, strict=True)  # stops here if the file does not match the model

        converted = {}
        for k, tensor in state.items():
            want = runtime_dtype[k]
            if name == "birefnet" and birefnet_dtype == "fp16" and tensor.is_floating_point():
                want = torch.float16
            converted[k] = (tensor.to(want) if tensor.dtype != want else tensor).contiguous()
        out_path = dst / f"{name}.safetensors"
        save_file(converted, str(out_path), metadata={"format": "pt"})

        model.load_state_dict(load_file(str(out_path)), strict=True)  # the saved file must load too
        log(f"  {name}: {megabytes(src / f'{name}.safetensors'):.0f} MB -> {megabytes(out_path):.0f} MB")
        del model, state, converted


def convert_encoders(dinov3_repo: str, vae_repo: str, dst: Path, token, verify: bool) -> None:
    from diffusers.models import AutoencoderKLFlux2
    from transformers import AutoModel

    extra = {"token": token} if token else {}
    generator = torch.Generator().manual_seed(0)
    dino_input = torch.rand(1, 3, 224, 224, generator=generator).to(torch.bfloat16)
    vae_input = torch.rand(1, 3, 64, 64, generator=generator).to(torch.bfloat16) * 2 - 1

    @torch.no_grad()
    def dino_out(model):
        return model(pixel_values=dino_input).last_hidden_state.float()

    @torch.no_grad()
    def vae_out(model):
        return model.encode(vae_input).latent_dist.mode().float()  # mode(): no random sampling, so it is comparable

    log(f"Loading {dinov3_repo} as bf16 ...")
    dino = AutoModel.from_pretrained(dinov3_repo, torch_dtype=torch.bfloat16, **extra).eval()
    reference = dino_out(dino) if verify else None
    dino.save_pretrained(dst / "dinov3", safe_serialization=True)
    del dino
    if verify:
        again = AutoModel.from_pretrained(dst / "dinov3", torch_dtype=torch.bfloat16).eval()
        assert torch.equal(reference, dino_out(again)), "DINOv3 output changed after saving"
    log(f"  dinov3: {megabytes(dst / 'dinov3'):.0f} MB")

    log(f"Loading {vae_repo} (vae) as bf16 ...")
    vae = AutoencoderKLFlux2.from_pretrained(vae_repo, subfolder="vae", torch_dtype=torch.bfloat16, **extra).eval()
    reference = vae_out(vae) if verify else None
    vae.save_pretrained(dst / "flux2_vae" / "vae", safe_serialization=True)
    del vae
    if verify:
        again = AutoencoderKLFlux2.from_pretrained(dst / "flux2_vae", subfolder="vae",
                                                   torch_dtype=torch.bfloat16).eval()
        assert torch.equal(reference, vae_out(again)), "FLUX.2 VAE output changed after saving"
    log(f"  flux2_vae: {megabytes(dst / 'flux2_vae'):.0f} MB")


def write_model_card(dst: Path, args, config: dict, encoders_converted: bool) -> None:
    encoder_line = ("The folders `dinov3/` and `flux2_vae/` hold the image encoders in bf16."
                    if encoders_converted else "The image encoders are not included. `config.json` names them.")
    (dst / "README.md").write_text(f"""---
license: other
base_model:
- {args.idol}
{'- ' + config['converted']['from']['dinov3'] if encoders_converted else ''}
{'- ' + config['converted']['from']['flux2_vae'] if encoders_converted else ''}
---

# Idol, repacked for ComfyUI-Idol

This repository holds the weights of [Idol]({'https://github.com/speridlabs/idol-sdk'}) in one folder that
the ComfyUI-Idol loader can use. The weights are the same. Only the storage type changed.

{encoder_line}

Licenses: Idol, DINOv3, and FLUX.2-dev each have their own license. The terms of each source model apply to this
repository. Read them before you share it.

Use it: in the ComfyUI-Idol loader, type this repository ID in `custom_hf_repo`.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--idol", default=DEFAULT_REPO, help="Idol Hugging Face repo or local folder")
    parser.add_argument("--out", default="idol-comfyui", help="output folder")
    parser.add_argument("--dinov3", default=None, help="DINOv3 repo or folder (default: the one in config.json)")
    parser.add_argument("--flux2-vae", default=None, help="FLUX.2 repo or folder with vae/ (default: config.json)")
    parser.add_argument("--skip-encoders", action="store_true", help="convert only the Idol weights")
    parser.add_argument("--birefnet-dtype", choices=["fp16", "keep"], default="fp16")
    parser.add_argument("--no-verify", action="store_true", help="skip the checks after saving the encoders")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token (or use HF_TOKEN)")
    parser.add_argument("--upload", metavar="REPO_ID", help="upload the result, for example your-name/idol-comfyui")
    parser.add_argument("--public", action="store_true", help="make the uploaded repository public")
    parser.add_argument("--confirm-licenses", action="store_true",
                        help="required with --public: you read the Idol, DINOv3 and FLUX.2-dev licenses")
    args = parser.parse_args()

    if args.public and not args.confirm_licenses:
        parser.error("--public needs --confirm-licenses. DINOv3 and FLUX.2-dev are gated and have their own licenses.")

    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    src = fetch_idol(args.idol, dst.parent / ".idol-download", args.token)
    config = json.loads((src / "config.json").read_text())

    log("Converting the Idol weights ...")
    convert_idol_weights(src, dst, config, args.birefnet_dtype)

    encoders = config["image_encoders"]
    if not args.skip_encoders:
        dinov3_repo = args.dinov3 or encoders["dinov3"]
        vae_repo = args.flux2_vae or encoders["flux2_vae"]
        convert_encoders(dinov3_repo, vae_repo, dst, args.token, verify=not args.no_verify)
        config["converted"] = {"from": {"idol": args.idol, "dinov3": dinov3_repo, "flux2_vae": vae_repo},
                               "encoder_dtype": "bfloat16", "birefnet_dtype": args.birefnet_dtype}
        config["image_encoders"] = {"dinov3": "dinov3", "flux2_vae": "flux2_vae"}  # relative to this folder
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    write_model_card(dst, args, config, not args.skip_encoders)
    log(f"Done. Output folder: {dst.resolve()} ({megabytes(dst) / 1000:.2f} GB)")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=args.token)
        api.create_repo(args.upload, private=not args.public, exist_ok=True, repo_type="model")
        log(f"Uploading to {args.upload} ({'public' if args.public else 'private'}) ...")
        api.upload_folder(folder_path=str(dst), repo_id=args.upload, repo_type="model",
                          commit_message="Idol weights repacked for ComfyUI-Idol")
        log(f"Uploaded: https://huggingface.co/{args.upload}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

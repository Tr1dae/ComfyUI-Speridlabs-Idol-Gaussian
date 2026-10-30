"""Glue between plain torch modules and ComfyUI's native model management.

ComfyUI's ``ModelPatcher`` + ``comfy.model_management.load_models_gpu`` is what lets core models share VRAM,
get evicted when something else needs the space, and get freed when the node cache drops them.  Plain
``nn.Module`` weights (like Idol's) can use it too, with two adjustments handled here:

1. ``ModelPatcher``'s partial ("lowvram") loading only works for modules built with ``comfy.ops``; anything else
   would be left half on the CPU and crash on the first forward.  ``WholeModelPatcher`` therefore always loads and
   unloads the whole module (all-or-nothing), which is still fully managed: ComfyUI will evict it whole when it
   needs the VRAM.
2. ``ModelPatcher`` assigns ``model.device``.  Some modules (BiRefNet) expose ``device`` as a read-only property,
   so every module is wrapped in ``ModuleHolder``, which owns a plain ``device`` attribute.
"""
from __future__ import annotations

import logging
from typing import Iterable

import torch

import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher

log = logging.getLogger("ComfyUI-Idol")


class ModuleHolder(torch.nn.Module):
    """Thin wrapper so ModelPatcher can freely set ``.device`` and the other bookkeeping attributes."""

    def __init__(self, module: torch.nn.Module, device: torch.device):
        super().__init__()
        self.module = module
        self.device = device


class WholeModelPatcher(ModelPatcher):
    """ModelPatcher for ordinary nn.Modules: always fully on the load device, or fully offloaded."""

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        # lowvram_model_memory is ignored on purpose: partial loading is unsafe for plain modules.
        return super().load(device_to or self.load_device, 0, force_patch_weights, True)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        if self.loaded_size() > 0:
            return 0  # already fully resident
        self.load(device_to, full_load=True)
        return self.loaded_size()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        # All-or-nothing: moving the whole module back frees at least what ComfyUI asked for.
        freed = self.loaded_size()
        if freed <= 0:
            return 0
        self.unpatch_model(device_to, unpatch_weights=True)
        return freed


def make_patcher(module: torch.nn.Module, load_device: torch.device, offload_device: torch.device) -> WholeModelPatcher:
    """Register ``module`` with ComfyUI's model manager. The module ends up on ``offload_device``."""
    module = module.eval().requires_grad_(False).to(offload_device)
    return WholeModelPatcher(ModuleHolder(module, offload_device), load_device, offload_device)


def module_of(patcher: ModelPatcher) -> torch.nn.Module:
    return patcher.model.module


def load_to_gpu(patchers: Iterable[ModelPatcher], activation_bytes: int = 0) -> None:
    """Ask ComfyUI to make room (evicting other models if needed) and load these models.

    ``activation_bytes`` is the working memory the upcoming computation needs on top of the weights."""
    mm.load_models_gpu(list(patchers), memory_required=int(activation_bytes))


def unload_patchers(patchers: Iterable[ModelPatcher]) -> None:
    """Offload these models right now, through ComfyUI's own bookkeeping."""
    unload = getattr(mm, "unload_model_and_clones", None) or getattr(mm, "unload_model_clones", None)
    for patcher in patchers:
        try:
            if unload is not None:
                unload(patcher)
            else:
                patcher.detach()
        except Exception as error:  # never let cleanup break a finished generation
            log.warning("Could not unload %s: %s", type(module_of(patcher)).__name__, error)
    mm.soft_empty_cache()

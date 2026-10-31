# ComfyUI-Idol

ComfyUI-Idol adds three nodes to ComfyUI. The nodes turn one image into a 3D Gaussian splat (a 3D object stored as many small colored blobs). The nodes use [Idol](https://github.com/speridlabs/idol-sdk). The result is a standard 3DGS `.ply` file.

## Install

1. Open a terminal in `ComfyUI/custom_nodes`.
2. Clone this repository into a folder named `ComfyUI-Idol`.
3. Install the packages in the Python environment of ComfyUI.
4. Restart ComfyUI.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Tr1dae/ComfyUI-Speridlabs-Idol-Gaussian.git ComfyUI-Idol
pip install -r ComfyUI-Idol/requirements.txt
```

On the Windows portable build, the embedded Python cannot build `idol-sdk` in an isolated environment (`Cannot import 'hatchling.build'`). Install the build backend first and turn off build isolation:

```bash
python_embeded\python.exe -m pip install hatchling
python_embeded\python.exe -m pip install --no-build-isolation -r ComfyUI\custom_nodes\ComfyUI-Idol\requirements.txt
```

## Weights

No Hugging Face account or login is needed. On the first run the loader downloads everything into `ComfyUI/models/idol/`:

| Folder | Content | Source | Size |
|---|---|---|---|
| `speridlabs_idol/` | Idol (denoiser, decoders, BiRefNet) | [speridlabs/idol](https://huggingface.co/speridlabs/idol) | 3 GB |
| `encoders/dinov3-vith16plus/` | DINOv3 ViT-H+ image encoder | [timm/vit_huge_plus_patch16_dinov3.lvd1689m](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3.lvd1689m) | 3.4 GB |
| `encoders/flux2-vae/` | FLUX.2 VAE | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 0.2 GB |

The Idol SDK names two gated repositories for the encoders: [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) and [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev). The sources above are not gated and give the same results:

- timm's DINOv3 holds the same tensors as Meta's under other names. The loader renames them once, after the download. All 615 tensors are bit-identical to Meta's. DINOv3 is under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/), and a copy of it is saved next to the weights.
- FLUX.2 klein holds the FLUX.2-dev VAE in bf16, the dtype that Idol runs the VAE in.

Nothing goes to the Hugging Face cache, unless you type a Hugging Face repository ID in `dinov3_source` or `flux2_vae_source`.

### Manual download

If the automatic download fails, create the folders below and place each file at the listed path. Leave `dinov3_source` and `flux2_vae_source` empty so the loader uses them.

**Idol** → `ComfyUI/models/idol/speridlabs_idol/`

| File | URL |
|---|---|
| `config.json` | https://huggingface.co/speridlabs/idol/resolve/main/config.json |
| `denoiser.safetensors` | https://huggingface.co/speridlabs/idol/resolve/main/denoiser.safetensors |
| `decoder.safetensors` | https://huggingface.co/speridlabs/idol/resolve/main/decoder.safetensors |
| `decoder_gs.safetensors` | https://huggingface.co/speridlabs/idol/resolve/main/decoder_gs.safetensors |
| `birefnet.safetensors` | https://huggingface.co/speridlabs/idol/resolve/main/birefnet.safetensors |

**FLUX.2 VAE** → `ComfyUI/models/idol/encoders/flux2-vae/vae/`

| File | URL |
|---|---|
| `config.json` | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/config.json |
| `diffusion_pytorch_model.safetensors` | https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main/vae/diffusion_pytorch_model.safetensors |

**DINOv3 ViT-H+** → `ComfyUI/models/idol/encoders/dinov3-vith16plus/`

Drop-in files must be the transformers layout (what Meta ships). Automatic download uses the ungated timm copy and renames it once; a raw timm `model.safetensors` is **not** drop-in. For a manual place-as-is install, use Meta's files (gated — accept the terms and log in if the links return 401):

| File | URL |
|---|---|
| `config.json` | https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/main/config.json |
| `model.safetensors` | https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/main/model.safetensors |
| `LICENSE.md` (optional) | https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/main/LICENSE.md |

To use a different local Idol folder, put the five Idol files in any subdirectory of `models/idol/` and select that folder in the loader.

The optional inputs `dinov3_source` and `flux2_vae_source` are empty by default (automatic). They also accept a local folder or a Hugging Face repository ID. For FLUX.2, use the folder that contains `vae/`. A gated repository ID needs access and `hf auth login`. The loader checks access before it downloads large files.

The optional input `custom_hf_repo` accepts a repository that you made with the converter (see the next section). It overrides the `weights` list.

## Nodes

- Idol Model Loader: loads all weights to the CPU. It registers them with the model manager of ComfyUI (the part that moves models between the GPU and the CPU).
- Idol Image to 3D Gaussians: takes an `IMAGE` and an optional `MASK`. It returns `IDOL_GAUSSIAN`, the preprocessed 1024-pixel image, and the predicted camera as JSON. It processes a batch one image at a time. The mask skips background removal.
- Idol Save PLY: writes a `.ply` file to the output folder. It returns the absolute path of the file and `model_3d`, a 3D file that the core splat nodes accept.

## Use

A ready-made workflow is in `example_workflows/`. ComfyUI lists it under the templates of this node pack.

1. Connect `Load Image` to `Idol Image to 3D Gaussians`.
2. Connect `Idol Model Loader` to `Idol Image to 3D Gaussians`.
3. Connect `Idol Image to 3D Gaussians` to `Idol Save PLY`.
4. Set `mode` to `real` for photographs. Set `mode` to `synthetic` for renders and generated images.
5. Pick a quality `preset`. It sets `steps` and `num_gaussians`. If you edit those values, the preset switches to `custom`.

   | preset | steps | num_gaussians |
   |---|---|---|
   | lowest | 15 | 65,536 |
   | low | 25 | 131,072 |
   | medium (reference) | 50 | 262,144 |
   | high | 75 | 524,288 |
   | ultra | 100 | 1,048,576 |

6. To skip background removal, connect a mask. `Load Image` drops the alpha channel from `IMAGE` and outputs it inverted as `MASK` (1 = transparent). `invert_mask` is on by default for that, so for a PNG with transparency just connect the `MASK` of `Load Image`. For a mask with 1 = object (most background-removal nodes), turn `invert_mask` off.
7. To view the result, connect `model_3d` of `Idol Save PLY` to the core `Preview Splat` node. `Get Splat` turns it into a `SPLAT` for the other core splat nodes (render, transform, splat to mesh).

## Convert the weights (optional)

The nodes work with the original weights. A conversion is not necessary, and a conversion does not enable partial offloading. ComfyUI can offload only layers that use its own layer classes (`comfy.ops`). The Idol code uses plain PyTorch layers, and a weight file cannot change that.

The converter makes a smaller copy of the weights that you control:

- It stores each Idol weight in the data type that the model uses at run time. It stores BiRefNet in fp16 (16-bit floating point).
- It stores DINOv3 and the FLUX.2 VAE in bf16 (a 16-bit format with a wide range). The download is about half the size.
- It writes one folder. The `config.json` in the folder points to the local encoders, so the loader needs no access to the gated repositories.
- It checks the saved files. It can then upload the folder to your Hugging Face repository.

To run the converter on your machine:

```bash
python ComfyUI-Idol/tools/convert_idol.py --out converted --upload your-name/idol-comfyui
```

To run the converter in Google Colab, open `notebooks/Idol_to_ComfyUI.ipynb`. The notebook needs no GPU.

To use the result, type `your-name/idol-comfyui` in the `custom_hf_repo` input of the loader. If the repository is private, run `hf auth login` first.

DINOv3 and FLUX.2-dev are gated and have their own licenses. The upload is private by default. The `--public` option also needs `--confirm-licenses`. Read both licenses before you make the repository public.

## Memory

ComfyUI-Idol uses the native model management of ComfyUI. Each of the six networks has a `ModelPatcher`. The networks are the denoiser, the octree decoder, the Gaussian decoder, BiRefNet, DINOv3, and the FLUX.2 VAE. Each load uses `comfy.model_management.load_models_gpu`.

- Each stage loads only the models that it needs. BiRefNet loads only when the image has no alpha channel. The order is encoders, denoiser, decoders.
- When VRAM (the memory of the GPU) is low, ComfyUI moves other models to the CPU. This includes models of other nodes in the same workflow.
- Models stay on the GPU between runs if space is free. ComfyUI frees them when it removes the loader from its node cache.
- The `offload_after_run` option moves all Idol models to CPU memory at once.
- The seed changes only this node. The node restores the global random state after each run.
- The node shows the progress bar and obeys the cancel button.
- The node moves the finished Gaussians off the GPU.

ComfyUI cannot move parts of a plain `torch.nn.Module` to the CPU. Each model therefore moves as a whole. See `idol_mm.py`. The largest single model is DINOv3 at about 1.7 GB, so this limit does not matter on small GPUs.

## Differences from the SDK

- This package does not use `torch.compile`. It changes module instances, and this does not work with managed offloading.
- The `fast_sampling` option keeps three other speed-ups: batched CFG, cuDNN attention, and TF32.
- Batched CFG (classifier-free guidance) runs both guidance passes in one batch. TF32 is a faster math mode of Nvidia GPUs.
- These speed-ups need a CUDA GPU with compute capability 8.0 or higher.
- If VRAM is low, turn off `fast_sampling`. Batched CFG doubles the activation memory of the denoiser.
- A seed gives the same random order as the reference pipeline.

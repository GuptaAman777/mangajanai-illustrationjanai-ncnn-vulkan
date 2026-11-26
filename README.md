# MangaJaNai & IllustrationJaNai — Vulkan-Capable ncnn Models

This repository provides **22 image super-resolution models** (MangaJaNai & IllustrationJaNai) converted from FP16 ONNX into **Vulkan-capable `ncnn` models** (`.ncnn.param` + `.ncnn.bin`), verified to run with **zero CPU layer fallbacks** (`support_vulkan = True` for all layers) on both FP16-capable and FP32-only Vulkan GPUs.

> **Direct Model Downloads (NCNN Only)**: All **22 `.ncnn.param`** and **22 `.ncnn.bin`** files (**44 files total**, NCNN models only) are uploaded individually without any `.zip` compression in [**GitHub Releases**](https://github.com/GuptaAman777/mangajanai-illustrationjanai-ncnn-vulkan/releases).
>
> **Use with `psd-converter`**: You can use these Vulkan NCNN models directly inside [**ImageTool (`psd-converter`)**](https://github.com/GuptaAman777/psd-converter) for local Vulkan-accelerated AI super-resolution.

---

## Directory Structure

```text
.
├── scripts/          # Conversion and Vulkan/ONNX verification scripts
└── ncnn/             # Converted ncnn graph files (.ncnn.param; .ncnn.bin + .ncnn.param in Releases)
    ├── ESRGAN_2x/    # 7 models
    ├── ESRGAN_4x/    # 7 models
    ├── FDAT/         # 6 models
    └── SPAN/         # 2 models
```

---

## Model Specifications & Input Requirements

All converted models share the same `ncnn` blob interface:
- **Input blob**: `in0` — `3`-channel RGB tensor `(3, H, W)` in float32 normalized to `[0.0, 1.0]`
- **Output blob**: `out0` — `3`-channel RGB tensor `(3, H * scale, W * scale)` in float32

| Family | Scale | Input `H, W` Alignment | Converted Models (`ncnn/<family>/`) |
| :--- | :---: | :--- | :--- |
| **SPAN** | `2x` | Any `H, W` | `2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16.ncnn.{param,bin}`<br>`2x_IllustrationJaNai_V3detail_SPAN_S_40k_fp16.ncnn.{param,bin}` |
| **ESRGAN_4x** | `4x` | Any `H, W` | `4x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.ncnn.{param,bin}`<br>`4x_MangaJaNai_1300p_V1_ESRGAN_75k_fp16.ncnn.{param,bin}`<br>`4x_MangaJaNai_1400p_V1_ESRGAN_105k_fp16.ncnn.{param,bin}`<br>`4x_MangaJaNai_1500p_V1_ESRGAN_105k_fp16.ncnn.{param,bin}`<br>`4x_MangaJaNai_1600p_V1_ESRGAN_70k_fp16.ncnn.{param,bin}`<br>`4x_MangaJaNai_1920p_V1_ESRGAN_105k_fp16.ncnn.{param,bin}`<br>`4x_MangaJaNai_2048p_V1_ESRGAN_70k_fp16.ncnn.{param,bin}` |
| **ESRGAN_2x** | `2x` | `H % 2 == 0` and `W % 2 == 0` | `2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.ncnn.{param,bin}`<br>`2x_MangaJaNai_1300p_V1_ESRGAN_75k_fp16.ncnn.{param,bin}`<br>`2x_MangaJaNai_1400p_V1_ESRGAN_70k_fp16.ncnn.{param,bin}`<br>`2x_MangaJaNai_1500p_V1_ESRGAN_90k_fp16.ncnn.{param,bin}`<br>`2x_MangaJaNai_1600p_V1_ESRGAN_90k_fp16.ncnn.{param,bin}`<br>`2x_MangaJaNai_1920p_V1_ESRGAN_70k_fp16.ncnn.{param,bin}`<br>`2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16.ncnn.{param,bin}` |
| **FDAT (4x)** | `4x` | `H % 8 == 0` and `W % 8 == 0` | `4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.ncnn.{param,bin}`<br>`4x_IllustrationJaNai_V3detail_FDAT_M_40k_fp16.ncnn.{param,bin}`<br>`4x_IllustrationJaNai_V3denoise_FDAT_XL_32k_fp16.ncnn.{param,bin}`<br>`4x_IllustrationJaNai_V3detail_FDAT_XL_27k_fp16.ncnn.{param,bin}` |
| **FDAT (2x)** | `2x` | `H % 16 == 0` and `W % 16 == 0` | `2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.ncnn.{param,bin}`<br>`2x_IllustrationJaNai_V3detail_FDAT_M_unshuffle_40k_fp16.ncnn.{param,bin}` |

> **Note on alignment**: `ESRGAN_2x` and `FDAT` have the same divisibility requirements as the original ONNX models. For arbitrary image sizes, reflect-pad the input to the required multiple before running inference and crop the output back to `(H * scale, W * scale)`.

---

## Using with `psd-converter`

These converted Vulkan `ncnn` models are compatible with [**`GuptaAman777/psd-converter`**](https://github.com/GuptaAman777/psd-converter) (*ImageTool v7.0 — PSD & AI Image Suite*):
- Download the matching `.ncnn.param` and `.ncnn.bin` pair directly from [**GitHub Releases**](https://github.com/GuptaAman777/mangajanai-illustrationjanai-ncnn-vulkan/releases) (no archive extraction required).
- Point `psd-converter`'s local Vulkan NCNN upscaling workflow to the downloaded model files to upscale manga, illustrations, and extracted `.psd` layers on any Vulkan-capable GPU.

---

## Running Inference on Any Vulkan GPU

### 1. Python (`ncnn`)

Install `ncnn` (`1.0.20261020` or newer, built with Vulkan + `SDPA`/`LayerNorm`/`Reorg` support):

```bash
pip install ncnn numpy pillow
```

Example script (`upscale.py`) with automatic alignment padding and cross-GPU Vulkan setup:

```python
import math
import ncnn
import numpy as np
from PIL import Image


def upscale_image(
    img_path: str,
    out_path: str,
    param_path: str,
    bin_path: str,
    scale: int,
    align: int = 1,
    gpu_index: int = 0,
    use_fp16: bool = True,
):
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    h, w, _ = arr.shape

    # Reflect-pad to required alignment (1 for SPAN/ESRGAN_4x, 2 for ESRGAN_2x, 8 for FDAT 4x, 16 for FDAT 2x)
    pad_h = (math.ceil(h / align) * align) - h
    pad_w = (math.ceil(w / align) * align) - w
    if pad_h or pad_w:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    # Convert HWC -> NCHW (1, 3, H_pad, W_pad)
    chw = np.transpose(arr, (2, 0, 1))[None, ...].copy()

    net = ncnn.Net()
    net.opt.use_vulkan_compute = ncnn.get_gpu_count() > 0
    if net.opt.use_vulkan_compute:
        net.set_vulkan_device(gpu_index)
        # Set all three to False on older GPUs without 16-bit shader/storage support,
        # or True for faster inference on modern Vulkan GPUs.
        net.opt.use_fp16_packed = use_fp16
        net.opt.use_fp16_storage = use_fp16
        net.opt.use_fp16_arithmetic = use_fp16

    assert net.load_param(param_path) == 0
    assert net.load_model(bin_path) == 0

    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(chw, batch_index=0).clone())
        ret, mat_out = ex.extract("out0")
        assert ret == 0

    out = mat_out.numpy(batch_index=0)[0]  # (3, H_pad*scale, W_pad*scale)
    out = out[:, : h * scale, : w * scale]
    out = np.clip(np.transpose(out, (1, 2, 0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)


if __name__ == "__main__":
    # Example: FDAT 4x (align=8, scale=4)
    upscale_image(
        "input.png",
        "output.png",
        "ncnn/FDAT/4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.ncnn.param",
        "ncnn/FDAT/4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.ncnn.bin",
        scale=4,
        align=8,
    )
```

> **Important Python binding note**: Always pass `ncnn.Mat(chw, batch_index=0).clone()` (4D `(1, 3, H, W)` with `batch_index=0`) and extract with `mat_out.numpy(batch_index=0)` so row/channel stride (`cstep`) matches `ncnn`'s memory layout.

---

### 2. C++ (`ncnn`)

```cpp
#include "net.h"
#include "gpu.h"

ncnn::Net net;
net.opt.use_vulkan_compute = (ncnn::get_gpu_count() > 0);
net.set_vulkan_device(0);

// Optional: set to false for maximum precision or GPUs without FP16 support
net.opt.use_fp16_packed = true;
net.opt.use_fp16_storage = true;
net.opt.use_fp16_arithmetic = true;

net.load_param("ncnn/FDAT/4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.ncnn.param");
net.load_model("ncnn/FDAT/4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.ncnn.bin");

// in_mat: ncnn::Mat(w, h, 3) normalized to [0, 1] in RGB order
ncnn::Extractor ex = net.create_extractor();
ex.input("in0", in_mat);

ncnn::Mat out_mat;
ex.extract("out0", out_mat);
```

---

## Cross-GPU Vulkan Compatibility Notes

1. **100% Vulkan-native operators**:
   - Every computational layer across all 22 models reports `support_vulkan = True` in `ncnn`:
     - `SPAN`: `BinaryOp`, `Concat`, `Convolution`, `PixelShuffle`, `Sigmoid`, `Split`, `Swish`
     - `ESRGAN_4x`: `BinaryOp`, `Concat`, `Convolution`, `Interp`, `Split`
     - `ESRGAN_2x`: `BinaryOp`, `Concat`, `Convolution`, `Interp`, `Reorg`, `Split`
     - `FDAT`: `BinaryOp`, `Clip`, `Convolution`, `ConvolutionDepthWise`, `Deconvolution`, `GELU`, `Gemm`, `InnerProduct`, `LayerNorm`, `MemoryData`, `Permute`, `Pooling`, `Reduction`, `Reorg`, `Reshape`, `SDPA`, `Slice`, `Split`, `UnaryOp`
2. **FP16 & Non-FP16 GPU compatibility**:
   - Weights in `.ncnn.bin` are stored in FP16 (`pnnx fp16=1`, matching the source FP16 ONNX precision and halving file size).
   - On GPUs that do **not** support `VK_KHR_16bit_storage` or FP16 arithmetic (or when `use_fp16_* = False`), `ncnn` automatically upcasts weights to FP32 on load and executes FP32 Vulkan shaders.
   - For `FDAT` Cosine Channel Attention, L2 normalization scales `Q`/`K` by `0.0625` before squaring to prevent FP16 overflow (`> 65504`) on large image tiles.
3. **Tiled inference for large images**:
   - For high-resolution images (especially with `FDAT_XL` or `ESRGAN`), process in tiles (e.g., `128x128` or `256x256` with `16`–`32` px overlap, aligned to `2`, `8`, or `16` depending on the model) to stay within GPU VRAM limits.

---

## How Conversion & Fixes Were Implemented

If you need to re-convert or customize the models, the environment used was:
- `ncnn==1.0.20261020`
- `pnnx==20261020`
- `onnx==1.23.2`
- `onnxruntime==1.31.1`
- `torch==2.15.0`

Summary of the conversion pipeline per family:
1. **`SPAN` & `ESRGAN_4x`**:
   - Cast ONNX initializers, `Cast` attributes, and graph I/O types from `FLOAT16` to `FLOAT` (FP32) so `pnnx`'s ONNX frontend matches `Conv` nodes cleanly.
   - Run `pnnx <fp32.onnx> fp16=1 input=<32x32.npy> input2=<48x48.npy>` (using small `.npy` trace inputs avoids `pnnx`'s default `512x512` CPU ONNX Runtime OOM).
2. **`ESRGAN_2x`**:
   - Remove the 6D `PixelUnshuffle(2)` ONNX subgraph (`nodes[:13]`) and redundant trailing `Slice` (`nodes[-2:]`), cast the RRDB body to FP32 ONNX, convert with `pnnx`, and prepend `Reorg pixelunshuffle_0 1 1 in0 unshuffled 0=2 1=0` in `.ncnn.param`.
3. **`FDAT` (`M`, `XL`, `2x_unshuffle`)**:
   - Reconstruct the network in PyTorch using `<= 4D` tensors and Vulkan-native ops (`1x1 Conv2d` instead of `MatMul`, `nn.PixelUnshuffle(8)`/`nn.PixelShuffle(8)` + `F.scaled_dot_product_attention` for Spatial Window Attention, `F.scaled_dot_product_attention` + scale-safe L2 norm for Cosine Channel Attention), load weights directly from `.onnx`, verify against ONNX Runtime, trace with `torch.jit.trace`, and convert with `pnnx ... fp16=1 inputshape=[1,3,16,16] inputshape2=[1,3,32,48]`.

### Conversion & Verification Scripts

Reproducible scripts are included in [`scripts/`](scripts/):
- [`scripts/convert_span_esr4.py`](scripts/convert_span_esr4.py) — Converts `SPAN` and `ESRGAN_4x` ONNX models
- [`scripts/convert_esr2.py`](scripts/convert_esr2.py) — Converts `ESRGAN_2x` ONNX models with Vulkan `Reorg` unshuffle
- [`scripts/convert_fdat.py`](scripts/convert_fdat.py) — Reconstructs and converts `FDAT` (`M`, `XL`, `2x_unshuffle`) models
- [`scripts/verify_all.py`](scripts/verify_all.py) — Verifies all 22 models against ONNX Runtime in CPU, Vulkan FP32, and Vulkan FP16 modes

---

## Sources & Credits

All original pretrained weights, architectures, and conversion/inference tools belong to their respective authors and projects:

- **MangaJaNai & IllustrationJaNai Models**: [`the-database/MangaJaNai`](https://github.com/the-database/MangaJaNai) (by `the-database`, with companion apps [`MangaJaNaiGUI`](https://github.com/the-database/MangaJaNaiGUI) and [`MangaJaNaiCLI`](https://github.com/the-database/MangaJaNaiCLI))
- **FDAT Architecture** (*Fast Dual Attention Transformer*): [`stinkybread/FDAT`](https://github.com/stinkybread/FDAT) and [`stinkybread/FDAT-Models`](https://github.com/stinkybread/FDAT-Models)
- **SPAN Architecture** (*Swift Parameter-free Attention Network*): [`hongyuanyu/SPAN`](https://github.com/hongyuanyu/SPAN)
- **ESRGAN Architecture** (*Enhanced Super-Resolution Generative Adversarial Networks*): [`xinntao/ESRGAN`](https://github.com/xinntao/ESRGAN)
- **Training & Model Loader Ecosystem**: [`the-database/traiNNer-redux`](https://github.com/the-database/traiNNer-redux) and [`chaiNNer-org/spandrel`](https://github.com/chaiNNer-org/spandrel)
- **Conversion & Inference Frameworks**: [`Tencent/ncnn`](https://github.com/Tencent/ncnn) and [`pnnx/pnnx`](https://github.com/pnnx/pnnx)
- **Compatible Desktop Tool**: [`GuptaAman777/psd-converter`](https://github.com/GuptaAman777/psd-converter) (*ImageTool v7.0 — PSD & AI Image Suite*)

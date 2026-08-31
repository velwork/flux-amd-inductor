# FLUX AMD Inductor Optimization

A reproducible ComfyUI setup for running FLUX Q5_K_S + baked LoRA on an AMD RX 9060 XT with TorchInductor.

## Results

Tested on:

- GPU: AMD Radeon RX 9060 XT 16 GB
- Architecture: `gfx1200`
- ROCm: `7.14`
- PyTorch: `2.12.0+rocm7.14.0`
- ComfyUI: `0.34.2`
- Attention: PyTorch attention
- Model: FLUX Q5_K_S GGUF
- Resolution: 768×768
- Steps: 20

Observed performance:

| Configuration | Result |
|---|---:|
| Runtime LoRA | ~5.3 s/it |
| LoRA-free + Inductor | ~1.86–1.89 s/it |
| Baked LoRA + Inductor | ~1.77–1.9 s/it |
| Final total generation | ~40–50 s / 20 steps |
| Recompilations | 0 in the validated fast-path tests |

TAESD was also tested for the VAE stage and reduced one validated total generation from 48.94 s to 39.90 s.

## What was happening

The main problem was not the AMD GPU itself.

With a runtime LoRA on quantized GGUF weights, the execution path produced TorchDynamo/Inductor recompilations. Diagnostics showed guard failures involving `comfy_cast_weights` and varying tensor layouts/shapes.

Removing the runtime LoRA restored the fast Inductor path.

## Solution

### 1. Bake the LoRA into the existing Q5 model

Instead of applying the LoRA during every forward pass:

```text
Q5 GGUF
  -> runtime LoRA patch
  -> forward
```

the LoRA was applied once to the existing Q5 weights and the result was re-quantized to Q5_K_S:

```text
Q5 GGUF
  -> dequantize
  -> apply LoRA once
  -> re-quantize to Q5_K_S
  -> baked GGUF
```

This removes the runtime LoRA patching path.

The original model and original LoRA are not modified.

### 2. Keep the normal Inductor path

The final workflow uses the normal TorchCompile/Inductor path.

CK attention is not used.

Runtime LoRA is not used because the LoRA effect is already baked into the model.

### 3. Make FLUX text conditioning token length dynamic

A second issue appeared when changing prompts.

The FLUX text-conditioning token dimension could cause a new Inductor graph for longer prompts, producing a very slow path (~23 s/it in one test).

Only the FLUX context/token dimension was made dynamic; the latent/model dimensions stayed static.

After this change, a 295-word stress prompt ran at about 2.53–2.55 s/it without recompilation.

This means longer prompts can cost more compute, but changing the prompt no longer triggers the old recompilation slowdown.

### 4. Background warmup

A custom warmup extension is used to prepare the target graph in the background after ComfyUI starts.

The warmup is designed to:

- run in the background
- stay out of the visible queue
- produce no output/history entry
- avoid interfering with a user generation
- run once per process when needed
- use the same baked Q5 + Inductor path as the main workflow

The goal is to pay the compile cost during startup instead of during the first user generation.

## Final workflow

The canonical workflow is:

`FLUX_Baked_Q5_Inductor_768.json`

Place it in:

```text
user/default/workflows/
```

The workflow is configured for:

- 768×768
- 20 steps
- CFG 3.5
- DPM++ 2M
- beta scheduler
- baked Q5_K_S FLUX model
- normal TorchCompile/Inductor
- no runtime LoRA

## Cache

The tested persistent cache locations were:

```text
C:\ComfyCache\torchinductor
C:\ComfyCache\triton
```

Relevant environment variables:

```text
TORCHINDUCTOR_CACHE_DIR=C:\ComfyCache\torchinductor
TORCHINDUCTOR_FX_GRAPH_CACHE=1
TRITON_CACHE_DIR=C:\ComfyCache\triton
```

Do not delete these caches unless you intentionally want to force recompilation.

## Files in this repository

```text
flux-amd-inductor/
├── README.md
├── workflows/
│   └── FLUX_Baked_Q5_Inductor_768.json
├── patches/
│   ├── torch_compile_dynamic_context.patch
│   ├── gguf_guard_fix.patch
│   └── rmsnorm_guard_fix.patch
├── warmup/
│   └── __init__.py
└── benchmarks/
    └── results.md
```

The repository scaffold intentionally does not include the large baked GGUF model.

## Reproduction notes

This project was validated on one specific hardware/software combination. Performance will vary across GPUs, ROCm versions, PyTorch versions, ComfyUI versions, prompts, resolutions, and samplers.

The reported ~1.9 s/it result should therefore be treated as a measured reference, not a universal guarantee.

## Backups

Before applying the source patches to another installation, make backups of the corresponding ComfyUI files.

The original working setup also retained backups of the modified source files.

## License / model notes

This repository contains optimization notes, workflow configuration, and source patches. Model files and LoRA files are not redistributed here.

Check the original licenses/terms for any model or LoRA you use.

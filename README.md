# FLUX on AMD: Q5_K_S + LoRA + TorchInductor Optimization

> **From ~5.3 s/it to ~1.9 s/it on an AMD Radeon RX 9060 XT — while keeping the LoRA effect.**

This repository documents a working ComfyUI setup and the source-level changes used to get **FLUX Q5_K_S + LoRA** running efficiently on an AMD RX 9060 XT with **ROCm + PyTorch + TorchInductor**.

The important part is not just the final benchmark. The goal was to identify **why the fast path was being lost**, then remove the causes one by one.

---

## TL;DR

On the test system:

```text
AMD Radeon RX 9060 XT 16 GB
ROCm 7.14
PyTorch 2.12.0+rocm7.14.0
ComfyUI 0.34.2
FLUX Q5_K_S GGUF
768 × 768
20 steps
PyTorch attention
TorchInductor
```

The measured path went approximately:

```text
Runtime LoRA                 ~5.3 s/it
           ↓
LoRA removed / baseline      ~1.86–1.89 s/it
           ↓
LoRA baked into Q5           ~1.77–1.9 s/it
           ↓
Dynamic FLUX context         stable across prompt changes
           ↓
TAESD VAE                    ~39.9 s total in one validated run
```

Representative final result:

```text
20 steps
~1.9 s/it
~40–50 s total
0 recompilations in validated fast-path tests
```

---

# What was the problem?

The initial setup used a **runtime LoRA on quantized GGUF weights**.

The result was much slower than the LoRA-free model, even though the underlying GPU/Inductor path was capable of much better performance.

TorchDynamo/Inductor diagnostics showed repeated recompilation/guard problems around the runtime LoRA path, including changing tensor layouts/shapes and `comfy_cast_weights`-related guards.

In other words:

```text
Q5 GGUF
   ↓
runtime LoRA patching
   ↓
TorchDynamo sees changing state/layouts
   ↓
recompilation
   ↓
performance collapses
```

Removing the runtime LoRA immediately brought the model back to roughly:

```text
~1.86–1.89 s/it
```

That strongly suggested that **runtime LoRA patching — not the AMD GPU itself — was the main bottleneck.**

---

# The main fix: bake the LoRA into Q5

Instead of applying the LoRA during every forward pass, the LoRA was applied **once** to a copy of the existing Q5 model.

Conceptually:

```text
BEFORE

Q5 GGUF
  ↓
runtime LoRA patch
  ↓
forward
```

became:

```text
AFTER

Q5 GGUF
  ↓
dequantize affected weights
  ↓
apply LoRA once
  ↓
re-quantize to Q5_K_S
  ↓
baked GGUF
  ↓
TorchInductor
```

The original model and original LoRA were kept unchanged.

The resulting baked model preserves the LoRA effect without requiring runtime LoRA weight patching during inference.

## Why not merge from FP8?

An FP8 → merge → Q5 route was considered, but a small numerical check showed more reconstruction error than opening the existing Q5 weights, applying the LoRA, and re-quantizing that result.

Because the goal was to keep the original Q5 representation as the starting point, the latter approach was chosen.

---

# Second fix: dynamic FLUX text-conditioning length

After the LoRA issue was removed, another problem appeared when changing prompts.

A short prompt could run around:

```text
~1.9 s/it
```

while a changed/longer prompt could fall into a much slower path around:

```text
~23 s/it
```

The cause was traced to the **FLUX text-conditioning context token dimension** being treated as static by the normal TorchCompile/Inductor path.

The fix was intentionally narrow:

- Make only the FLUX text-conditioning token axis dynamic.
- Keep latent/model dimensions static.
- Keep normal TorchCompile + Inductor.
- Do not reintroduce runtime LoRA.

The tested dynamic token range was:

```text
1–1024
```

After the change:

```text
Normal prompt          ~1.9 s/it
295-word stress prompt ~2.53–2.55 s/it
Recompilation          0 observed in the validation run
```

The longer prompt is still somewhat slower because it is doing more text-conditioning work. The important part is that it **does not fall back into the old recompilation-heavy ~23 s/it behavior.**

---

# Third optimization: background warmup

TorchInductor still has an unavoidable first-use compilation cost.

Instead of making the user wait during the first real image generation, this setup includes a small custom warmup extension.

The warmup:

- runs in the background after ComfyUI starts
- prepares the same target Baked Q5 + Inductor execution path
- does not create an output image
- does not create a visible queue entry
- does not write a history entry
- avoids interfering with an active user generation
- runs once per process when needed

The idea is simple:

```text
ComfyUI starts
      ↓
background warmup
      ↓
compile/cache preparation
      ↓
[Baked FLUX Warmup] Ready
      ↓
user presses Generate
      ↓
fast path
```

The warmup can take several minutes when a new compile is required. That time is deliberately paid **before** the first user generation instead of during it.

---

# Fourth optimization: TAESD

The sampling speed is mostly determined by the FLUX model itself, so changing the VAE does not magically turn:

```text
1.9 s/it → 1.0 s/it
```

Instead, the benefit is in the **post-sampling VAE decode path**.

The normal VAE produced one validated total runtime of:

```text
48.94 s
```

Switching to TAESD produced:

```text
39.90 s
```

for the same validated generation.

That is about a **9 second / 18% reduction in total wall-clock time** for that test.

TAESD is therefore treated here as an **optional total-runtime optimization**, not the core Inductor optimization.

---

# Benchmark history

The rough progression on the same setup was:

| Stage | Approx. result |
|---|---:|
| Initial / unoptimized | ~250–400 s |
| After early optimization | ~100–120 s |
| Runtime LoRA + optimized path | ~5.3 s/it |
| LoRA-free + Inductor | ~1.86–1.89 s/it |
| Baked LoRA + Inductor | ~1.77–1.9 s/it |
| Baked LoRA + TAESD | ~39.9 s total in one validated run |

Representative fast-path measurements:

| Configuration | Sampling | Total |
|---|---:|---:|
| Runtime LoRA | ~5.3 s/it | ~106–107 s |
| LoRA-free + Inductor | ~1.86–1.89 s/it | ~42–46 s |
| Baked LoRA + Inductor | ~1.77–1.9 s/it | ~42 s range |
| Baked LoRA + TAESD | ~1.90 s/it | **39.90 s** |

One especially clean baked-LoRA benchmark recorded:

```text
1.77 s/it
41.74 s total
0 recompilations
```

---

# Reproducibility

## Recommended baseline

Keep these conditions fixed when comparing results:

```text
Resolution:   768 × 768
Steps:        20
CFG:          3.5
Sampler:      DPM++ 2M
Scheduler:    beta
Attention:    PyTorch attention
Backend:      TorchInductor
Model:        FLUX Q5_K_S baked GGUF
```

Different GPUs, ROCm versions, PyTorch versions, ComfyUI versions, prompts, resolutions, samplers, and memory states can produce different numbers.

The reported results are therefore **measured references, not universal guarantees.**

---

# Repository contents

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

The large baked GGUF model is intentionally **not included** in this repository.

The original Q5 model and LoRA should also be retained separately.

---

# Cache

The validated setup used persistent compiler caches:

```text
C:\ComfyCache\torchinductor
C:\ComfyCache\triton
```

with:

```text
TORCHINDUCTOR_CACHE_DIR=C:\ComfyCache\torchinductor
TORCHINDUCTOR_FX_GRAPH_CACHE=1
TRITON_CACHE_DIR=C:\ComfyCache\triton
```

Avoid deleting these caches unless you intentionally want to force recompilation.

---

# Important details

### Runtime LoRA

Do **not** load the original LoRA at runtime when using the baked model.

The LoRA effect is already present in the baked GGUF.

### Trigger words

Baking a LoRA changes the **weights**, not the text prompt.

If the LoRA depends on a trigger token/phrase, that trigger can still be required in the prompt.

### CK attention

This setup uses PyTorch attention.

CK attention was not part of the final fast-path configuration.

### 1024 × 1024

The published workflow is validated at:

```text
768 × 768
```

A 1024 × 1024 run may require a separate Inductor graph/cache and will naturally require more compute.

---

# Why this repository exists

The useful part of this project was not discovering a magical switch.

The useful part was following the performance regression back to the actual execution behavior:

```text
slow runtime LoRA
      ↓
Dynamo/Inductor recompilations
      ↓
bake LoRA into Q5
      ↓
fast Inductor path returns
      ↓
prompt changes trigger another issue
      ↓
dynamic FLUX context
      ↓
stable fast path across prompts
      ↓
TAESD reduces remaining VAE overhead
```

The result is a practical example of how **quantization, LoRA injection, TorchDynamo guards, dynamic shapes, and AMD/ROCm execution can interact in ComfyUI.**

---

# Status

✅ Working on the tested RX 9060 XT setup

✅ Baked LoRA + Q5_K_S + TorchInductor

✅ Prompt changes tested

✅ Long-prompt behavior tested

✅ 0 recompilations in the validated fast-path runs

✅ Background warmup

✅ Optional TAESD VAE optimization

---

# Contributing / reproducing

If you reproduce the setup on another AMD GPU or another ROCm/PyTorch combination, benchmark the same workflow and report:

```text
GPU
VRAM
ROCm
PyTorch
ComfyUI
FLUX quantization
Resolution
Steps
Sampler
s/it
Total time
Recompilations
```

The most interesting comparison is whether the **baked-LoRA path keeps the fast Inductor behavior** on other RDNA/ROCm systems.

# DeepSeek V4 Flash mini-oracle

This directory contains a deterministic FP32 reference graph for a reduced DeepSeek V4 Flash. It is a correctness fixture for future C, Vulkan, quantized, and out-of-core implementations; it is not a trained language model and its generated token IDs have no linguistic meaning.

## What remains isomorphic

The fixture keeps these production path categories:

- interleaved partial RoPE on the trailing head slice;
- shared-KV multi-query attention (`K == V`);
- grouped low-rank attention output projection;
- sliding attention with a 16-token raw KV window;
- Compressed Sparse Attention with rate 4, overlapping `Ca/Cb` windows, and a lightning indexer;
- Heavily Compressed Attention with rate 128;
- four mHC residual streams and Sinkhorn-projected combination matrices;
- three hash-routed MoE layers followed by learned `sqrtsoftplus` routing;
- 16 routed experts, top-6 activation, one shared expert, and bounded SwiGLU;
- one MTP block using the shared embedding and vocabulary head;
- stateless full-sequence and explicit incremental-state interfaces.

Widths, expert count, vocabulary size, repeated layer count, sliding window, and context ceiling are deliberately reduced.

## Canonical fixture

`tiny_config.json` builds a 7-layer base model plus one MTP block:

| Property | Mini value |
|---|---:|
| Hidden size | 64 |
| Attention heads | 4 |
| Head dimension | 16 |
| Routed experts | 16 |
| Active experts | 6 |
| mHC streams | 4 |
| Maximum sequence | 132 |
| Compression schedule | `0,0,4,128,4,128,4,0` |

The committed CPU oracle currently records:

- 1,100,290 parameters;
- 4,401,160 bytes of FP32 parameter storage;
- a 130-token teacher-forced/incremental comparison;
- boundaries 3/4, 15/16/17, and 126/127/128/129;
- complete route and attention diagnostics at selected positions;
- main and MTP logits fingerprints;
- deterministic greedy token IDs.

Raw tensor SHA-256 values are diagnostic fingerprints for the recorded environment. Cross-platform conformance should use the documented tolerances and exact route/argmax comparisons, not require byte-identical floating-point tensors.

## CPU setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch pytest
python tools/dsv4_mini/run_tests.py
```

Generate the full canonical oracle:

```bash
python -m tools.dsv4_mini.make_oracle \
  --config tools/dsv4_mini/tiny_config.json \
  --output tests/fixtures/dsv4-mini/oracle.json \
  --device cpu \
  --sequence-length 130
```

The test runner disables third-party pytest plugin autoloading and constrains BLAS/OpenMP thread counts. This avoids shutdown deadlocks and oversubscription observed in instrumented Python environments; the mini-oracle itself does not depend on third-party pytest plugins.

The additive Makefile targets are:

```bash
make test-dsv4-mini
make dsv4-mini-oracle
```

They are deliberately separate from the existing C-only `make test`, so a clean C build does not acquire a mandatory PyTorch dependency.

## RTX 2060 Super validation

Install a CUDA-enabled PyTorch build compatible with the installed NVIDIA driver, then verify the device explicitly:

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is not available"
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

Run:

```bash
python -m tools.dsv4_mini.make_oracle \
  --config tools/dsv4_mini/tiny_config.json \
  --output results/dsv4-mini-rtx2060s.json \
  --device cuda \
  --sequence-length 130
```

A 2060 Super has ample memory for this fixture. This path uses ordinary CUDA FP32 tensor operations. It does **not** claim native FP8 or FP4 execution on Turing; software FP8/FP4 codecs and the BC-250 Vulkan path are later milestones.

Expected conformance:

- route indices, hash routes, index selections, argmax IDs, and generated IDs match exactly;
- CPU FP32 full/incremental comparison uses `rtol=1e-5`, `atol=1e-6`;
- CUDA FP32 comparison may use `rtol=2e-4`, `atol=2e-5` when comparing against CPU-selected logits.

## Files

- `config.py`: fail-closed configuration loader and derived layer map.
- `reference.py`: mHC, attention/compressors, MoE, MTP, and decode state.
- `initialization.py`: deterministic adversarial synthetic weights.
- `oracle.py`: parity run, traces, logits samples, and checksums.
- `make_oracle.py`: command-line generator.
- `tiny_config.json`: canonical reduced dimensions.

## Non-goals of this milestone

This code does not load the production checkpoint, train a model, produce meaningful text, encode native FP8/FP4 weights, implement C/Vulkan kernels, or measure real-model route locality. The next milestone should add byte-exact software codecs and a packed checkpoint while keeping this FP32 oracle unchanged as the mathematical reference.

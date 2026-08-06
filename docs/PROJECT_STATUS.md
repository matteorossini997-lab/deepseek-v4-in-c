# Project Status

Date: 2026-08-06

## Current state

`deepseek-v4-in-c` is a staged migration from
`FareedKhan-dev/kimi-k3-in-c`. The repository baseline remains commit
`85ab2cd901aa81b70caac7711f06864d594b8ff3`, but the tree now contains two
independent DeepSeek foundations alongside the unchanged Kimi K3 production
runtime.

The production executable is still the Kimi K3 CPU engine:

- the default binary and public runtime identifiers use the `k3` namespace;
- the executable graph implements KDA, Gated MLA, attention residuals and
  Stable LatentMoE;
- the production tokenizer and checkpoint binder remain Kimi-specific;
- no DeepSeek checkpoint is loaded or executed;
- Vulkan, SSD streaming, Doctor and Autotune are not yet connected to a
  DeepSeek graph.

The checked-in DeepSeek components are correctness and inspection foundations,
not a production inference claim.

## Implemented migration foundations

### P0-A — baseline and provenance

P0-A records the target architecture, exact source refs, license obligations,
and the mandatory source-review checklist.

### P0-B1 — configuration contract

P0-B1 provides:

- portable C99 `DSV4Config` and fail-closed configuration loading;
- normalization of the reviewed Transformers and inference JSON layouts;
- conflict detection when aliases are simultaneously present;
- structural, routing, quantization, context, HCA, sparse-index, MTP and
  compression-schedule checks;
- an exact supported Flash-profile validator;
- `contract_mask`, exposing values derived from explicit constants in the
  reviewed official `inference/model.py` contract;
- `dsv4-inspect`, deterministic text/JSON summaries;
- Make, CMake, CTest and dedicated GitHub Actions coverage.

P0-B1 does not scan Safetensors headers, bind tensors, load weights or execute
the DeepSeek graph.

### FP32 mini-isomorphic oracle

The CPU mini-oracle provides a deterministic reduced DeepSeek V4 Flash graph
and canonical fixture. It covers:

- static hash routing and learned correction-bias top-6 routing;
- normalized routed weights and a shared expert;
- manifold-constrained Hyper-Connections and Sinkhorn combination matrices;
- sliding attention, compressed sparse attention and heavily compressed
  attention boundary state;
- grouped/shared-KV attention behavior;
- one MTP block through a learned-routing decoder layer and shared vocabulary
  head;
- full-sequence versus incremental parity, exact route parity and deterministic
  oracle generation;
- 24 Python tests, Make targets and a CPU GitHub Actions workflow.

The mini-oracle is FP32 and synthetic. It does not validate native FP8/FP4
packing, production tensor names, checkpoint byte layout, Vulkan kernels or
real-checkpoint route locality.

## Target

The target is a DeepSeek V4 Flash-only runtime for AMD BC-250 that combines:

1. the C99, Safetensors, out-of-core and correctness foundations inherited from
   K3-in-C;
2. the Doctor -> Autotune control plane, Vulkan runtime, adaptive multiclass
   cache, telemetry and quality gates developed in `q36-bc250-lab`;
3. the canonical mini-oracle and a new DeepSeek-specific production model layer
   derived from the official model and inference repository.

The project distinguishes:

- **KEEP**: generic code retained with minimal changes;
- **PORT**: code brought from another repository after source review;
- **REWRITE**: model-specific code replaced for DeepSeek;
- **EXPERIMENT**: measurable candidate kept outside the release path;
- **REJECT**: candidate not adopted after review;
- **DEFER**: work blocked by an earlier correctness or state milestone.

## Active milestone

The active implementation milestone is **P0-B2: Safetensors tensor inventory
and model-profile validation**.

P0-B2 is inspection-only. It must prove, from real index/header descriptors:

- shard membership and file presence;
- tensor names, shapes, dtypes and absolute byte spans;
- duplicate, overlap, truncation and unsupported-dtype rejection;
- FP8 block-scale and FP4 expert packing geometry;
- deterministic tensor classification and per-class byte totals;
- model working-set totals used by later Doctor and Autotune stages.

P0-B2 must not execute checkpoint weights. The mini-oracle supplies canonical
model terminology and later parity gates, but it must not be used to invent
production tensor names or sizes.

### P0-B2 implementation status

The `p0b2/safetensors-inventory` branch implements the inspection-only
inventory described above: strict index/header parsing, dtype and byte-span
validation, deterministic tensor classes and totals, exact Flash FP8/FP4
storage-geometry checks, O(1) lookup, JSON/TSV reporting, synthetic fixtures
and an independent Python descriptor comparison.

The implementation is verified on synthetic two-shard fixtures only. The
reviewed 46-shard checkpoint is not present in this environment, so real
checkpoint totals and tensor coverage remain an explicit hardware/data gate.

### P1-A implementation status — native CPU routing and mHC

The `p1a/cpu-routing-hc` stacked branch adds portable FP32 C99 operators for
`sqrtsoftplus`, the canonical hash route, learned correction-bias top-6 routing,
unweighted RMS normalization and the exact mini-oracle mHC/Sinkhorn order.
Canonical vectors are generated from the merged PyTorch classes with explicit
weights and inputs. Attention, experts, MTP and a complete C decoder remain out
of scope for this increment.

## Non-negotiable source-review rule

Before developing from or integrating any external or sibling repository:

1. identify the owning repository and exact ref;
2. inspect its current branch/PR state and recent relevant changes;
3. read the applicable license, NOTICE and contribution instructions;
4. inspect source files and tests, not only summaries or README text;
5. compare the candidate against the target repository's current implementation;
6. record KEEP, PORT, REWRITE, EXPERIMENT, REJECT or DEFER;
7. record the source ref and local modifications in `docs/PROVENANCE.md`;
8. preserve attribution in copied or substantially derived files.

A PR that ports code without this evidence is incomplete even when tests pass.

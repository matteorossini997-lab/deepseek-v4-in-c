# Project Status

Date: 2026-08-06

## Current state

`deepseek-v4-in-c` remains a staged migration from
`FareedKhan-dev/kimi-k3-in-c`. The repository baseline is commit
`85ab2cd901aa81b70caac7711f06864d594b8ff3`, the same commit exposed by the
upstream K3 repository before the migration branches were created.

The checked-in production runtime is still the Kimi K3 CPU engine:

- public runtime identifiers and the default binary use the `k3` namespace;
- the executable model graph implements KDA, Gated MLA, attention residuals and
  Stable LatentMoE;
- the production tokenizer and tensor names are Kimi-specific;
- the production execution path is CPU-only;
- the historical documentation and benchmark data describe Kimi K3, not
  DeepSeek V4 Flash.

P0-B1 adds a separate DeepSeek configuration contract and configuration-only
inspector. This does **not** make the existing runtime a DeepSeek inference
engine.

## Implemented migration foundations

### P0-A — baseline and provenance

P0-A records the target architecture, exact source refs, license obligations,
and the mandatory source-review checklist. It changes documentation and
contribution process only.

### P0-B1 — configuration contract

P0-B1 provides:

- portable C99 `DSV4Config` and fail-closed configuration loading;
- normalization of the two reviewed official DeepSeek V4 Flash JSON layouts;
- conflict detection when aliases are simultaneously present;
- structural, routing, quantization, context, HCA, sparse-index, MTP, and
  compression-schedule checks;
- an exact supported Flash-profile validator;
- `contract_mask`, which exposes values derived from explicit constants in the
  reviewed official `inference/model.py` contract;
- `dsv4-inspect`, which prints a deterministic configuration summary in text or
  JSON;
- Make, CMake, CTest, dedicated Actions, and full-CI coverage.

P0-B1 deliberately does not:

- scan Safetensors headers or inventory tensors;
- validate checkpoint tensor shapes, offsets, shards, or byte counts;
- load or execute FP8/FP4 weights;
- implement routing, MoE, sparse attention, Hyper-Connections, state, MTP
  verification, Vulkan, SSD streaming, Doctor, or Autotune;
- claim DeepSeek token generation.

## Target

The target is a DeepSeek V4 Flash-only runtime for AMD BC-250 that combines:

1. the C99, Safetensors, out-of-core, tiny-oracle and correctness foundations
   inherited from K3-in-C;
2. the Doctor -> Autotune control plane, Vulkan runtime, adaptive multiclass
   cache, telemetry and quality gates developed in `q36-bc250-lab`;
3. a new DeepSeek-specific model layer derived from the official model and
   inference repository.

The project deliberately distinguishes change classes:

- **KEEP**: generic code retained with minimal changes;
- **PORT**: code brought from another repository after source review;
- **REWRITE**: model-specific code replaced for DeepSeek;
- **EXPERIMENT**: measurable candidate kept outside the release path;
- **REJECT**: candidate not adopted after review;
- **DEFER**: work blocked by an earlier correctness or state milestone.

## Active milestone

P0-B1 is implemented on the stacked migration branch and is awaiting the normal
review/merge sequence after P0-A.

The next implementation milestone is **P0-B2: Safetensors tensor inventory and
model-profile validation**. Before writing it, the owning K3 Safetensors reader,
its tests, the current official DeepSeek shard/index metadata, and any relevant
recent source changes must be reviewed at exact refs.

P0-B2 must remain inspection-only: it will prove tensor names, shapes, dtypes,
shards, offsets, byte spans, quantization geometry, and working-set totals before
any checkpoint-backed inference is attempted.

## Non-negotiable source-review rule

Before developing from or integrating any external or sibling repository:

1. identify the owning repository and exact ref;
2. inspect its current branch/PR state and recent relevant changes;
3. read the applicable license, NOTICE and contribution instructions;
4. inspect the source files and their tests, not only summaries or README text;
5. compare the candidate against the target repository's current implementation;
6. record whether the result is KEEP, PORT, REWRITE, EXPERIMENT, REJECT or DEFER;
7. record the source ref and local modifications in `docs/PROVENANCE.md`;
8. preserve source attribution in copied or substantially derived files.

A PR that ports code without this evidence is incomplete even when tests pass.

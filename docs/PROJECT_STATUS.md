# Project Status

Date: 2026-08-06

## Current state

`deepseek-v4-in-c` is currently a renamed GitHub fork of
`FareedKhan-dev/kimi-k3-in-c`. The target repository's `main` branch points to
commit `85ab2cd901aa81b70caac7711f06864d594b8ff3`, the same commit exposed by the
upstream K3 repository.

The checked-in runtime is therefore still the Kimi K3 CPU engine:

- public identifiers and binaries use the `k3` namespace;
- the model graph implements KDA, Gated MLA, attention residuals and Stable
  LatentMoE;
- the tokenizer and tensor names are Kimi-specific;
- the production path is CPU-only;
- the documentation and benchmark data describe Kimi K3, not DeepSeek V4
  Flash.

No checked-in file should be described as a DeepSeek V4 Flash implementation
until it passes the model-contract and oracle gates defined in the integrated
design.

## Target

The target is a DeepSeek V4 Flash-only runtime for AMD BC-250 that combines:

1. the C99, Safetensors, out-of-core, tiny-oracle and correctness foundations
   inherited from K3-in-C;
2. the Doctor -> Autotune control plane, Vulkan runtime, adaptive multiclass
   cache, telemetry and quality gates developed in `q36-bc250-lab`;
3. a new DeepSeek-specific model layer derived from the official model and
   inference repositories.

The project deliberately distinguishes five change classes:

- **KEEP**: generic code retained with minimal changes;
- **PORT**: code brought from another repository after source review;
- **REWRITE**: model-specific code replaced for DeepSeek;
- **EXPERIMENT**: measurable candidate kept outside the release path;
- **DEFER**: work blocked by an earlier correctness or state milestone.

## Active milestone

The active milestone is **P0-A: baseline and provenance**.

P0-A changes documentation and contribution gates only. It does not rename the
runtime, import implementation code, change build outputs, or claim DeepSeek
support. Its purpose is to make the repository's real state and every future
source dependency auditable before semantic work begins.

The next implementation milestone is P0-B/P1: DeepSeek configuration,
checkpoint inspection and model-contract validation. Before that work starts,
the official DeepSeek repository and the existing
`q36-bc250-lab:feat/dsv4-mini-oracle` design must be reviewed at their then-current
refs.

## Non-negotiable source-review rule

Before developing from or integrating any external or sibling repository:

1. identify the owning repository and exact ref;
2. inspect its current branch/PR state and recent relevant changes;
3. read the applicable license, NOTICE and contribution instructions;
4. inspect the source files and their tests, not only summaries or README text;
5. compare the candidate against the target repository's current implementation;
6. record whether the result is KEEP, PORT, REWRITE, EXPERIMENT or REJECT;
7. record the source ref and local modifications in `docs/PROVENANCE.md`;
8. preserve source attribution in copied or substantially derived files.

A PR that ports code without this evidence is incomplete even when tests pass.

# P1-G Native MoE Implementation Plan

**Goal:** Implement a portable FP32 one-token MoE matching the mini-oracle while preserving production-facing explicit hash-table routing.

**Base:** `p1f/native-attention-runtime@1106f248ccea9bc3b06cb031e8ca58542e23c8c7`

## Task 1 — RED fixture and contract

Before production code:

- add `tools/dsv4_mini/make_moe_fixture.py` using the real `SparseMoE` with explicit deterministic weights;
- add `tests/unit/test_dsv4_moe.c` with a test-local declaration of the intended API;
- exercise hash and learned layers, route-pair diagnostics, output parity and transactional refusal;
- run a RED workflow that generates the fixture and reaches an undefined `dsv4_moe_forward_f32` link symbol.

## Task 2 — GREEN native MoE

Add `include/dsv4/dsv4_moe.h` and `src/cpu/dsv4_moe.c`.

Implement:

- fail-closed config and weight geometry validation;
- explicit hash-table routing with unbiased sqrt-softplus weights;
- learned routing through P1-A;
- FP32 SwiGLU expert execution with canonical clamp order;
- selected routed expert accumulation plus one unweighted shared expert;
- transactional output/route publication.

Replace test-local declarations with the production header only after RED is observed.

## Task 3 — Verification and integration

Require:

- deterministic fixture regeneration;
- all existing 24 mini-oracle tests;
- GCC and Clang focused parity;
- GCC `-fanalyzer`;
- ASan + UBSan;
- Make, CMake and CTest integration;
- permanent GCC/Clang workflow;
- mini-oracle workflow fixture regeneration;
- `git diff --check`.

## Task 4 — Provenance and clean PR

Update `PROJECT_STATUS` and `PROVENANCE` with exact refs and semantic differences. Remove temporary RED/GREEN/bootstrap tooling, rewrite the branch to one logical commit over P1-F, open a draft stacked PR, and require all inherited workflows to pass on the final clean SHA.
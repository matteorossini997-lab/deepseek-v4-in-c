# P1-A CPU Routing and mHC Implementation Plan

**Goal:** land the first native DeepSeek FP32 C operators while preserving exact mini-oracle semantics and all existing gates.

## Task 1 — Canonical vectors

- Add a PyTorch generator that imports the merged mini-oracle classes.
- Use explicit inputs, weights, biases and Hyper-Connection parameters.
- Verify the learned route set against `topk(sorted=False)` and reject ties.
- Emit exact C99 hexadecimal FP32 constants.
- Regenerate in the CPU mini-oracle workflow and compare byte-for-byte.

## Task 2 — Portable C API

- Add `include/dsv4/dsv4_cpu_ops.h`.
- Add stable scalar math, hash routing, learned top-k and mHC/Sinkhorn in `src/cpu/dsv4_cpu_ops.c`.
- Define deterministic top-k slot order and fail-closed invalid/numeric statuses.

## Task 3 — TDD gates

- Add exact route-ID tests.
- Compare logits, normalized weights, post weights, combination matrix and collapsed stream against canonical vectors.
- Test correction-bias semantics, hash uniqueness and invalid iteration/cycle cases.
- Run warnings-as-errors, GCC analyzer, ASan and UBSan.

## Task 4 — Build and CI integration

- Add `test_dsv4_cpu_ops` to Make and CMake/CTest.
- Add a focused GCC/Clang CPU operator workflow.
- Extend the mini-oracle workflow with fixture regeneration.
- Run full CI, config, inventory and all 24 Python tests on the final clean head.

## Task 5 — History and provenance

- Record exact source refs and local canonical top-k ordering.
- Keep the branch stacked on P0-B2.
- Remove bootstrap files and rewrite to logical commits before review.

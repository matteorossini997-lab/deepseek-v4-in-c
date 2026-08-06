# P1-B Attention State and Layout Implementation Plan

**Goal:** establish native DeepSeek attention state boundaries and tensor layout
primitives before implementing attention numerics.

## Task 1 — State API

- Add `DSV4AttentionState` and deterministic traces.
- Support only sliding, CSA rate 4 and HCA rate 128.
- Enforce contiguous positions and transactional rejection.
- Expose raw ring slot/start and compressor/indexer emission counts.

## Task 2 — Layout primitives

- Implement one-head shared-KV expansion.
- Implement group-isolated row-major projection matching `GroupedLinear`.
- Add size, finite-value and overflow guards.

## Task 3 — Canonical tests

- Reproduce sliding-window boundaries.
- Reproduce the nine-token CSA count sequence.
- Reproduce the HCA position-127 boundary.
- Verify invalid steps do not mutate state.
- Verify exact shared-KV copies and grouped projection values.

## Task 4 — Build and CI

- Add `test_dsv4_attention_state` to Make and CMake/CTest.
- Include it in weightless `make test`.
- Add a focused GCC/Clang workflow with analyzer and sanitizers.
- Run full CI, config, inventory, CPU routing/mHC and all 24 mini-oracle tests.

## Task 5 — History and provenance

- Record exact mini-oracle source paths and local metadata-only boundary.
- Keep the branch stacked on P1-A.
- Remove integration helpers and rewrite to one logical commit before review.

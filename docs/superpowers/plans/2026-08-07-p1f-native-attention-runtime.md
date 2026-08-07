# P1-F Native Attention Runtime Implementation Plan

> Execute this plan with strict test-first development and source provenance updates.

**Goal:** Implement a portable FP32 batch-one incremental C99 runtime matching `MiniAttention.step` across sliding, CSA and HCA modes.

**Base:** `p1e/postprojected-attention-step@70c79636aa92af42552da035290348e9e504ffa0`

## Task 1 — RED contract and canonical fixture

Create only test-side artifacts first:

- `tools/dsv4_mini/make_attention_runtime_fixture.py` constructing a small valid `MiniConfig`, filling a real `MiniAttention` with deterministic binary-fraction weights and running canonical incremental steps;
- `tests/unit/test_dsv4_attention_runtime.c` with the expected public contract declared locally until the production header exists;
- a temporary branch-only RED workflow that regenerates the fixture and attempts to compile/link the test against the existing P1-E sources.

Required RED evidence: fixture generation succeeds and build reaches the expected undefined runtime symbols. Do not add runtime production implementation before this failure is observed.

## Task 2 — GREEN runtime core

Add:

- `include/dsv4/dsv4_attention_runtime.h`;
- `src/cpu/dsv4_attention_runtime.c`.

Implement:

- checked dimensions and mode-dependent weight validation;
- dense linear helper;
- weighted and unweighted RMS normalization;
- owned chronological raw KV window;
- owned attention compressor projected-window state and append-only compressed KV;
- independent CSA indexer compressor and projections;
- clone/apply/swap transactional state;
- one-token FP32 step calling P1-E for the verified post-projection graph;
- trace publication matching mini-oracle counts and selected indices;
- init/reset/destroy/status helpers.

Replace the test-local declarations with the production header without changing the asserted behavior.

## Task 3 — Focused verification

Run in CI on the branch:

- canonical fixture regeneration and byte comparison;
- focused GCC and Clang builds with `-Werror -ffp-contract=off`;
- parity unit test across sliding, CSA and HCA sequences;
- rejection/transactionality tests;
- GCC `-fanalyzer`;
- ASan + UBSan;
- CMake + CTest;
- `git diff --check`.

## Task 4 — Permanent integration

Only after focused gates pass:

- add `test-dsv4-attention-runtime` to Make and weightless `make test`;
- add CMake library/test and CTest registration;
- extend `.github/workflows/dsv4-mini.yml` to regenerate the new fixture;
- add a dedicated permanent GCC/Clang runtime workflow;
- update `docs/PROJECT_STATUS.md` and `docs/PROVENANCE.md` with exact refs and limitations.

Use a temporary bootstrap only if needed to apply multi-file integration and publish only after all gates pass. Remove temporary bootstrap/patcher artifacts before the final commit.

## Task 5 — Clean stacked PR

- compare final tree against P1-E;
- ensure only intentional runtime/test/build/docs/CI files remain;
- squash/rewrite branch to one logical commit while preserving the verified tree;
- open a draft PR based on `p1e/postprojected-attention-step`;
- require the runtime matrix, mini-oracle, full CI and all inherited DeepSeek gates to pass on the final clean SHA;
- update the PR body with final run IDs, exact head SHA, file count and explicit non-goals.
# P1-H Native Decoder Layer Implementation Plan

**Goal:** Compose P1-A, P1-F and P1-G into a transactionally stateful FP32 one-token decoder layer matching `DecoderLayer.step`.

**Base:** `p1g/native-moe@7195f1e3ff54e310ee35fb5eacdee446c31a8c78`

## Task 1 — RED canonical layer contract

Before production layer code:

- add `tools/dsv4_mini/make_decoder_layer_fixture.py` using real `DecoderLayer.step`;
- fill attention, MoE, both HC modules and both weighted RMS norms with explicit binary-fraction formulas;
- cover sliding/hash, CSA/learned and HCA/learned sequences;
- add `tests/unit/test_dsv4_decoder_layer.c` with a test-local public contract;
- include a rollback test that injects a non-finite shared-expert weight, then retries the same position after repair;
- run a RED workflow that generates the fixture and fails exactly on undefined `dsv4_decoder_layer_*` symbols.

## Task 2 — GREEN decoder runtime

Add:

- `include/dsv4/dsv4_decoder_layer.h`;
- `src/cpu/dsv4_decoder_layer.c`.

Extend P1-F additively:

- expose `dsv4_attention_runtime_clone` in `dsv4_attention_runtime.h`;
- reuse the existing deep-copy implementation instead of creating a second state serializer.

Implement:

- layer config consistency validation;
- weighted RMSNorm helper;
- exact mini-oracle HC post-mix (`comb^T @ residual + post * sublayer`);
- clone -> attention -> MoE -> commit orchestration;
- atomic output/diagnostic publication;
- create/reset/destroy/status API.

Replace the test-local declarations with the production header only after RED is observed.

## Task 3 — Focused and robustness gates

Require:

- deterministic decoder-layer fixture regeneration;
- all existing mini-oracle tests;
- GCC and Clang focused parity;
- GCC `-fanalyzer` for decoder-layer glue and the newly public attention clone path;
- ASan + UBSan;
- CMake/CTest;
- rollback retry test;
- `git diff --check`.

## Task 4 — Permanent integration

After focused gates pass:

- add `test-dsv4-decoder-layer` to Make and weightless `make test`;
- add CMake library/test linked against attention runtime, MoE and CPU ops;
- extend mini-oracle CI to regenerate the decoder fixture;
- add a permanent GCC/Clang decoder-layer workflow;
- update project status and provenance with exact refs and limitations.

## Task 5 — Clean stacked PR

Remove temporary RED/GREEN/bootstrap/patcher artifacts, compare against P1-G, rewrite to one logical commit, open a draft PR based on `p1g/native-moe`, and require the decoder workflow plus every inherited workflow to pass on the final clean SHA.
# P1-I Native Base Model Shell Implementation Plan

**Goal:** Implement incremental FP32 base-model logits by composing P1-H decoder layers with embedding, HyperHead, final norm and LM head.

**Base:** `p1h/native-decoder-layer@df79a91c4e24bf3629dd4bd6c57557163393a0ce`

## Task 1 — RED model fixture and contract

Before production model code:

- add `tools/dsv4_mini/make_model_shell_fixture.py` using the real `MiniReferenceModel` embedding/layers/HyperHead/output norm/LM head while deliberately skipping MTP;
- fill all weights with deterministic binary-fraction formulas, reusing the P1-H layer filler;
- run 129 incremental tokens so the base model crosses sliding, CSA and HCA boundaries;
- record final streams, logits and argmax token per position;
- add `tests/unit/test_dsv4_model.c` with test-local API declarations;
- include a multi-layer rollback test by removing a learned correction bias from a later layer, then repairing it and retrying the same position;
- run a RED workflow that reaches undefined `dsv4_model_*` symbols after fixture generation and dependency compilation.

## Task 2 — GREEN model shell

Add:

- `include/dsv4/dsv4_model.h`;
- `src/cpu/dsv4_model.c`.

Extend P1-H additively:

- expose `dsv4_decoder_layer_clone` using the existing decoder config plus a deep-cloned attention runtime.

Implement:

- model config consistency validation and internal copy of layer configs;
- owned decoder-layer array and next position;
- embedding row gather and HC stream expansion;
- deep-clone-all -> sequential layer execution -> commit-all transaction;
- exact HyperHead flatten/unweighted-RMS/sigmoid-collapse;
- final weighted RMSNorm and dense LM head;
- create/reset/destroy/status API;
- output streams and logits published only after final finite validation.

Replace test-local declarations with the production header only after RED is observed.

## Task 3 — Verification

Require:

- deterministic 129-token fixture regeneration;
- all existing mini-oracle tests;
- GCC and Clang focused model parity;
- GCC analyzers for model shell and decoder clone path;
- ASan + UBSan;
- CMake/CTest;
- multi-layer rollback retry;
- `git diff --check`.

## Task 4 — Permanent integration and clean PR

- add model gate to Make/weightless `make test` and CMake/CTest;
- extend mini-oracle CI with the model-shell fixture;
- add permanent GCC/Clang model workflow;
- update project status/provenance with exact refs and MTP deferral;
- remove RED/GREEN/bootstrap/patcher artifacts;
- rewrite to one logical commit over P1-H;
- open a draft PR and require the new workflow plus every inherited gate on the clean SHA.
# P1-J Native MTP — Implementation Plan

> Execute test-first. Do not change the public parity claim until the same canonical fixture passes both the Python reference and C runtime.

## Task 1 — Canonical MTP fixture

Create `tools/dsv4_mini/make_mtp_fixture.py`.

- reuse `decoder_config`, deterministic pattern helpers and P1-I base-model fill helpers;
- instantiate the real `MiniReferenceModel`;
- fill the base model deterministically, then fill only MTP-owned weights with independent deterministic tags;
- reuse the exact shared embedding and LM head from the base model;
- produce base final streams by running the P1-I base path incrementally;
- feed those streams and the same token ids to `model.mtp.step` with its own state;
- emit MTP-owned weights, the shared embedding/head references, per-position base streams, MTP final streams, logits, argmax and learned route pairs as hexadecimal FP32 constants.

## Task 2 — RED public contract

Create `include/dsv4/dsv4_mtp.h` as a test-first contract only in the test branch context and `tests/unit/test_dsv4_mtp.c`.

The test must:

- reconstruct a `DSV4MTPConfig` whose decoder is the extra layer index and rate zero;
- map every emitted fixture weight directly;
- execute all fixture tokens incrementally;
- compare MTP streams/logits with tight FP32 tolerances;
- compare route expert/weight pairs independent of slot order when appropriate;
- verify argmax equality;
- inject a guaranteed late decoder/MoE error and prove retry of the same position succeeds without output mutation.

Run a RED workflow that generates the canonical fixture, compiles all P1-H dependencies, omits `src/cpu/dsv4_mtp.c`, and must fail at link time only on `dsv4_mtp_*` symbols.

## Task 3 — GREEN standalone runtime

Implement `src/cpu/dsv4_mtp.c` and finalize `include/dsv4/dsv4_mtp.h`.

Implementation helpers:

- checked size multiplication;
- weighted per-vector RMSNorm;
- dense row-major FP32 projection;
- unweighted flattened HC RMS + HyperHead;
- finite validation and transactional temporary buffers.

Runtime ownership:

- copy `DSV4MTPConfig`;
- own one `DSV4DecoderLayer`;
- own `next_position`;
- never own caller weight storage.

Step algorithm:

1. validate all mandatory pointers and capacities;
2. clone decoder state;
3. compute projected embedding and projected normalized streams;
4. build combined HC input;
5. run cloned decoder at `next_position`;
6. compute HyperHead, final norm and shared LM logits;
7. verify all results finite;
8. atomically replace live decoder and increment `next_position`;
9. publish streams/logits/routes/trace.

## Task 4 — Build and permanent CI

Integrate:

- Make target `test-dsv4-mtp` and weightless `make test`;
- CMake static library, unit executable and CTest;
- dedicated GCC/Clang workflow with analyzer and sanitizers;
- mini-oracle workflow regeneration/cmp of `dsv4_mtp_vectors.h`.

## Task 5 — Provenance and final verification

Record the exact P1-I base SHA, mini-oracle files, official current `MTPBlock`, official converter shared-weight behavior and q36 design-only branch.

Final gate on a clean one-commit P1-J head:

- focused MTP GCC/Clang;
- analyzer;
- ASan+UBSan;
- CMake/CTest;
- canonical fixture regeneration;
- full mini-oracle;
- full inherited CI;
- branch compare contains only intentional P1-J paths and no TDD/bootstrap scaffolding.

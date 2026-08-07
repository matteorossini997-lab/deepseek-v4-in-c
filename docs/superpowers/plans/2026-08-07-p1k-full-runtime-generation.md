# P1-K Full Runtime + Greedy Generation — Implementation Plan

## Task 1 — Canonical combined fixture

Create `tools/dsv4_mini/make_full_runtime_fixture.py`.

- instantiate the real `MiniReferenceModel` with `decoder_config()`;
- apply `fill_base_model` and `fill_mtp` so weights exactly match the existing P1-I/P1-J C fixture arrays;
- execute 20 incremental `MiniReferenceModel.step` calls and record base logits, MTP logits, both argmax values and MTP route pairs;
- run `MiniReferenceModel.generate` on a deterministic 7-token prompt for 5 new tokens and record the exact 12-token sequence;
- emit expectation arrays only; do not duplicate weights.

## Task 2 — RED contracts

Extend public P1-I/P1-J headers with clone declarations and add `include/dsv4/dsv4_full_model.h`.

Create `tests/unit/test_dsv4_full_model.c` that:

- builds the P1-I base config/weights from `dsv4_model_shell_vectors.h`;
- builds P1-J MTP config/weights from `dsv4_mtp_vectors.h`;
- verifies 20 combined steps against the new expectation fixture;
- injects a guaranteed learned-MTP missing-bias failure after the staged base step, asserts all outputs remain sentinel values, restores the bias and retries token zero successfully;
- verifies greedy prompt+5 output exactly;
- rejects mismatched shared embedding/head pointers.

RED compiles all P1-I/P1-J dependencies without clone/full-runtime implementations and must fail only on the new full-runtime symbols.

## Task 3 — Deep-clone primitives

Implement `dsv4_model_clone`:

- allocate a new model shell;
- deep-copy layer configs;
- deep-clone every P1-H decoder layer;
- copy `next_position`;
- never copy caller-owned weights.

Implement `dsv4_mtp_clone`:

- allocate runtime;
- copy config;
- clone its P1-H decoder;
- copy `next_position`.

Clone failures must not mutate the source.

## Task 4 — Full transactional runtime

Implement `src/cpu/dsv4_full_model.c`.

- own one base runtime and one MTP runtime;
- deep-copy enough config for atomic reset;
- require matching vocab/hidden/HC/max-seq geometry and MTP layer index after the base layers;
- on step, require MTP embedding/head pointers to be identical to base embedding/head pointers;
- clone both runtimes, run base clone, then MTP clone on staged base streams;
- commit both only after all results succeed and are finite;
- publish outputs only after commit.

## Task 5 — Greedy generation

Implement generation on a reset full runtime.

- validate prompt, output capacity and sequence limit;
- process all prompt tokens through the combined step;
- choose each generated token by argmax of **base logits only**;
- append token, honor optional EOS, feed it to obtain the next logits;
- return prompt+completion length and tokens;
- MTP remains part of transaction/diagnostics but does not influence token selection.

## Task 6 — Integration and permanent CI

Add Make/CMake/CTest targets and a dedicated GCC/Clang workflow with analyzer and sanitizers. Add the tenth canonical fixture to `dsv4-mini.yml` regeneration.

Final clean-head acceptance:

- focused combined parity and generation;
- rollback across base+MTP;
- GCC/Clang;
- analyzer;
- ASan+UBSan;
- CMake/CTest;
- full mini-oracle;
- full CI;
- one logical P1-K commit and no TDD/bootstrap scaffolding.

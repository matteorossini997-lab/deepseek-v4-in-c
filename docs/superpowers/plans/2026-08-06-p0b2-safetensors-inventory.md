# P0-B2 Safetensors Inventory Implementation Plan

> Execute task-by-task with tests and review checkpoints. Do not advance to
> checkpoint-backed inference until every P0-B2 gate is green.

**Goal:** prove the production checkpoint's metadata, storage geometry and
working-set totals without reading tensor payloads.

**Architecture:** a standalone C99 DeepSeek inventory module parses the Hugging
Face index and each Safetensors header. A CLI emits deterministic reports and can
validate the inventory against `DSV4Config`. K3 production I/O remains unchanged.

**Tooling:** C99, Make, CMake/CTest, Python standard library, GitHub Actions.

## Task 1 — Freeze source review and design

Files:

- add `docs/superpowers/specs/2026-08-06-dsv4-safetensors-inventory-design.md`;
- add this plan;
- update `docs/PROVENANCE.md` and `docs/PROJECT_STATUS.md` after implementation.

Record target, K3, official DeepSeek code and immutable checkpoint refs. Record
that recent repository changes were checked and that the real checkpoint is not
present in this environment.

## Task 2 — RED inventory contract

Files:

- add `include/dsv4/dsv4_inventory.h`;
- add `tests/unit/test_dsv4_inventory.c` before production implementation.

The first build must fail because inventory symbols are missing. Preserve the
failure evidence in the PR description.

## Task 3 — GREEN strict parser and inventory

Files:

- add `src/io/dsv4_inventory.c`.

Implement index parsing, header-only shard scanning, dtype bit widths, offset and
span checks, duplicate/path/overflow guards, O(1) lookup, classification and
aggregate totals. Run the focused C test with `-Werror`, ASan+UBSan and GCC
`-fanalyzer`.

## Task 4 — Flash profile validation

Extend the focused test before production changes with invalid FP8, packed-FP4,
routing and identifier geometry. Implement `dsv4_inventory_validate_flash`
against a validated `DSV4Config`.

## Task 5 — CLI and independent verifier

Files:

- add `src/cli/dsv4_inventory_cli.c`;
- add `tools/make_dsv4_inventory_fixture.py`;
- add `tools/verify_dsv4_inventory.py`.

Build `dsv4-inventory`, generate a deterministic two-shard synthetic fixture,
emit full JSON/TSV, and compare every raw descriptor using the independent Python
parser.

## Task 6 — Build-system integration

Files:

- modify `Makefile`;
- modify `CMakeLists.txt`.

Add `dsv4-inventory`, `test-dsv4-inventory` and matching CTest entries without
removing config-contract, mini-oracle or K3 targets. `make test` must include the
focused inventory gate.

## Task 7 — CI integration

Files:

- add `.github/workflows/dsv4-inventory.yml`;
- extend full CI only where needed to invoke the new Make target.

The dedicated workflow must run focused GCC/Clang builds, ASan+UBSan,
`-fanalyzer`, CMake/CTest and the independent Python comparison. Existing config,
mini-oracle and complete K3 CI must remain green.

## Task 8 — Documentation and PR

Update project status/provenance with exact files, semantic differences, TDD
runs and explicit unverified hardware/data gates. Open a draft PR from
`p0b2/safetensors-inventory` to `main`.

## Completion gate

P0-B2 is complete only when:

1. focused inventory CI is green;
2. full C CI is green;
3. config-contract and 24-test mini-oracle workflows remain green;
4. Make and CMake expose equivalent inventory tests;
5. the branch contains no generated shard payloads;
6. documentation does not claim real-checkpoint verification.

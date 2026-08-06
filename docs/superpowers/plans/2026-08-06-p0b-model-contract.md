# P0-B1 DeepSeek Model Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development`.
> The first implementation commit must contain tests without production code and the
> dedicated CI job must fail for the expected missing-header/source reason.

**Goal:** Add a portable, fail-closed DeepSeek V4 Flash configuration normalizer and
configuration-only inspector without changing the existing Kimi K3 runtime.

**Architecture:** `dsv4_config` accepts the two official configuration layouts at the
recorded DeepSeek source revision: the root Transformers `config.json` and
`inference/config.json`. It normalizes aliases into one `DSV4Config`, rejects missing,
conflicting or structurally inconsistent fields, records values supplied by the official
`model.py` contract, and exposes a production-profile check. `dsv4-inspect` is a
configuration-only CLI; Safetensors inventory is a later P0-B2 task.

**Tech Stack:** portable C99, existing `third_party/json.h`, Make, CMake, GitHub Actions.

## Global constraints

- Source contract: `deepseek-ai/DeepSeek-V4-Flash@60d8d70770c6776ff598c94bb586a859a38244f1`.
- Current K3 binary, public API and tests remain unchanged.
- Missing required fields are errors; there are no guessed architecture defaults.
- Values omitted by `inference/config.json` may only come from explicit constants in the
  reviewed official `inference/model.py`, and must be reported as contract-derived.
- Both official layouts normalize to identical production fields.
- Conflicting aliases are rejected even when either layout alone would be valid.
- No model weights or generated packed artifacts are committed.
- Make and CMake expose the same new tests and CLI.
- Every commit is independently reviewable.

---

### Task 1: RED — executable model-contract tests

**Files:**
- Create: `tests/unit/test_dsv4_config.c`
- Create: `tests/fixtures/dsv4/config_hf.json`
- Create: `tests/fixtures/dsv4/config_inference.json`
- Create: `tests/fixtures/dsv4/config_missing.json`
- Create: `tests/fixtures/dsv4/config_conflict.json`
- Create: `tests/fixtures/dsv4/config_bad_ratios.json`
- Create: `tests/fixtures/dsv4/config_bad_topk.json`
- Create: `tests/fixtures/dsv4/config_bad_format.json`
- Create: `.github/workflows/dsv4-config.yml`

**Interfaces:**
- Consumes the future `dsv4_config.h`.
- Requires:
  `int dsv4_config_load_file(DSV4Config *, const char *, DSV4ConfigError *)`;
  `int dsv4_config_is_flash_profile(const DSV4Config *, DSV4ConfigError *)`;
  `const char *dsv4_config_layout_name(DSV4ConfigLayout)`.
- The test binary modes are `hf`, `inference`, and `reject`.

- [ ] Commit tests and fixtures without production code.
- [ ] Open a stacked draft PR against `p0/baseline-contract`.
- [ ] Verify the dedicated job fails because `dsv4_config.h` is absent.

### Task 2: GREEN — parser and profile validation

**Files:**
- Create: `include/dsv4/dsv4_config.h`
- Create: `src/model/dsv4_config.c`

**Required behavior:**
- Normalize official aliases and nested values.
- Reject absent values and integer/type/range errors.
- Reject conflicting aliases.
- Validate compression count equals base layers plus MTP layers.
- Validate top-k, routing, HCA, index and quantization geometry.
- Report model.py-derived values through `contract_mask`.
- Validate the exact DeepSeek V4 Flash production profile separately.

- [ ] Implement the smallest parser satisfying the tests.
- [ ] Verify the dedicated CI job passes.
- [ ] Verify the existing CI matrix remains green.

### Task 3: Inspector and build-system parity

**Files:**
- Create: `src/cli/dsv4_inspect.c`
- Modify: `Makefile`
- Modify: `CMakeLists.txt`
- Modify: `.github/workflows/dsv4-config.yml`

**Required behavior:**
- `dsv4-inspect [--json] <config.json>` prints a deterministic normalized summary.
- Exit `0` for a valid production profile, `2` for usage, and `3` for invalid config/profile.
- `make test-dsv4-config`, `make dsv4-inspect`, and matching CMake/ctest entries exist.
- Existing default `make` still builds only the K3 runtime until a later migration task.

### Task 4: Provenance and completion evidence

**Files:**
- Modify: `docs/PROVENANCE.md`
- Modify: `docs/PROJECT_STATUS.md`

- [ ] Record exact official files reviewed and local differences.
- [ ] Record RED and GREEN CI run URLs.
- [ ] Run the full PR CI matrix.
- [ ] Confirm the stacked PR changes only P0-B1 files relative to `p0/baseline-contract`.

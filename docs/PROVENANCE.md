# Provenance and Source Review Register

This file is the authoritative register for code, designs and test methodology
brought into `deepseek-v4-in-c` from another repository.

## Repository snapshots reviewed for P0-A

| Source | Ref reviewed | License / notice | Result |
|---|---|---|---|
| `matteorossini997-lab/deepseek-v4-in-c` | `main@85ab2cd901aa81b70caac7711f06864d594b8ff3` | Apache-2.0; existing `NOTICE` | Target baseline. Still identical to upstream K3 at this ref. |
| `FareedKhan-dev/kimi-k3-in-c` | `main@85ab2cd901aa81b70caac7711f06864d594b8ff3` | Apache-2.0; project and third-party notices in `NOTICE` | Origin of the baseline runtime. KEEP generic infrastructure; REWRITE Kimi model layer. |
| `matteorossini997-lab/q36-bc250-lab` | `main@5b6da88f5039b9124a239b16e6c7988e4183273f` | MIT; retain copyright and permission notice in copied/substantially derived files | Source candidate for BC-250 Doctor/Autotune, Vulkan runtime, adaptive cache, state and validation technology. |
| `matteorossini997-lab/q36-bc250-lab` | `feat/dsv4-mini-oracle@d6684ca33d4da496cd2e805ce159c2db7892f98a` | Same repository license | Reviewed as design-only input: one specification and no implementation. ADAPT requirements only. |
| `deepseek-ai/DeepSeek-V4-Flash` | official Hugging Face repository, observed revision `60d8d70770c6776ff598c94bb586a859a38244f1` | MIT model repository; model weights remain subject to the repository terms | Authoritative model/config/inference contract. Re-resolve before every model-layer milestone. |

## P0-B1: DeepSeek configuration contract

**Review date:** 2026-08-06  
**Target:** `DSV4Config`, `dsv4_config_load_file`,
`dsv4_config_is_flash_profile`, and the configuration-only `dsv4-inspect` CLI.

| Source | Exact ref and files reviewed | Decision |
|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash` | `60d8d70770c6776ff598c94bb586a859a38244f1`: root `config.json`, `inference/config.json`, `inference/model.py`, `inference/convert.py`, `inference/kernel.py`, and license | REWRITE the configuration contract in portable C99. No source file copied. Values absent from one official JSON layout may be supplied only when explicit constants in the reviewed `model.py`, and are exposed through `contract_mask`. |
| `FareedKhan-dev/kimi-k3-in-c` | `85ab2cd901aa81b70caac7711f06864d594b8ff3`: `include/k3/k3_cfg.h`, `tests/unit/test_cfg.c`, `third_party/json.h`, `Makefile`, `CMakeLists.txt` | KEEP fail-closed discipline and build/test conventions; REWRITE all Kimi fields, aliases and profile semantics. |
| `matteorossini997-lab/q36-bc250-lab` | `feat/dsv4-mini-oracle@d6684ca33d4da496cd2e805ce159c2db7892f98a`: `docs/superpowers/specs/2026-08-06-dsv4-mini-oracle-design.md` | ADAPT requirements only. No implementation copied. |

Local differences and invariants:

- both official JSON layouts normalize into one `DSV4Config`;
- simultaneously present aliases must agree;
- absent, mistyped, out-of-range, inconsistent or unsupported fields are hard
  errors;
- exact Flash-profile validation is separate from structural validity;
- `contract_mask` distinguishes JSON values from explicit official constants;
- fixtures are compact structurally faithful layouts, not byte-for-byte copies;
- no Safetensors inspection, binding, inference or production-weight claim.

TDD and verification evidence:

- RED Actions run `31092829144` failed because production header/source were
  absent;
- dedicated GREEN run `31093229027` passed valid layouts and refusal cases;
- build integration run `31093780602` passed Make, CMake and nine CTest cases;
- permanent config workflow run `31094450505` passed on cleaned history;
- full existing CI run `31094449453` passed on the same tree.

## FP32 mini-isomorphic oracle

**Review/merge date:** 2026-08-06  
**Target:** `tools/dsv4_mini`, `tests/python`, canonical
`tests/fixtures/dsv4-mini/oracle.json`, Make targets and CPU workflow.

| Source | Exact ref and files reviewed | Decision |
|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash` | `60d8d70770c6776ff598c94bb586a859a38244f1`: configuration and inference model/operator order | REWRITE a reduced deterministic FP32 reference. No production weights or official source file copied. |
| `matteorossini997-lab/q36-bc250-lab` | `feat/dsv4-mini-oracle@d6684ca33d4da496cd2e805ce159c2db7892f98a`: mini-oracle design specification | ADAPT the fidelity boundary and acceptance criteria. |
| target repository | PR #2, merged as `13952eff5dcf75ce75e692d5a03e714e47de2fd4` | Local implementation and canonical fixture. |

The mini-oracle preserves reduced forms of hash and learned top-6 routing,
correction-bias selection, shared expert execution, mHC/Sinkhorn, shared-KV
attention, grouped projection, sliding/CSA/HCA state and one MTP block. It is
FP32 synthetic reference code; it does not validate checkpoint tensor names,
FP8/FP4 bitstreams, Vulkan or SSD behavior.

Recorded verification includes 24 Python tests, deterministic full/incremental
parity, route parity, boundary checks at sliding/CSA/HCA emission positions and
canonical oracle generation. The CPU workflow is authoritative for this
milestone; RTX 2060 Super and BC-250 runs remain unverified.

## Existing baseline provenance

The target was forked from K3-in-C. Its Apache-2.0 `NOTICE` documents vendored
JSON/tokenizer components and Kimi-specific modifications. Those notices remain
authoritative until files are removed or replaced.

## Planned source matrix

| Target area | Primary source to review first | Expected action | Import status |
|---|---|---|---|
| Safetensors scanner and aligned reads | K3 `src/io/k3_st.c`, `tests/unit/test_st.c`, `tools/verify_st.py` | KEEP/EXTEND | P0-B2 active |
| Config reader and binder discipline | K3 config/binder plus official DeepSeek config/model | REWRITE pattern | Configuration normalizer complete; tensor binder not started |
| Expert loader and trunk packing | K3 `src/io/k3_load.c`, `src/io/k3_trunk.c`, pack tools | ADAPT | Not started |
| Tiny oracle methodology | Local mini-oracle plus official DeepSeek graph | KEEP/EXTEND | FP32 mini-oracle complete; native C/Vulkan parity not started |
| DeepSeek graph and formats | Official DeepSeek config, `inference/model.py`, `kernel.py`, `convert.py` | REWRITE | Config and FP32 mini reference complete; production graph/formats not started |
| Doctor and Autotune | q36 Doctor/Autotune sources and tests | PORT/GENERALIZE | Not started |
| Vulkan tensor runtime | q36 GPU API, Vulkan source, shader/test infrastructure | EXTRACT/PORT | Not started |
| Adaptive expert cache | q36 adaptive cache v4 and multiclass implementation | PORT/GENERALIZE | Not started |
| State codecs/checkpointing | q36 codec framework and persistent store | PORT FRAMEWORK/REWRITE STATE | Not started |
| Benchmark, quality and capture | K3 harnesses plus q36 bench/quality/capture | MERGE METHODS | Not started |

## Required entry for every future port

| Field | Required value |
|---|---|
| Target component | Exact target files and public interfaces |
| Owning source repository | `owner/repository` |
| Source ref | Full commit SHA; branch name alone is insufficient |
| Files reviewed | Exact paths, including tests and design documents |
| License review | License and NOTICE obligations |
| Upstream state | Relevant branches, PRs and commits checked on review date |
| Decision | KEEP, PORT, REWRITE, EXPERIMENT, REJECT or DEFER |
| Local changes | Semantic differences from source implementation |
| Verification | Tests, parity gates and hardware status |
| Reviewer/date | Person or agent and ISO date |

## Attribution rules

- Preserve Apache-2.0 notices for retained K3 files and vendored components.
- Preserve the MIT copyright and permission notice for copied or substantially
  derived q36 files.
- Add an SPDX identifier and `Derived from ... @ <sha>` comment for substantial
  source-derived implementations.
- Reimplemented ideas still require a provenance row, but not a misleading
  byte-derived claim.
- Generated fixtures must identify generator and authoritative reference ref.
- Model weights and packed artifacts must not be committed without explicit
  license and repository-policy review.

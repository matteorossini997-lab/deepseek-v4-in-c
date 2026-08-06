# Provenance and Source Review Register

This file is the authoritative register for code, designs and test methodology
brought into `deepseek-v4-in-c` from another repository.

## Repository snapshots reviewed for P0-A

| Source | Ref reviewed | License / notice | Result |
|---|---|---|---|
| `matteorossini997-lab/deepseek-v4-in-c` | `main@85ab2cd901aa81b70caac7711f06864d594b8ff3` | Apache-2.0; existing `NOTICE` | Target baseline. Still identical to upstream K3 at this ref. |
| `FareedKhan-dev/kimi-k3-in-c` | `main@85ab2cd901aa81b70caac7711f06864d594b8ff3` | Apache-2.0; project and third-party notices in `NOTICE` | Origin of all current runtime code. KEEP generic infrastructure; REWRITE Kimi model layer. |
| `matteorossini997-lab/q36-bc250-lab` | `main@5b6da88f5039b9124a239b16e6c7988e4183273f` | MIT; retain copyright and permission notice in copied/substantially derived files | Source candidate for BC-250 Doctor/Autotune, Vulkan runtime, adaptive cache, state and validation technology. No code imported in P0-A. |
| `matteorossini997-lab/q36-bc250-lab` | `feat/dsv4-mini-oracle@d6684ca33d4da496cd2e805ce159c2db7892f98a` | Same repository license | Reviewed as design-only branch: one added specification and no implementation. ADAPT design after rechecking official DeepSeek sources. |
| `deepseek-ai/DeepSeek-V4-Flash` | official Hugging Face repository, observed revision `60d8d70770c6776ff598c94bb586a859a38244f1` | MIT model repository; model weights remain subject to the repository's terms | Authoritative model/config/inference contract. No code or weights imported in P0-A. Re-resolve the current revision before every model-layer milestone. |

## P0-B1: DeepSeek configuration contract

**Review date:** 2026-08-06  
**Target:** `DSV4Config`, `dsv4_config_load_file`,
`dsv4_config_is_flash_profile`, and the configuration-only `dsv4-inspect` CLI.

| Source | Exact ref and files reviewed | Decision |
|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash` | `60d8d70770c6776ff598c94bb586a859a38244f1`: root `config.json`, `inference/config.json`, `inference/model.py`, `inference/convert.py`, `inference/kernel.py`, and repository license | REWRITE the configuration contract in portable C99. No source file was copied. Values absent from one official JSON layout may be supplied only when they are explicit constants in the reviewed `model.py`, and are exposed through `contract_mask`. |
| `FareedKhan-dev/kimi-k3-in-c` | `85ab2cd901aa81b70caac7711f06864d594b8ff3`: `include/k3/k3_cfg.h`, `tests/unit/test_cfg.c`, `third_party/json.h`, `Makefile`, and `CMakeLists.txt` | KEEP the fail-closed discipline and build/test conventions; REWRITE all Kimi fields, aliases, structural checks, and profile semantics. |
| `matteorossini997-lab/q36-bc250-lab` | `feat/dsv4-mini-oracle@d6684ca33d4da496cd2e805ce159c2db7892f98a`: `docs/superpowers/specs/2026-08-06-dsv4-mini-oracle-design.md` | ADAPT requirements only. The branch contained no implementation. |

Local differences and invariants:

- both official JSON layouts normalize into one `DSV4Config`;
- aliases present simultaneously must agree; conflicts are hard errors;
- missing, mistyped, out-of-range, structurally inconsistent, or unsupported
  fields are hard errors;
- the exact Flash production profile is checked separately from basic structural
  validity;
- `contract_mask` distinguishes values read from JSON from explicit constants
  derived from the reviewed official `model.py` contract;
- fixtures are small, structurally faithful official-layout fixtures assembled
  from the reviewed configuration and model contract. They are not claimed to
  be byte-for-byte copies of either official JSON document;
- P0-B1 does not inspect Safetensors, bind tensors, load weights, execute the
  DeepSeek graph, or claim DeepSeek inference support.

TDD and verification evidence:

- RED: Actions run `31092829144` failed exactly because
  `dsv4_config.h` and `src/model/dsv4_config.c` were absent;
- dedicated GREEN: run `31093229027` passed both valid layouts, seven
  fail-closed cases, and inspector output;
- Make/CMake integration proof: run `31093780602` passed
  `make test-dsv4-config`, both build systems, and all nine CTest cases before
  publishing the build integration;
- permanent configuration workflow: run `31093948860` passed;
- complete existing CI matrix: run `31093948456` passed, including GCC/Clang,
  warnings-as-errors, sanitizers, lint, tokenizer parity, and weightless K3
  oracle tests.

## Existing baseline provenance

The current target tree was forked unchanged from K3-in-C. Its existing
Apache-2.0 `NOTICE` documents vendored JSON/tokenizer components and Kimi-specific
modifications. Those notices remain authoritative until files are removed or
replaced. Renaming the repository does not remove the obligation to preserve
them.

## Planned source matrix

| Target area | Primary source to review first | Expected action | Import status |
|---|---|---|---|
| Safetensors scanner and aligned reads | K3 `src/io/k3_st.c` and tests | KEEP/EXTEND | P0-B2 not started |
| Config reader and binder discipline | K3 `include/k3/k3_cfg.h`, `src/model/k3_bind.c` | REWRITE pattern | Configuration normalizer complete in P0-B1; tensor binder not started |
| Expert loader and trunk packing | K3 `src/io/k3_load.c`, `src/io/k3_trunk.c`, pack tools | ADAPT | Not started |
| Tiny oracle methodology | K3 oracle/reference tools plus q36 `feat/dsv4-mini-oracle` design | ADAPT/REWRITE | Design reviewed; implementation not started |
| DeepSeek graph and formats | Official DeepSeek `config.json`, `inference/model.py`, `kernel.py`, `convert.py` | REWRITE | Configuration contract complete; graph, tensor formats, and inference not started |
| Doctor and Autotune | q36 main Doctor/Autotune sources and tests | PORT/GENERALIZE | Not started |
| Vulkan tensor runtime | q36 `q36_gpu.h`, `q36_vulkan.c`, shader/test infrastructure | EXTRACT/PORT | Not started |
| Adaptive expert cache | q36 adaptive cache v4 and multiclass designs/implementation | PORT/GENERALIZE | Not started |
| State codecs/checkpointing | q36 TurboQuant framework and KV store | PORT FRAMEWORK/REWRITE STATE | Not started |
| Benchmark, quality and capture | K3 harnesses plus q36 bench/quality/capture | MERGE METHODS | Not started |

## Required entry for every future port

Add one row before merging implementation code:

| Field | Required value |
|---|---|
| Target component | Exact target files and public interfaces |
| Owning source repository | `owner/repository` |
| Source ref | Full commit SHA; branch name may be included but is insufficient alone |
| Files reviewed | Exact paths, including tests and design documents |
| License review | License and NOTICE obligations |
| Upstream state | Relevant branches/PRs/commits checked on the review date |
| Decision | KEEP, PORT, REWRITE, EXPERIMENT, REJECT or DEFER |
| Local changes | Semantic differences from the source implementation |
| Verification | Tests, parity gates and hardware status |
| Reviewer/date | Person or agent and ISO date |

## Attribution rules

- Preserve Apache-2.0 notices for retained K3 files and vendored components.
- Preserve the MIT copyright and permission notice for copied or substantially
  derived q36 files.
- Add an SPDX identifier and a concise `Derived from ... @ <sha>` comment when a
  file contains a substantial source-derived implementation.
- Reimplemented ideas with no copied expression still require a provenance row,
  but not a misleading claim that the implementation is byte-derived.
- Generated fixtures must identify the generator and authoritative reference
  revision in their manifest.
- Model weights and generated packed artifacts are not source code and must not
  be committed unless their license and repository policy explicitly allow it.

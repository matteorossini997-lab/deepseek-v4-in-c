# DeepSeek V4 Flash Mini-Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic FP32 mini-oracle for DeepSeek V4 Flash that validates full and incremental execution without model weights.

**Architecture:** A standalone PyTorch package mirrors the released V4 operator order at reduced dimensions. Stateless full-sequence execution and explicit per-layer incremental caches share primitives but use independent data-flow paths, then a compact JSON oracle records diagnostic outputs.

**Tech Stack:** Python 3.10+, PyTorch, pytest, GitHub Actions CPU runner.

## Global Constraints

- No production checkpoint download.
- No dependency on Transformers, TileLang, Triton, or Safetensors.
- No native FP8/FP4 claim in this milestone.
- All fixture generation is deterministic and fail-closed.
- Existing C engine targets remain unchanged; Python tests are additive.

---

### Task 1: Configuration contract

**Files:**
- Create: `tools/dsv4_mini/config.py`
- Create: `tools/dsv4_mini/tiny_config.json`
- Test: `tests/python/test_dsv4_mini_config.py`

- [ ] Write tests for derived layer types and invalid dimensions.
- [ ] Run tests and verify missing-module failure.
- [ ] Implement immutable loader and validation.
- [ ] Run tests to green.

### Task 2: Routing and mHC primitives

**Files:**
- Create: `tools/dsv4_mini/reference.py`
- Create: `tools/dsv4_mini/initialization.py`
- Test: `tests/python/test_dsv4_mini_routing.py`
- Test: `tests/python/test_dsv4_mini_hyperconnection.py`

- [ ] Write tests for hash uniqueness, correction-bias selection, normalized route weights, and Sinkhorn row/column sums.
- [ ] Verify failures.
- [ ] Implement minimal primitives and adversarial initialization.
- [ ] Run tests to green.

### Task 3: Attention and compressor state

**Files:**
- Modify: `tools/dsv4_mini/reference.py`
- Test: `tests/python/test_dsv4_mini_attention.py`

- [ ] Write tests for sliding-window boundaries and rate-4/rate-128 emission boundaries.
- [ ] Verify failures.
- [ ] Implement stateless and incremental compressor paths plus shared-KV attention.
- [ ] Run tests to green.

### Task 4: Full model, MTP, and parity

**Files:**
- Modify: `tools/dsv4_mini/reference.py`
- Modify: `tools/dsv4_mini/initialization.py`
- Test: `tests/python/test_dsv4_mini_incremental.py`

- [ ] Write full/incremental parity tests through position 129.
- [ ] Verify failures.
- [ ] Implement decoder stack, head, MTP block, and explicit decode state.
- [ ] Run tests to green.

### Task 5: Oracle and CI integration

**Files:**
- Create: `tools/dsv4_mini/oracle.py`
- Create: `tools/dsv4_mini/make_oracle.py`
- Create: `tools/dsv4_mini/README.md`
- Create: `tests/python/test_dsv4_mini_oracle.py`
- Create: `.github/workflows/dsv4-mini.yml`
- Modify: `Makefile`
- Modify: `.gitignore`

- [ ] Write oracle schema/determinism tests.
- [ ] Verify failures.
- [ ] Implement compact trace/checksum generation and CLI.
- [ ] Generate canonical CPU oracle.
- [ ] Run the complete Python gate and compileall.
- [ ] Commit and publish a draft PR when repository write access is available.

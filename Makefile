# Kimi K3 inference engine.
#
#   make                build the engine (bin/k3)
#   make test           run every test that needs no model weights
#   make bench          kernel microbenchmarks
#   make portable       build without -march=native (for distribution)
#   make debug          -O0 -g with assertions
#   make asan / ubsan   sanitizer builds
#   make format         clang-format the tree
#   make clean
#
# Nothing here requires a checkpoint. `make test` is the gate that must stay green.

# ---------------------------------------------------------------------------- config --
CC       ?= cc
PYTHON   ?= python3
BUILD    ?= build
BIN      ?= bin
PREFIX   ?= /usr/local

# -march=native is a real win on the expert matmuls but produces a binary that will not
# run on an older CPU. `make portable` drops it.
ARCH     ?= -march=native

# -Wpointer-arith is not cosmetic: weight pointers are `const void *`, and arithmetic on
# a void pointer is a silent GNU extension that strides by ONE BYTE. Without this flag
# that mistake compiles clean under -Wall -Wextra and returns the wrong tensor.
#
# -ffp-contract=off keeps floating-point results reproducible across compilers by
# disabling automatic FMA contraction. The test-suite compares against a reference to a
# fixed tolerance; letting the compiler fuse changes results by more than that.
WARN     := -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter
CFLAGS   ?= -O3 -std=gnu99 $(WARN) $(ARCH) -fopenmp -ffp-contract=off
LDFLAGS  ?= -lm -fopenmp

# Flat include search across the module dirs: sources use "k3.h", "k3_cache.h" etc
# rather than path-qualified includes, which keeps them relocatable.
INCLUDES := -Iinclude -Iinclude/k3 -Ithird_party \
            -Isrc/core -Isrc/io -Isrc/cache -Isrc/model -Isrc/tokenizer

# ----------------------------------------------------------------------------- files --
ENGINE_SRC := src/core/k3_ops.c \
              src/io/k3_st.c src/io/k3_load.c src/io/k3_trunk.c \
              src/cache/k3_cache.c \
              src/model/k3_bind.c
ENGINE_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(ENGINE_SRC))

CLI_SRC    := src/cli/k3_run.c
CLI_BIN    := $(BIN)/k3

# Tests that need no checkpoint. These run in CI on every push.
UNIT_TESTS := test_ops test_cache test_st test_cfg test_tok test_dsv4_config test_dsv4_inventory test_dsv4_cpu_ops test_dsv4_attention_state test_dsv4_attention_numeric test_dsv4_sparse_index test_dsv4_attention_step test_dsv4_attention_runtime test_dsv4_moe scale_test k3_model
# Tests that need real shards. Built and run by `make test-all` with SHARD_DIR set;
# see the weights-test target below.
WEIGHT_TESTS := test_expert test_real_layer

TEST_BINS  := $(addprefix $(BIN)/,$(UNIT_TESTS))
WEIGHT_BINS := $(addprefix $(BIN)/,$(WEIGHT_TESTS))

FIXTURES   ?= tests/fixtures
TOK_FILES  ?= $(HOME)/k3model

# The safetensors test rebuilds an index and writes it to $(BUILD) rather than /tmp, so
# two concurrent `make test` runs cannot race on one filename and `make clean` removes it.

# ---------------------------------------------------------------------------- targets --
.PHONY: all test test-all bench portable debug asan ubsan format clean install help \
        tok cfg ops cache st oracle weights-test test-dsv4-config dsv4-inspect \
        test-dsv4-mini dsv4-mini-oracle test-dsv4-inventory dsv4-inventory \
        test-dsv4-cpu-ops test-dsv4-attention-state \
        test-dsv4-attention-numeric test-dsv4-sparse-index \
        test-dsv4-attention-step test-dsv4-attention-runtime \
        test-dsv4-moe

all: $(CLI_BIN)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CLI_BIN): $(CLI_SRC) $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(CLI_SRC) $(ENGINE_OBJ) -o $@ $(LDFLAGS)

$(BIN):
	@mkdir -p $(BIN)

# Each test links only what it needs, so a failure points at one subsystem.
$(BIN)/test_ops: tests/unit/test_ops.c $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_cache: tests/unit/test_cache.c $(BUILD)/src/cache/k3_cache.o \
                   $(BUILD)/src/io/k3_load.o $(BUILD)/src/io/k3_st.o \
                   $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_st: tests/unit/test_st.c $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# The DeepSeek config contract is portable C99 and independent of the K3 engine.
$(BIN)/test_dsv4_config: tests/unit/test_dsv4_config.c src/model/dsv4_config.c include/dsv4/dsv4_config.h third_party/json.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function -Iinclude/dsv4 -Ithird_party tests/unit/test_dsv4_config.c src/model/dsv4_config.c -o $@ -lm

$(BIN)/dsv4-inspect: src/cli/dsv4_inspect.c src/model/dsv4_config.c include/dsv4/dsv4_config.h third_party/json.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function -Iinclude/dsv4 -Ithird_party src/cli/dsv4_inspect.c src/model/dsv4_config.c -o $@ -lm


$(BIN)/test_dsv4_inventory: tests/unit/test_dsv4_inventory.c src/io/dsv4_inventory.c include/dsv4/dsv4_inventory.h include/dsv4/dsv4_config.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -Iinclude/dsv4 tests/unit/test_dsv4_inventory.c src/io/dsv4_inventory.c -o $@ -lm

$(BIN)/dsv4-inventory: src/cli/dsv4_inventory_cli.c src/io/dsv4_inventory.c src/model/dsv4_config.c include/dsv4/dsv4_inventory.h include/dsv4/dsv4_config.h third_party/json.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -Iinclude/dsv4 -Ithird_party src/cli/dsv4_inventory_cli.c src/io/dsv4_inventory.c src/model/dsv4_config.c -o $@ -lm

$(BIN)/test_dsv4_cpu_ops: tests/unit/test_dsv4_cpu_ops.c src/cpu/dsv4_cpu_ops.c include/dsv4/dsv4_cpu_ops.h tests/fixtures/dsv4_cpu_ops_vectors.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 -Itests/fixtures tests/unit/test_dsv4_cpu_ops.c src/cpu/dsv4_cpu_ops.c -o $@ -lm

$(BIN)/test_dsv4_attention_state: tests/unit/test_dsv4_attention_state.c src/cpu/dsv4_attention_state.c include/dsv4/dsv4_attention_state.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 tests/unit/test_dsv4_attention_state.c src/cpu/dsv4_attention_state.c -o $@ -lm

$(BIN)/test_dsv4_attention_numeric: tests/unit/test_dsv4_attention_numeric.c src/cpu/dsv4_attention_numeric.c include/dsv4/dsv4_attention_numeric.h tests/fixtures/dsv4_attention_numeric_vectors.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 -Itests/fixtures tests/unit/test_dsv4_attention_numeric.c src/cpu/dsv4_attention_numeric.c -o $@ -lm

$(BIN)/test_dsv4_sparse_index: tests/unit/test_dsv4_sparse_index.c src/cpu/dsv4_sparse_index.c include/dsv4/dsv4_sparse_index.h tests/fixtures/dsv4_sparse_index_vectors.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 -Itests/fixtures tests/unit/test_dsv4_sparse_index.c src/cpu/dsv4_sparse_index.c -o $@ -lm

$(BIN)/test_dsv4_attention_step: tests/unit/test_dsv4_attention_step.c src/cpu/dsv4_attention_step.c src/cpu/dsv4_attention_numeric.c src/cpu/dsv4_attention_state.c src/cpu/dsv4_sparse_index.c include/dsv4/dsv4_attention_step.h include/dsv4/dsv4_attention_numeric.h include/dsv4/dsv4_attention_state.h include/dsv4/dsv4_sparse_index.h tests/fixtures/dsv4_attention_step_vectors.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 -Itests/fixtures tests/unit/test_dsv4_attention_step.c src/cpu/dsv4_attention_step.c src/cpu/dsv4_attention_numeric.c src/cpu/dsv4_attention_state.c src/cpu/dsv4_sparse_index.c -o $@ -lm

$(BIN)/test_dsv4_attention_runtime: tests/unit/test_dsv4_attention_runtime.c src/cpu/dsv4_attention_runtime.c src/cpu/dsv4_attention_step.c src/cpu/dsv4_attention_numeric.c src/cpu/dsv4_attention_state.c src/cpu/dsv4_sparse_index.c include/dsv4/dsv4_attention_runtime.h include/dsv4/dsv4_attention_step.h include/dsv4/dsv4_attention_numeric.h include/dsv4/dsv4_attention_state.h include/dsv4/dsv4_sparse_index.h tests/fixtures/dsv4_attention_runtime_vectors.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 -Itests/fixtures tests/unit/test_dsv4_attention_runtime.c src/cpu/dsv4_attention_runtime.c src/cpu/dsv4_attention_step.c src/cpu/dsv4_attention_numeric.c src/cpu/dsv4_attention_state.c src/cpu/dsv4_sparse_index.c -o $@ -lm

$(BIN)/test_dsv4_moe: tests/unit/test_dsv4_moe.c src/cpu/dsv4_moe.c src/cpu/dsv4_cpu_ops.c include/dsv4/dsv4_moe.h include/dsv4/dsv4_cpu_ops.h tests/fixtures/dsv4_moe_vectors.h | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 -Itests/fixtures tests/unit/test_dsv4_moe.c src/cpu/dsv4_moe.c src/cpu/dsv4_cpu_ops.c -o $@ -lm

# The tokenizer and config reader are portable C99 with no OpenMP and no platform calls,
# so they build and are verifiable on any machine, including one with no checkpoint.
$(BIN)/test_tok: tests/unit/test_tok.c | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function $(INCLUDES) $< -o $@

$(BIN)/test_cfg: tests/unit/test_cfg.c src/core/k3_ops.c | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function $(INCLUDES) $^ -o $@ -lm

# Allocates at REAL model widths (a ~1.8 GB KDA layer), so it needs the optimised build
# rather than the portable C99 one the tokenizer and config tests use.
$(BIN)/scale_test: tests/unit/scale_test.c $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/k3_model: tests/unit/k3_model.c $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/bench_kernels: benchmarks/bench_kernels.c $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

dsv4-inspect: $(BIN)/dsv4-inspect

dsv4-inventory: $(BIN)/dsv4-inventory

test-dsv4-config: $(BIN)/test_dsv4_config $(BIN)/dsv4-inspect
	$(BIN)/test_dsv4_config hf $(FIXTURES)/dsv4/config_hf.json
	$(BIN)/test_dsv4_config inference $(FIXTURES)/dsv4/config_inference.json
	$(BIN)/test_dsv4_config reject $(FIXTURES)/dsv4/config_missing.json num_hidden_layers
	$(BIN)/test_dsv4_config reject $(FIXTURES)/dsv4/config_conflict.json conflict
	$(BIN)/test_dsv4_config reject $(FIXTURES)/dsv4/config_bad_ratios.json compress_ratios
	$(BIN)/test_dsv4_config reject $(FIXTURES)/dsv4/config_bad_topk.json n_activated_experts
	$(BIN)/test_dsv4_config reject $(FIXTURES)/dsv4/config_bad_format.json expert_dtype
	$(BIN)/dsv4-inspect --json $(FIXTURES)/dsv4/config_hf.json | grep -F '"profile":"deepseek-v4-flash"'
	$(BIN)/dsv4-inspect --json $(FIXTURES)/dsv4/config_inference.json | grep -F '"layout":"inference"'


test-dsv4-inventory: $(BIN)/test_dsv4_inventory $(BIN)/dsv4-inventory
	$(BIN)/test_dsv4_inventory
	$(PYTHON) tools/run_dsv4_inventory_gate.py \
		--generator tools/make_dsv4_inventory_fixture.py \
		--cli $(BIN)/dsv4-inventory \
		--config $(FIXTURES)/dsv4/config_hf.json \
		--verifier tools/verify_dsv4_inventory.py \
		--workdir $(BUILD)/dsv4-inventory-gate

test-dsv4-cpu-ops: $(BIN)/test_dsv4_cpu_ops
	$(BIN)/test_dsv4_cpu_ops

test-dsv4-attention-state: $(BIN)/test_dsv4_attention_state
	$(BIN)/test_dsv4_attention_state

test-dsv4-attention-numeric: $(BIN)/test_dsv4_attention_numeric
	$(BIN)/test_dsv4_attention_numeric

test-dsv4-sparse-index: $(BIN)/test_dsv4_sparse_index
	$(BIN)/test_dsv4_sparse_index

test-dsv4-attention-step: $(BIN)/test_dsv4_attention_step
	$(BIN)/test_dsv4_attention_step

test-dsv4-attention-runtime: $(BIN)/test_dsv4_attention_runtime
	$(BIN)/test_dsv4_attention_runtime

test-dsv4-moe: $(BIN)/test_dsv4_moe
	$(BIN)/test_dsv4_moe

## test: everything that needs no model weights
test: $(TEST_BINS) $(BIN)/dsv4-inspect $(BIN)/dsv4-inventory
	@echo "== op kernels ==";        ./$(BIN)/test_ops $(FIXTURES)/ops
	@echo "== streaming cache ==";   ./$(BIN)/test_cache $(FIXTURES)/cache
	@echo "== safetensors ==";       ./$(BIN)/test_st $(FIXTURES)/st $(BUILD)/st_index.json \
	    plain.f32.2d plain.bf16.1d tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32
	@echo "== config reader ==";     ./$(BIN)/test_cfg fixture $(FIXTURES)/ref_k3.json
	@echo "== DeepSeek config ==";    $(MAKE) --no-print-directory test-dsv4-config
	@echo "== DeepSeek inventory =="; $(MAKE) --no-print-directory test-dsv4-inventory
	@echo "== DeepSeek CPU ops ==";   $(MAKE) --no-print-directory test-dsv4-cpu-ops
	@echo "== DeepSeek attn state =="; $(MAKE) --no-print-directory test-dsv4-attention-state
	@echo "== DeepSeek attn numeric =="; $(MAKE) --no-print-directory test-dsv4-attention-numeric
	@echo "== DeepSeek sparse index =="; $(MAKE) --no-print-directory test-dsv4-sparse-index
	@echo "== DeepSeek attention step =="; $(MAKE) --no-print-directory test-dsv4-attention-step
	@echo "== DeepSeek attention runtime =="; $(MAKE) --no-print-directory test-dsv4-attention-runtime
	@echo "== DeepSeek MoE =="; $(MAKE) --no-print-directory test-dsv4-moe
	@echo "== tokenizer ==";         ./$(BIN)/test_tok $(TOK_FILES) roundtrip src/core/k3_ops.c \
	    || echo "  (skipped: no tokenizer files at $(TOK_FILES))"
	@echo "== real dimensions ==";   ./$(BIN)/scale_test
	@echo "== full-model oracle =="; ./$(BIN)/k3_model $(FIXTURES)
	@echo; echo "ALL WEIGHTLESS TESTS PASSED"

## test-all: adds tests that need a real checkpoint (set SHARD_DIR)
test-all: test
	@test -n "$(SHARD_DIR)" || { echo "set SHARD_DIR=/path/to/shards"; exit 2; }
	$(MAKE) weights-test SHARD_DIR=$(SHARD_DIR)

weights-test: $(WEIGHT_BINS)
	./$(BIN)/test_expert $(SHARD_DIR) 1 64
	./$(BIN)/test_real_layer $(SHARD_DIR) 1 4 8

$(BIN)/test_expert: tests/unit/test_expert.c $(BUILD)/src/io/k3_load.o \
                    $(BUILD)/src/io/k3_st.o $(BUILD)/src/core/k3_ops.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_real_layer: tests/unit/test_real_layer.c $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

## tok: tokenizer parity against the reference implementation
tok: $(BIN)/test_tok
	@$(PYTHON) tools/tok_parity.py ./$(BIN)/test_tok

## cfg: config reader against both supported config layouts
cfg: $(BIN)/test_cfg
	@./$(BIN)/test_cfg fixture $(FIXTURES)/ref_k3.json
	@test -f "$(TOK_FILES)/config.json" && ./$(BIN)/test_cfg real $(TOK_FILES)/config.json \
	    || echo "  (skipped real config: none at $(TOK_FILES))"

## bench: kernel microbenchmarks, no weights required
bench: $(BIN)/bench_kernels
	./$(BIN)/bench_kernels

## test-dsv4-mini: deterministic FP32 DeepSeek V4 mini-oracle tests (requires PyTorch + pytest)
test-dsv4-mini:
	$(PYTHON) tools/dsv4_mini/run_tests.py

## dsv4-mini-oracle: regenerate the canonical 130-token CPU oracle
dsv4-mini-oracle:
	$(PYTHON) -m tools.dsv4_mini.make_oracle \
		--config tools/dsv4_mini/tiny_config.json \
		--output tests/fixtures/dsv4-mini/oracle.json \
		--device cpu \
		--sequence-length 130

## portable: no -march=native, runs on any x86-64
portable:
	$(MAKE) ARCH="-mavx2 -mfma" all

## debug: -O0 -g, assertions on
debug:
	$(MAKE) CFLAGS="-O0 -g3 -std=gnu99 $(WARN) -fopenmp -ffp-contract=off" all

asan:
	$(MAKE) CFLAGS="-O1 -g -std=gnu99 $(WARN) -fsanitize=address,undefined -fno-omit-frame-pointer" \
	        LDFLAGS="-lm -fsanitize=address,undefined" ARCH= all

ubsan:
	$(MAKE) CFLAGS="-O1 -g -std=gnu99 $(WARN) -fsanitize=undefined" \
	        LDFLAGS="-lm -fsanitize=undefined" ARCH= all

format:
	@command -v clang-format >/dev/null || { echo "clang-format not installed"; exit 1; }
	clang-format -i $(shell find src include tests benchmarks -name '*.c' -o -name '*.h' 2>/dev/null)

# Installs the binary AND the public headers, matching CMake's install rules exactly
# the two build systems are documented as interchangeable, so they must stay so.
# third_party/json.h goes with them: k3_cfg.h includes it and exposes jval in its
# signatures, so an installed k3_cfg.h without it does not compile.
install: $(CLI_BIN)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(CLI_BIN) $(DESTDIR)$(PREFIX)/bin/k3
	install -d $(DESTDIR)$(PREFIX)/include/k3
	install -m 644 include/k3/*.h $(DESTDIR)$(PREFIX)/include/k3/
	install -m 644 third_party/json.h $(DESTDIR)$(PREFIX)/include/k3/

clean:
	rm -rf $(BUILD) $(BIN)

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'

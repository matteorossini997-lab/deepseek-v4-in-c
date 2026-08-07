#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    make_path = Path('Makefile')
    make = make_path.read_text(encoding='utf-8')
    make = replace_once(
        make,
        'test_dsv4_model test_dsv4_mtp scale_test k3_model',
        'test_dsv4_model test_dsv4_mtp test_dsv4_full_model scale_test k3_model',
        'Make UNIT_TESTS',
    )
    make = replace_once(
        make,
        '        test-dsv4-moe test-dsv4-decoder-layer test-dsv4-model test-dsv4-mtp\n',
        '        test-dsv4-moe test-dsv4-decoder-layer test-dsv4-model test-dsv4-mtp test-dsv4-full-model\n',
        'Make PHONY',
    )
    mtp_rule = '''$(BIN)/test_dsv4_mtp: tests/unit/test_dsv4_mtp.c src/cpu/dsv4_mtp.c src/cpu/dsv4_decoder_layer.c src/cpu/dsv4_attention_runtime.c src/cpu/dsv4_attention_step.c src/cpu/dsv4_attention_numeric.c src/cpu/dsv4_attention_state.c src/cpu/dsv4_sparse_index.c src/cpu/dsv4_moe.c src/cpu/dsv4_cpu_ops.c include/dsv4/dsv4_mtp.h include/dsv4/dsv4_decoder_layer.h include/dsv4/dsv4_attention_runtime.h include/dsv4/dsv4_attention_step.h include/dsv4/dsv4_attention_numeric.h include/dsv4/dsv4_attention_state.h include/dsv4/dsv4_sparse_index.h include/dsv4/dsv4_moe.h include/dsv4/dsv4_cpu_ops.h tests/fixtures/dsv4_mtp_vectors.h tests/fixtures/dsv4_model_shell_vectors.h | $(BIN)\n\t$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 -Itests/fixtures tests/unit/test_dsv4_mtp.c src/cpu/dsv4_mtp.c src/cpu/dsv4_decoder_layer.c src/cpu/dsv4_attention_runtime.c src/cpu/dsv4_attention_step.c src/cpu/dsv4_attention_numeric.c src/cpu/dsv4_attention_state.c src/cpu/dsv4_sparse_index.c src/cpu/dsv4_moe.c src/cpu/dsv4_cpu_ops.c -o $@ -lm\n'''
    full_rule = '''\n$(BIN)/test_dsv4_full_model: tests/unit/test_dsv4_full_model.c tests/unit/dsv4_full_fixture_builders.h src/cpu/dsv4_full_model.c src/cpu/dsv4_model.c src/cpu/dsv4_mtp.c src/cpu/dsv4_decoder_layer.c src/cpu/dsv4_attention_runtime.c src/cpu/dsv4_attention_step.c src/cpu/dsv4_attention_numeric.c src/cpu/dsv4_attention_state.c src/cpu/dsv4_sparse_index.c src/cpu/dsv4_moe.c src/cpu/dsv4_cpu_ops.c include/dsv4/dsv4_full_model.h include/dsv4/dsv4_model.h include/dsv4/dsv4_mtp.h tests/fixtures/dsv4_full_runtime_vectors.h tests/fixtures/dsv4_model_shell_vectors.h tests/fixtures/dsv4_mtp_vectors.h | $(BIN)\n\t$(CC) -O2 -std=c99 $(WARN) -Werror -ffp-contract=off -Iinclude/dsv4 -Itests/fixtures -Itests/unit tests/unit/test_dsv4_full_model.c src/cpu/dsv4_full_model.c src/cpu/dsv4_model.c src/cpu/dsv4_mtp.c src/cpu/dsv4_decoder_layer.c src/cpu/dsv4_attention_runtime.c src/cpu/dsv4_attention_step.c src/cpu/dsv4_attention_numeric.c src/cpu/dsv4_attention_state.c src/cpu/dsv4_sparse_index.c src/cpu/dsv4_moe.c src/cpu/dsv4_cpu_ops.c -o $@ -lm\n'''
    make = replace_once(make, mtp_rule, mtp_rule + full_rule, 'Make full-model rule')
    make = replace_once(
        make,
        'test-dsv4-mtp: $(BIN)/test_dsv4_mtp\n\t$(BIN)/test_dsv4_mtp\n',
        'test-dsv4-mtp: $(BIN)/test_dsv4_mtp\n\t$(BIN)/test_dsv4_mtp\n\n'
        'test-dsv4-full-model: $(BIN)/test_dsv4_full_model\n\t$(BIN)/test_dsv4_full_model\n',
        'Make full-model target',
    )
    make = replace_once(
        make,
        '\t@echo "== DeepSeek MTP =="; $(MAKE) --no-print-directory test-dsv4-mtp\n',
        '\t@echo "== DeepSeek MTP =="; $(MAKE) --no-print-directory test-dsv4-mtp\n'
        '\t@echo "== DeepSeek full runtime =="; $(MAKE) --no-print-directory test-dsv4-full-model\n',
        'Make weightless full runtime',
    )
    make_path.write_text(make, encoding='utf-8')

    cmake_path = Path('CMakeLists.txt')
    cmake = cmake_path.read_text(encoding='utf-8')
    mtp_block = '''add_library(dsv4_mtp STATIC src/cpu/dsv4_mtp.c)\ntarget_include_directories(dsv4_mtp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)\ntarget_link_libraries(dsv4_mtp PUBLIC m dsv4_decoder_layer PRIVATE k3_flags)\n\nadd_executable(test_dsv4_mtp tests/unit/test_dsv4_mtp.c)\ntarget_include_directories(test_dsv4_mtp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)\ntarget_link_libraries(test_dsv4_mtp PRIVATE dsv4_mtp)\n'''
    full_block = '''\nadd_library(dsv4_full_model STATIC src/cpu/dsv4_full_model.c)\ntarget_include_directories(dsv4_full_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)\ntarget_link_libraries(dsv4_full_model PUBLIC m dsv4_model dsv4_mtp PRIVATE k3_flags)\n\nadd_executable(test_dsv4_full_model tests/unit/test_dsv4_full_model.c)\ntarget_include_directories(test_dsv4_full_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit)\ntarget_link_libraries(test_dsv4_full_model PRIVATE dsv4_full_model)\n'''
    cmake = replace_once(cmake, mtp_block, mtp_block + full_block, 'CMake full runtime')
    cmake = replace_once(
        cmake,
        'add_test(NAME dsv4_mtp COMMAND test_dsv4_mtp)\n',
        'add_test(NAME dsv4_mtp COMMAND test_dsv4_mtp)\nadd_test(NAME dsv4_full_model COMMAND test_dsv4_full_model)\n',
        'CMake full CTest',
    )
    cmake_path.write_text(cmake, encoding='utf-8')

    status_path = Path('docs/PROJECT_STATUS.md')
    status = status_path.read_text(encoding='utf-8')
    note = '''\n### P1-K full runtime + greedy generation\n\nThe FP32 diagnostic runtime now stages the P1-I base model and P1-J MTP as one token transaction: both runtimes are deep-cloned, base executes first, MTP consumes the staged base streams, and neither state commits unless both succeed. Greedy generation resets the combined runtime, prefills the prompt, and chooses every standard generated token from base-model logits only; MTP is still evaluated and validated but does not select the token.\n'''
    if '### P1-K full runtime + greedy generation' not in status:
        status = status.rstrip() + '\n' + note
    status_path.write_text(status, encoding='utf-8')

    provenance_path = Path('docs/PROVENANCE.md')
    provenance = provenance_path.read_text(encoding='utf-8')
    entry = '''\n## P1-K full runtime and generation source review\n\n**Review date:** 2026-08-07\n\n- Target base: `p1j/native-mtp@b4355d29a2e410ca4555ae8e129b3983f3f07d97`.\n- Mini-oracle: `MiniReferenceModel.step` and `MiniReferenceModel.generate`; successful combined sequencing and greedy base-logit argmax are ported.\n- Official DeepSeek V4 Flash: base `Transformer.forward` remains distinct from MTP; the reviewed standard `inference/generate.py` at source commit `216d76676e09d4e0aa3f73784e9ad53a6d0b2735` chooses tokens from base-model logits. P1-K keeps that ownership and uses deterministic greedy argmax for the current CPU reference milestone.\n- q36 `feat/dsv4-mini-oracle@d6684ca33d4da496cd2e805ce159c2db7892f98a` remains design-only.\n\nDecision: **KEEP** P1-I/P1-J numerics; **PORT** successful combined step/generation semantics; **REWRITE** ownership into a stronger C transaction that commits neither base nor MTP after a late failure. No official source code or checkpoint bytes are copied.\n'''
    if '## P1-K full runtime and generation source review' not in provenance:
        provenance = provenance.rstrip() + '\n' + entry
    provenance_path.write_text(provenance, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

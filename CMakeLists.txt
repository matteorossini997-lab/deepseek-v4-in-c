# Kimi K3 inference engine.
#
# The Makefile is the reference build and what CI uses. This exists for IDE integration
# and for platforms where GNU make is awkward. Both produce the same binary.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure

cmake_minimum_required(VERSION 3.16)

project(kimi-k3-in-c
        VERSION 0.1.0
        DESCRIPTION "Kimi K3 inference engine in portable C"
        LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -march=native gives a real speedup on the expert matmuls but produces a binary tied to
# the build machine. Off by default so a packaged build is portable.
option(K3_NATIVE_ARCH "Optimise for the building CPU (-march=native)" OFF)
option(K3_SANITIZE    "Build with ASan + UBSan"                        OFF)

find_package(OpenMP)

add_library(k3_flags INTERFACE)
target_compile_options(k3_flags INTERFACE
  -Wall -Wextra
  # Not cosmetic: weight pointers are `const void *`, and pointer arithmetic on void is a
  # GNU extension that strides by one byte. Without this the mistake compiles silently.
  -Wpointer-arith
  -Wshadow -Wvla
  -Wno-unused-parameter
  # Keeps floating-point results reproducible across compilers by disabling automatic FMA
  # contraction; the op tests compare against a reference at a fixed tolerance.
  -ffp-contract=off)

if(K3_NATIVE_ARCH)
  target_compile_options(k3_flags INTERFACE -march=native)
else()
  # Documented baseline. The engine assumes AVX2 + FMA are available.
  target_compile_options(k3_flags INTERFACE -mavx2 -mfma)
endif()

if(K3_SANITIZE)
  target_compile_options(k3_flags INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(k3_flags INTERFACE -fsanitize=address,undefined)
endif()

# --------------------------------------------------------------------------- engine --
add_library(k3 STATIC
  src/core/k3_ops.c
  src/io/k3_st.c
  src/io/k3_load.c
  src/io/k3_trunk.c
  src/cache/k3_cache.c
  src/model/k3_bind.c)

target_include_directories(k3 PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/k3
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/io
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cache
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer)

target_link_libraries(k3 PUBLIC k3_flags m)
if(OpenMP_C_FOUND)
  target_link_libraries(k3 PUBLIC OpenMP::OpenMP_C)
endif()

# ------------------------------------------------------------------------------ cli --
add_executable(k3_cli src/cli/k3_run.c)
set_target_properties(k3_cli PROPERTIES OUTPUT_NAME k3)
target_link_libraries(k3_cli PRIVATE k3)

# ---------------------------------------------------------------------------- tests --
# Every test below runs WITHOUT model weights. The checkpoint is 1.56 TB; if correctness
# needed it, correctness would go unchecked.
enable_testing()

function(k3_add_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE k3)
endfunction()

k3_add_test(test_ops    tests/unit/test_ops.c)
k3_add_test(test_cache  tests/unit/test_cache.c)
k3_add_test(test_st     tests/unit/test_st.c)
k3_add_test(test_cfg    tests/unit/test_cfg.c)

add_library(dsv4_config STATIC src/model/dsv4_config.c)
target_include_directories(dsv4_config
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(dsv4_config PUBLIC m PRIVATE k3_flags)

add_executable(test_dsv4_config tests/unit/test_dsv4_config.c)
target_link_libraries(test_dsv4_config PRIVATE dsv4_config)

add_executable(dsv4-inspect src/cli/dsv4_inspect.c)
target_link_libraries(dsv4-inspect PRIVATE dsv4_config)


add_library(dsv4_inventory STATIC src/io/dsv4_inventory.c)
target_include_directories(dsv4_inventory PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_inventory PUBLIC m PRIVATE k3_flags)

add_executable(test_dsv4_inventory tests/unit/test_dsv4_inventory.c)
target_link_libraries(test_dsv4_inventory PRIVATE dsv4_inventory)

add_executable(dsv4-inventory src/cli/dsv4_inventory_cli.c)
target_link_libraries(dsv4-inventory PRIVATE dsv4_inventory dsv4_config)

add_library(dsv4_cpu_ops STATIC src/cpu/dsv4_cpu_ops.c)
target_include_directories(dsv4_cpu_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_cpu_ops PUBLIC m PRIVATE k3_flags)

add_executable(test_dsv4_cpu_ops tests/unit/test_dsv4_cpu_ops.c)
target_include_directories(test_dsv4_cpu_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
target_link_libraries(test_dsv4_cpu_ops PRIVATE dsv4_cpu_ops)

add_library(dsv4_attention_state STATIC src/cpu/dsv4_attention_state.c)
target_include_directories(dsv4_attention_state PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_attention_state PUBLIC m PRIVATE k3_flags)

add_executable(test_dsv4_attention_state tests/unit/test_dsv4_attention_state.c)
target_link_libraries(test_dsv4_attention_state PRIVATE dsv4_attention_state)

add_library(dsv4_attention_numeric STATIC src/cpu/dsv4_attention_numeric.c)
target_include_directories(dsv4_attention_numeric PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_attention_numeric PUBLIC m PRIVATE k3_flags)

add_executable(test_dsv4_attention_numeric tests/unit/test_dsv4_attention_numeric.c)
target_include_directories(test_dsv4_attention_numeric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
target_link_libraries(test_dsv4_attention_numeric PRIVATE dsv4_attention_numeric)

add_library(dsv4_sparse_index STATIC src/cpu/dsv4_sparse_index.c)
target_include_directories(dsv4_sparse_index PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_sparse_index PUBLIC m PRIVATE k3_flags)

add_executable(test_dsv4_sparse_index tests/unit/test_dsv4_sparse_index.c)
target_include_directories(test_dsv4_sparse_index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
target_link_libraries(test_dsv4_sparse_index PRIVATE dsv4_sparse_index)

add_library(dsv4_attention_step STATIC src/cpu/dsv4_attention_step.c)
target_include_directories(dsv4_attention_step PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_attention_step PUBLIC m dsv4_attention_numeric dsv4_attention_state dsv4_sparse_index PRIVATE k3_flags)

add_executable(test_dsv4_attention_step tests/unit/test_dsv4_attention_step.c)
target_include_directories(test_dsv4_attention_step PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
target_link_libraries(test_dsv4_attention_step PRIVATE dsv4_attention_step)

add_library(dsv4_attention_runtime STATIC src/cpu/dsv4_attention_runtime.c)
target_include_directories(dsv4_attention_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_attention_runtime PUBLIC m dsv4_attention_step dsv4_attention_numeric PRIVATE k3_flags)

add_executable(test_dsv4_attention_runtime tests/unit/test_dsv4_attention_runtime.c)
target_include_directories(test_dsv4_attention_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
target_link_libraries(test_dsv4_attention_runtime PRIVATE dsv4_attention_runtime)

add_library(dsv4_moe STATIC src/cpu/dsv4_moe.c)
target_include_directories(dsv4_moe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_moe PUBLIC m dsv4_cpu_ops PRIVATE k3_flags)

add_executable(test_dsv4_moe tests/unit/test_dsv4_moe.c)
target_include_directories(test_dsv4_moe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
target_link_libraries(test_dsv4_moe PRIVATE dsv4_moe)

add_library(dsv4_decoder_layer STATIC src/cpu/dsv4_decoder_layer.c)
target_include_directories(dsv4_decoder_layer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_decoder_layer PUBLIC m dsv4_attention_runtime dsv4_moe dsv4_cpu_ops PRIVATE k3_flags)

add_executable(test_dsv4_decoder_layer tests/unit/test_dsv4_decoder_layer.c)
target_include_directories(test_dsv4_decoder_layer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
target_link_libraries(test_dsv4_decoder_layer PRIVATE dsv4_decoder_layer)

add_library(dsv4_model STATIC src/cpu/dsv4_model.c)
target_include_directories(dsv4_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/dsv4)
target_link_libraries(dsv4_model PUBLIC m dsv4_decoder_layer PRIVATE k3_flags)

add_executable(test_dsv4_model tests/unit/test_dsv4_model.c)
target_include_directories(test_dsv4_model PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
target_link_libraries(test_dsv4_model PRIVATE dsv4_model)

k3_add_test(test_tok    tests/unit/test_tok.c)
k3_add_test(scale_test  tests/unit/scale_test.c)
k3_add_test(k3_model    tests/unit/k3_model.c)

set(FIX ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)

# Every test the Makefile's `test` target runs is registered here too. The two build
# systems are documented as interchangeable, so a test present in one and absent from
# the other silently gives ctest less coverage than `make test`.
add_test(NAME ops        COMMAND test_ops   ${FIX}/ops)
add_test(NAME cache      COMMAND test_cache ${FIX}/cache)
add_test(NAME config     COMMAND test_cfg   fixture ${FIX}/ref_k3.json)

set(DSV4_FIX ${FIX}/dsv4)
add_test(NAME dsv4_config_hf COMMAND test_dsv4_config hf ${DSV4_FIX}/config_hf.json)
add_test(NAME dsv4_config_inference COMMAND test_dsv4_config inference ${DSV4_FIX}/config_inference.json)
add_test(NAME dsv4_config_missing COMMAND test_dsv4_config reject ${DSV4_FIX}/config_missing.json num_hidden_layers)
add_test(NAME dsv4_config_conflict COMMAND test_dsv4_config reject ${DSV4_FIX}/config_conflict.json conflict)
add_test(NAME dsv4_config_bad_ratios COMMAND test_dsv4_config reject ${DSV4_FIX}/config_bad_ratios.json compress_ratios)
add_test(NAME dsv4_config_bad_topk COMMAND test_dsv4_config reject ${DSV4_FIX}/config_bad_topk.json n_activated_experts)
add_test(NAME dsv4_config_bad_format COMMAND test_dsv4_config reject ${DSV4_FIX}/config_bad_format.json expert_dtype)
add_test(NAME dsv4_inspect_hf COMMAND dsv4-inspect --json ${DSV4_FIX}/config_hf.json)
set_tests_properties(dsv4_inspect_hf PROPERTIES PASS_REGULAR_EXPRESSION "\"layout\":\"hf\"")
add_test(NAME dsv4_inspect_inference COMMAND dsv4-inspect --json ${DSV4_FIX}/config_inference.json)
set_tests_properties(dsv4_inspect_inference PROPERTIES PASS_REGULAR_EXPRESSION "\"layout\":\"inference\"")



find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME dsv4_inventory_unit COMMAND test_dsv4_inventory)
add_test(
  NAME dsv4_inventory_gate
  COMMAND ${Python3_EXECUTABLE}
${CMAKE_CURRENT_SOURCE_DIR}/tools/run_dsv4_inventory_gate.py
--generator ${CMAKE_CURRENT_SOURCE_DIR}/tools/make_dsv4_inventory_fixture.py
--cli $<TARGET_FILE:dsv4-inventory>
--config ${DSV4_FIX}/config_hf.json
--verifier ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_dsv4_inventory.py
--workdir ${CMAKE_CURRENT_BINARY_DIR}/dsv4-inventory-gate)

add_test(NAME dsv4_cpu_ops COMMAND test_dsv4_cpu_ops)
add_test(NAME dsv4_attention_state COMMAND test_dsv4_attention_state)
add_test(NAME dsv4_attention_numeric COMMAND test_dsv4_attention_numeric)
add_test(NAME dsv4_sparse_index COMMAND test_dsv4_sparse_index)
add_test(NAME dsv4_attention_step COMMAND test_dsv4_attention_step)
add_test(NAME dsv4_attention_runtime COMMAND test_dsv4_attention_runtime)
add_test(NAME dsv4_moe COMMAND test_dsv4_moe)
add_test(NAME dsv4_decoder_layer COMMAND test_dsv4_decoder_layer)
add_test(NAME dsv4_model COMMAND test_dsv4_model)

add_test(NAME scale      COMMAND scale_test)
add_test(NAME oracle     COMMAND k3_model   ${FIX})
add_test(NAME safetensors COMMAND test_st   ${FIX}/st ${CMAKE_CURRENT_BINARY_DIR}/idx.json
         plain.f32.2d plain.bf16.1d tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32)

# The tokenizer test needs the released vocabulary, which is not in this repository.
# Registered so `ctest` reports it as skipped rather than omitting it silently; point
# K3_TOK_FILES at a checkpoint directory to make it run.
if(DEFINED ENV{K3_TOK_FILES})
  add_test(NAME tokenizer COMMAND test_tok $ENV{K3_TOK_FILES} roundtrip
           ${CMAKE_CURRENT_SOURCE_DIR}/README.md)
endif()

add_executable(bench_kernels benchmarks/bench_kernels.c)
target_link_libraries(bench_kernels PRIVATE k3)

install(TARGETS k3_cli RUNTIME DESTINATION bin)
install(DIRECTORY include/k3 DESTINATION include)
# k3_cfg.h includes json.h and exposes jval in its signatures, so an installed header
# set without it does not compile. Kept in step with the Makefile's install target.
install(FILES third_party/json.h DESTINATION include/k3)

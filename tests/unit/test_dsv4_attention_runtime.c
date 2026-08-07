#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_attention_runtime_vectors.h"

#include "dsv4_attention_runtime.h"

#define H DSV4_RUNTIME_HIDDEN
#define QR DSV4_RUNTIME_Q_RANK
#define NH DSV4_RUNTIME_HEADS
#define HD DSV4_RUNTIME_HEAD_DIM
#define OG DSV4_RUNTIME_GROUPS
#define OR DSV4_RUNTIME_O_RANK
#define IH DSV4_RUNTIME_INDEX_HEADS
#define ID DSV4_RUNTIME_INDEX_DIM
#define MAX_RATE 128u
#define MAX_COMP_PROJ (2u * HD)
#define MAX_INDEX_COMP_PROJ (2u * ID)

typedef struct OwnedWeights {
    float q_a[QR * H];
    float q_a_norm[QR];
    float q_b[NH * HD * QR];
    float kv[HD * H];
    float kv_norm[HD];
    float sinks[NH];
    float o_a[OG * OR * (NH * HD / OG)];
    float o_b[H * OG * OR];
    float compress_kv[MAX_COMP_PROJ * H];
    float compress_gate[MAX_COMP_PROJ * H];
    float compress_bias[MAX_RATE * MAX_COMP_PROJ];
    float compress_norm[HD];
    float index_q[IH * ID * QR];
    float index_head[IH * H];
    float index_compress_kv[MAX_INDEX_COMP_PROJ * H];
    float index_compress_gate[MAX_INDEX_COMP_PROJ * H];
    float index_compress_bias[4u * MAX_INDEX_COMP_PROJ];
    float index_compress_norm[ID];
} OwnedWeights;

static int failures = 0;

static void fail(const char *what, size_t position, size_t index, float got, float expected) {
    fprintf(stderr, "FAIL %s pos=%zu idx=%zu got=%g expected=%g\n",
            what, position, index, (double)got, (double)expected);
    ++failures;
}

static void assert_close(const char *what, size_t position, size_t index,
                         float got, float expected, float tolerance) {
    if (!isfinite(got) || fabsf(got - expected) > tolerance)
        fail(what, position, index, got, expected);
}

static float pattern_value(size_t index, unsigned tag, unsigned denominator) {
    int integer = (int)((index * 17u + (size_t)tag * 11u) % 31u) - 15;
    return (float)integer / (float)denominator;
}

static float norm_value(size_t index, unsigned tag) {
    int integer = (int)((index * 5u + (size_t)tag * 3u) % 9u) - 4;
    return 1.0f + (float)integer / 64.0f;
}

static void fill_pattern(float *values, size_t count, unsigned tag, unsigned denominator) {
    size_t i;
    for (i = 0u; i < count; ++i) values[i] = pattern_value(i, tag, denominator);
}

static void fill_norm(float *values, size_t count, unsigned tag) {
    size_t i;
    for (i = 0u; i < count; ++i) values[i] = norm_value(i, tag);
}

static void fill_hidden(float *hidden, unsigned layer_idx, size_t position) {
    size_t i;
    for (i = 0u; i < H; ++i) {
        int integer = (int)((i * 7u + position * 5u + (size_t)layer_idx * 13u) % 23u) - 11;
        hidden[i] = (float)integer / 32.0f;
    }
}

static DSV4AttentionRuntimeConfig make_config(size_t rate) {
    DSV4AttentionRuntimeConfig config;
    memset(&config, 0, sizeof(config));
    config.hidden_size = H;
    config.q_lora_rank = QR;
    config.num_heads = NH;
    config.head_dim = HD;
    config.rope_dim = DSV4_RUNTIME_ROPE_DIM;
    config.output_groups = OG;
    config.output_rank = OR;
    config.sliding_window = DSV4_RUNTIME_WINDOW;
    config.compression_rate = rate;
    config.index_num_heads = IH;
    config.index_head_dim = ID;
    config.index_rope_dim = DSV4_RUNTIME_ROPE_DIM;
    config.index_top_k = DSV4_RUNTIME_INDEX_TOPK;
    config.rms_eps = DSV4_RUNTIME_RMS_EPS;
    config.rope_theta = DSV4_RUNTIME_ROPE_THETA;
    config.compress_rope_theta = DSV4_RUNTIME_COMPRESS_THETA;
    return config;
}

static DSV4AttentionRuntimeWeights fill_weights(OwnedWeights *owned,
                                                 unsigned layer_idx,
                                                 size_t rate) {
    unsigned base = 40u * layer_idx + 1u;
    size_t comp_proj = rate == 4u ? 2u * HD : HD;
    DSV4AttentionRuntimeWeights weights;
    memset(owned, 0, sizeof(*owned));
    memset(&weights, 0, sizeof(weights));

    fill_pattern(owned->q_a, QR * H, base + 1u, 128u);
    fill_norm(owned->q_a_norm, QR, base + 2u);
    fill_pattern(owned->q_b, NH * HD * QR, base + 3u, 128u);
    fill_pattern(owned->kv, HD * H, base + 4u, 128u);
    fill_norm(owned->kv_norm, HD, base + 5u);
    fill_pattern(owned->o_a, OG * OR * (NH * HD / OG), base + 6u, 128u);
    fill_pattern(owned->o_b, H * OG * OR, base + 7u, 128u);
    fill_pattern(owned->sinks, NH, base + 8u, 32u);

    weights.q_a_weight = owned->q_a;
    weights.q_a_norm_weight = owned->q_a_norm;
    weights.q_b_weight = owned->q_b;
    weights.kv_weight = owned->kv;
    weights.kv_norm_weight = owned->kv_norm;
    weights.sinks = owned->sinks;
    weights.o_a_weight = owned->o_a;
    weights.o_b_weight = owned->o_b;

    if (rate != 0u) {
        fill_pattern(owned->compress_kv, comp_proj * H, base + 11u, 128u);
        fill_pattern(owned->compress_gate, comp_proj * H, base + 12u, 128u);
        fill_pattern(owned->compress_bias, rate * comp_proj, base + 13u, 256u);
        fill_norm(owned->compress_norm, HD, base + 14u);
        weights.compress_kv_weight = owned->compress_kv;
        weights.compress_gate_weight = owned->compress_gate;
        weights.compress_position_bias = owned->compress_bias;
        weights.compress_norm_weight = owned->compress_norm;
    }
    if (rate == 4u) {
        fill_pattern(owned->index_q, IH * ID * QR, base + 20u, 128u);
        fill_pattern(owned->index_head, IH * H, base + 21u, 128u);
        fill_pattern(owned->index_compress_kv, MAX_INDEX_COMP_PROJ * H, base + 23u, 128u);
        fill_pattern(owned->index_compress_gate, MAX_INDEX_COMP_PROJ * H, base + 24u, 128u);
        fill_pattern(owned->index_compress_bias, 4u * MAX_INDEX_COMP_PROJ, base + 25u, 256u);
        fill_norm(owned->index_compress_norm, ID, base + 26u);
        weights.index_q_weight = owned->index_q;
        weights.index_head_weight = owned->index_head;
        weights.index_compress_kv_weight = owned->index_compress_kv;
        weights.index_compress_gate_weight = owned->index_compress_gate;
        weights.index_compress_position_bias = owned->index_compress_bias;
        weights.index_compress_norm_weight = owned->index_compress_norm;
    }
    return weights;
}

static void run_case(const char *name,
                     unsigned layer_idx,
                     size_t rate,
                     size_t steps,
                     const float *expected_output,
                     const unsigned *expected_raw,
                     const unsigned *expected_compressed,
                     const unsigned *expected_indexer,
                     const unsigned *expected_selected_count,
                     const unsigned *expected_selected) {
    DSV4AttentionRuntimeConfig config = make_config(rate);
    DSV4AttentionRuntimeStatus status = DSV4_AR_OK;
    DSV4AttentionRuntime *runtime = dsv4_attention_runtime_create(&config, &status);
    OwnedWeights owned;
    DSV4AttentionRuntimeWeights weights = fill_weights(&owned, layer_idx, rate);
    size_t position, i;
    if (runtime == NULL || status != DSV4_AR_OK) {
        fprintf(stderr, "FAIL %s create status=%d\n", name, (int)status);
        ++failures;
        return;
    }

    for (position = 0u; position < steps; ++position) {
        float hidden[H], output[H];
        size_t selected[DSV4_RUNTIME_INDEX_TOPK] = {DSV4_RUNTIME_SENTINEL, DSV4_RUNTIME_SENTINEL};
        size_t selected_count = 777u;
        DSV4AttentionRuntimeTrace trace = {999u, 999u, 999u};
        fill_hidden(hidden, layer_idx, position);
        for (i = 0u; i < H; ++i) output[i] = 99.0f;
        status = dsv4_attention_runtime_step_f32(
            runtime, &weights, (int64_t)position, hidden, output,
            selected, DSV4_RUNTIME_INDEX_TOPK, &selected_count, &trace);
        if (status != DSV4_AR_OK) {
            fprintf(stderr, "FAIL %s step=%zu status=%d\n", name, position, (int)status);
            ++failures;
            break;
        }
        for (i = 0u; i < H; ++i)
            assert_close(name, position, i, output[i], expected_output[position * H + i], 4e-5f);
        if (trace.raw_cache_length != expected_raw[position] ||
            trace.compressed_count != expected_compressed[position] ||
            trace.indexer_compressed_count != expected_indexer[position]) {
            fprintf(stderr,
                    "FAIL %s trace pos=%zu raw=%zu/%u comp=%zu/%u idx=%zu/%u\n",
                    name, position,
                    trace.raw_cache_length, expected_raw[position],
                    trace.compressed_count, expected_compressed[position],
                    trace.indexer_compressed_count, expected_indexer[position]);
            ++failures;
        }
        if (selected_count != expected_selected_count[position]) {
            fprintf(stderr, "FAIL %s selected-count pos=%zu got=%zu expected=%u\n",
                    name, position, selected_count, expected_selected_count[position]);
            ++failures;
        }
        for (i = 0u; i < selected_count; ++i) {
            unsigned expected = expected_selected[position * DSV4_RUNTIME_INDEX_TOPK + i];
            if (selected[i] != (size_t)expected) {
                fprintf(stderr, "FAIL %s selected pos=%zu slot=%zu got=%zu expected=%u\n",
                        name, position, i, selected[i], expected);
                ++failures;
            }
        }
    }
    dsv4_attention_runtime_destroy(runtime);
}

static void test_transactional_rejection(void) {
    DSV4AttentionRuntimeConfig config = make_config(0u);
    DSV4AttentionRuntimeStatus status = DSV4_AR_OK;
    DSV4AttentionRuntime *runtime = dsv4_attention_runtime_create(&config, &status);
    OwnedWeights owned;
    DSV4AttentionRuntimeWeights weights = fill_weights(&owned, 0u, 0u);
    float hidden[H], output[H];
    size_t selected[DSV4_RUNTIME_INDEX_TOPK];
    size_t count = 123u;
    DSV4AttentionRuntimeTrace trace = {7u, 8u, 9u};
    size_t i;
    if (runtime == NULL || status != DSV4_AR_OK) {
        ++failures;
        return;
    }
    fill_hidden(hidden, 0u, 0u);
    status = dsv4_attention_runtime_step_f32(runtime, &weights, 1, hidden, output,
                                             selected, DSV4_RUNTIME_INDEX_TOPK, &count, &trace);
    if (status != DSV4_AR_INVALID_ARGUMENT || count != 123u ||
        trace.raw_cache_length != 7u || trace.compressed_count != 8u ||
        trace.indexer_compressed_count != 9u) {
        fprintf(stderr, "FAIL transactional wrong-position\n");
        ++failures;
    }

    fill_hidden(hidden, 0u, 0u);
    hidden[3] = NAN;
    for (i = 0u; i < H; ++i) output[i] = 55.0f;
    status = dsv4_attention_runtime_step_f32(runtime, &weights, 0, hidden, output,
                                             selected, DSV4_RUNTIME_INDEX_TOPK, &count, &trace);
    if (status != DSV4_AR_NONFINITE) {
        fprintf(stderr, "FAIL transactional nonfinite status=%d\n", (int)status);
        ++failures;
    }
    for (i = 0u; i < H; ++i)
        if (output[i] != 55.0f) {
            fprintf(stderr, "FAIL transactional output mutated\n");
            ++failures;
            break;
        }

    fill_hidden(hidden, 0u, 0u);
    status = dsv4_attention_runtime_step_f32(runtime, &weights, 0, hidden, output,
                                             selected, DSV4_RUNTIME_INDEX_TOPK, &count, &trace);
    if (status != DSV4_AR_OK) {
        fprintf(stderr, "FAIL transactional valid-after-reject status=%d\n", (int)status);
        ++failures;
    } else {
        for (i = 0u; i < H; ++i)
            assert_close("transactional", 0u, i, output[i], dsv4_runtime_sliding_output[i], 4e-5f);
    }
    dsv4_attention_runtime_destroy(runtime);
}

static void test_missing_csa_weight_is_transactional(void) {
    DSV4AttentionRuntimeConfig config = make_config(4u);
    DSV4AttentionRuntimeStatus status = DSV4_AR_OK;
    DSV4AttentionRuntime *runtime = dsv4_attention_runtime_create(&config, &status);
    OwnedWeights owned;
    DSV4AttentionRuntimeWeights weights = fill_weights(&owned, 1u, 4u);
    float hidden[H], output[H];
    size_t selected[DSV4_RUNTIME_INDEX_TOPK];
    size_t count = 71u;
    DSV4AttentionRuntimeTrace trace = {4u, 5u, 6u};
    if (runtime == NULL || status != DSV4_AR_OK) {
        ++failures;
        return;
    }
    fill_hidden(hidden, 1u, 0u);
    weights.index_q_weight = NULL;
    status = dsv4_attention_runtime_step_f32(runtime, &weights, 0, hidden, output,
                                             selected, DSV4_RUNTIME_INDEX_TOPK, &count, &trace);
    if (status != DSV4_AR_INVALID_ARGUMENT || count != 71u ||
        trace.raw_cache_length != 4u || trace.compressed_count != 5u ||
        trace.indexer_compressed_count != 6u) {
        fprintf(stderr, "FAIL missing-csa-weight transactional\n");
        ++failures;
    }
    weights.index_q_weight = owned.index_q;
    status = dsv4_attention_runtime_step_f32(runtime, &weights, 0, hidden, output,
                                             selected, DSV4_RUNTIME_INDEX_TOPK, &count, &trace);
    if (status != DSV4_AR_OK) {
        fprintf(stderr, "FAIL valid after missing-csa-weight status=%d\n", (int)status);
        ++failures;
    }
    dsv4_attention_runtime_destroy(runtime);
}

int main(void) {
    run_case("sliding", 0u, 0u, DSV4_RUNTIME_SLIDING_STEPS,
             dsv4_runtime_sliding_output,
             dsv4_runtime_sliding_raw_count,
             dsv4_runtime_sliding_compressed_count,
             dsv4_runtime_sliding_indexer_count,
             dsv4_runtime_sliding_selected_count,
             dsv4_runtime_sliding_selected);
    run_case("csa", 1u, 4u, DSV4_RUNTIME_CSA_STEPS,
             dsv4_runtime_csa_output,
             dsv4_runtime_csa_raw_count,
             dsv4_runtime_csa_compressed_count,
             dsv4_runtime_csa_indexer_count,
             dsv4_runtime_csa_selected_count,
             dsv4_runtime_csa_selected);
    run_case("hca", 2u, 128u, DSV4_RUNTIME_HCA_STEPS,
             dsv4_runtime_hca_output,
             dsv4_runtime_hca_raw_count,
             dsv4_runtime_hca_compressed_count,
             dsv4_runtime_hca_indexer_count,
             dsv4_runtime_hca_selected_count,
             dsv4_runtime_hca_selected);
    test_transactional_rejection();
    test_missing_csa_weight_is_transactional();
    if (failures != 0) {
        fprintf(stderr, "%d runtime parity failures\n", failures);
        return 1;
    }
    puts("DeepSeek native attention runtime parity: OK");
    return 0;
}

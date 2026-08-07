#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsv4_attention_runtime.h"
#include "dsv4_moe.h"
#include "dsv4_decoder_layer_vectors.h"

#include "dsv4_decoder_layer.h"

#define H DSV4_DECODER_HIDDEN
#define HC DSV4_DECODER_HC
#define QR DSV4_DECODER_Q_RANK
#define NH DSV4_DECODER_HEADS
#define HD DSV4_DECODER_HEAD_DIM
#define OG DSV4_DECODER_GROUPS
#define OR DSV4_DECODER_O_RANK
#define IH DSV4_DECODER_INDEX_HEADS
#define ID DSV4_DECODER_INDEX_DIM
#define IK DSV4_DECODER_INDEX_TOPK
#define M DSV4_DECODER_INTERMEDIATE
#define E DSV4_DECODER_EXPERTS
#define K DSV4_DECODER_MOE_TOPK
#define V DSV4_DECODER_VOCAB
#define HC_MIX DSV4_DECODER_HC_MIX
#define HC_DIM (HC * H)
#define MAX_RATE 128u
#define MAX_COMP_PROJ (2u * HD)
#define MAX_INDEX_COMP_PROJ (2u * ID)

typedef struct OwnedAttentionWeights {
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
} OwnedAttentionWeights;

typedef struct OwnedMoEWeights {
    float router[E * H];
    float bias[E];
    uint32_t hash[V * K];
    float expert_gate[E * M * H];
    float expert_up[E * M * H];
    float expert_down[E * H * M];
    float shared_gate[M * H];
    float shared_up[M * H];
    float shared_down[H * M];
} OwnedMoEWeights;

typedef struct OwnedLayerWeights {
    float attention_hc_fn[HC_MIX * HC_DIM];
    float attention_hc_base[HC_MIX];
    float attention_hc_scale[3];
    float ffn_hc_fn[HC_MIX * HC_DIM];
    float ffn_hc_base[HC_MIX];
    float ffn_hc_scale[3];
    float input_norm[H];
    float post_attention_norm[H];
    OwnedAttentionWeights attention;
    OwnedMoEWeights moe;
} OwnedLayerWeights;

static int failures = 0;

static float attention_pattern(size_t index, unsigned tag, unsigned denominator) {
    int integer = (int)((index * 17u + (size_t)tag * 11u) % 31u) - 15;
    return (float)integer / (float)denominator;
}

static float attention_norm(size_t index, unsigned tag) {
    int integer = (int)((index * 5u + (size_t)tag * 3u) % 9u) - 4;
    return 1.0f + (float)integer / 64.0f;
}

static float moe_pattern(size_t index, unsigned tag, unsigned denominator) {
    int integer = (int)((index * 13u + (size_t)tag * 17u) % 29u) - 14;
    return (float)integer / (float)denominator;
}

static float hc_pattern(size_t index, unsigned tag, unsigned denominator) {
    int integer = (int)((index * 11u + (size_t)tag * 7u) % 27u) - 13;
    return (float)integer / (float)denominator;
}

static float hc_norm(size_t index, unsigned tag) {
    int integer = (int)((index * 5u + (size_t)tag * 2u) % 11u) - 5;
    return 1.0f + (float)integer / 64.0f;
}

static float hc_scale(size_t index, unsigned tag) {
    int integer = (int)((index * 3u + tag) % 7u) - 3;
    return 1.0f + (float)integer / 16.0f;
}

static void fill_attention_pattern(float *values, size_t count,
                                   unsigned tag, unsigned denominator) {
    size_t i;
    for (i = 0u; i < count; ++i) values[i] = attention_pattern(i, tag, denominator);
}

static void fill_attention_norm(float *values, size_t count, unsigned tag) {
    size_t i;
    for (i = 0u; i < count; ++i) values[i] = attention_norm(i, tag);
}

static void fill_moe_pattern(float *values, size_t count,
                             unsigned tag, unsigned denominator) {
    size_t i;
    for (i = 0u; i < count; ++i) values[i] = moe_pattern(i, tag, denominator);
}

static void fill_hc_pattern(float *values, size_t count,
                            unsigned tag, unsigned denominator) {
    size_t i;
    for (i = 0u; i < count; ++i) values[i] = hc_pattern(i, tag, denominator);
}

static void fill_hc_norm(float *values, size_t count, unsigned tag) {
    size_t i;
    for (i = 0u; i < count; ++i) values[i] = hc_norm(i, tag);
}

static void fill_hc_scale(float values[3], unsigned tag) {
    size_t i;
    for (i = 0u; i < 3u; ++i) values[i] = hc_scale(i, tag);
}

static DSV4AttentionRuntimeConfig make_attention_config(size_t rate) {
    DSV4AttentionRuntimeConfig config;
    memset(&config, 0, sizeof(config));
    config.hidden_size = H;
    config.q_lora_rank = QR;
    config.num_heads = NH;
    config.head_dim = HD;
    config.rope_dim = DSV4_DECODER_ROPE_DIM;
    config.output_groups = OG;
    config.output_rank = OR;
    config.sliding_window = DSV4_DECODER_WINDOW;
    config.compression_rate = rate;
    config.index_num_heads = IH;
    config.index_head_dim = ID;
    config.index_rope_dim = DSV4_DECODER_ROPE_DIM;
    config.index_top_k = IK;
    config.rms_eps = DSV4_DECODER_RMS_EPS;
    config.rope_theta = DSV4_DECODER_ROPE_THETA;
    config.compress_rope_theta = DSV4_DECODER_COMPRESS_THETA;
    return config;
}

static DSV4MoEConfig make_moe_config(void) {
    DSV4MoEConfig config;
    config.hidden_size = H;
    config.intermediate_size = M;
    config.num_routed_experts = E;
    config.top_k = K;
    config.num_hash_layers = DSV4_DECODER_HASH_LAYERS;
    config.vocab_size = V;
    config.route_scale = DSV4_DECODER_ROUTE_SCALE;
    config.swiglu_limit = DSV4_DECODER_SWIGLU_LIMIT;
    return config;
}

static DSV4DecoderLayerConfig make_layer_config(unsigned layer_idx, size_t rate) {
    DSV4DecoderLayerConfig config;
    memset(&config, 0, sizeof(config));
    config.layer_idx = layer_idx;
    config.hidden_size = H;
    config.hc_mult = HC;
    config.sinkhorn_iterations = DSV4_DECODER_SINKHORN_ITERS;
    config.hc_eps = DSV4_DECODER_HC_EPS;
    config.norm_eps = DSV4_DECODER_RMS_EPS;
    config.attention = make_attention_config(rate);
    config.moe = make_moe_config();
    return config;
}

static DSV4AttentionRuntimeWeights fill_attention_weights(
    OwnedAttentionWeights *owned, unsigned layer_idx, size_t rate) {
    DSV4AttentionRuntimeWeights weights;
    unsigned base = 40u * layer_idx + 1u;
    size_t comp_proj = rate == 4u ? 2u * HD : HD;
    memset(owned, 0, sizeof(*owned));
    memset(&weights, 0, sizeof(weights));

    fill_attention_pattern(owned->q_a, QR * H, base + 1u, 128u);
    fill_attention_norm(owned->q_a_norm, QR, base + 2u);
    fill_attention_pattern(owned->q_b, NH * HD * QR, base + 3u, 128u);
    fill_attention_pattern(owned->kv, HD * H, base + 4u, 128u);
    fill_attention_norm(owned->kv_norm, HD, base + 5u);
    fill_attention_pattern(owned->o_a, OG * OR * (NH * HD / OG), base + 6u, 128u);
    fill_attention_pattern(owned->o_b, H * OG * OR, base + 7u, 128u);
    fill_attention_pattern(owned->sinks, NH, base + 8u, 32u);

    weights.q_a_weight = owned->q_a;
    weights.q_a_norm_weight = owned->q_a_norm;
    weights.q_b_weight = owned->q_b;
    weights.kv_weight = owned->kv;
    weights.kv_norm_weight = owned->kv_norm;
    weights.sinks = owned->sinks;
    weights.o_a_weight = owned->o_a;
    weights.o_b_weight = owned->o_b;

    if (rate != 0u) {
        fill_attention_pattern(owned->compress_kv, comp_proj * H, base + 11u, 128u);
        fill_attention_pattern(owned->compress_gate, comp_proj * H, base + 12u, 128u);
        fill_attention_pattern(owned->compress_bias, rate * comp_proj, base + 13u, 256u);
        fill_attention_norm(owned->compress_norm, HD, base + 14u);
        weights.compress_kv_weight = owned->compress_kv;
        weights.compress_gate_weight = owned->compress_gate;
        weights.compress_position_bias = owned->compress_bias;
        weights.compress_norm_weight = owned->compress_norm;
    }
    if (rate == 4u) {
        fill_attention_pattern(owned->index_q, IH * ID * QR, base + 20u, 128u);
        fill_attention_pattern(owned->index_head, IH * H, base + 21u, 128u);
        fill_attention_pattern(owned->index_compress_kv, MAX_INDEX_COMP_PROJ * H,
                               base + 23u, 128u);
        fill_attention_pattern(owned->index_compress_gate, MAX_INDEX_COMP_PROJ * H,
                               base + 24u, 128u);
        fill_attention_pattern(owned->index_compress_bias, 4u * MAX_INDEX_COMP_PROJ,
                               base + 25u, 256u);
        fill_attention_norm(owned->index_compress_norm, ID, base + 26u);
        weights.index_q_weight = owned->index_q;
        weights.index_head_weight = owned->index_head;
        weights.index_compress_kv_weight = owned->index_compress_kv;
        weights.index_compress_gate_weight = owned->index_compress_gate;
        weights.index_compress_position_bias = owned->index_compress_bias;
        weights.index_compress_norm_weight = owned->index_compress_norm;
    }
    return weights;
}

static DSV4MoEWeights fill_moe_weights(OwnedMoEWeights *owned, unsigned layer_idx) {
    DSV4MoEWeights weights;
    unsigned base = 50u * layer_idx + 3u;
    size_t expert;
    size_t token;
    size_t slot;
    memset(owned, 0, sizeof(*owned));
    memset(&weights, 0, sizeof(weights));

    fill_moe_pattern(owned->router, E * H, base + 1u, 128u);
    fill_moe_pattern(owned->bias, E, base + 2u, 64u);
    for (expert = 0u; expert < E; ++expert) {
        fill_moe_pattern(owned->expert_gate + expert * M * H, M * H,
                         base + 10u + (unsigned)(3u * expert), 128u);
        fill_moe_pattern(owned->expert_up + expert * M * H, M * H,
                         base + 11u + (unsigned)(3u * expert), 128u);
        fill_moe_pattern(owned->expert_down + expert * H * M, H * M,
                         base + 12u + (unsigned)(3u * expert), 128u);
    }
    fill_moe_pattern(owned->shared_gate, M * H, base + 40u, 128u);
    fill_moe_pattern(owned->shared_up, M * H, base + 41u, 128u);
    fill_moe_pattern(owned->shared_down, H * M, base + 42u, 128u);
    for (token = 0u; token < V; ++token) {
        size_t start = token * 5u + (size_t)layer_idx * 3u;
        for (slot = 0u; slot < K; ++slot)
            owned->hash[token * K + slot] = (uint32_t)((start + slot * 7u) % E);
    }

    weights.router_weight = owned->router;
    weights.correction_bias = owned->bias;
    weights.hash_table = owned->hash;
    weights.expert_gate_weight = owned->expert_gate;
    weights.expert_up_weight = owned->expert_up;
    weights.expert_down_weight = owned->expert_down;
    weights.shared_gate_weight = owned->shared_gate;
    weights.shared_up_weight = owned->shared_up;
    weights.shared_down_weight = owned->shared_down;
    return weights;
}

static DSV4DecoderLayerWeights fill_layer_weights(
    OwnedLayerWeights *owned, unsigned layer_idx, size_t rate) {
    DSV4DecoderLayerWeights weights;
    unsigned base = 80u * layer_idx + 5u;
    memset(owned, 0, sizeof(*owned));
    memset(&weights, 0, sizeof(weights));

    fill_hc_norm(owned->input_norm, H, base + 1u);
    fill_hc_norm(owned->post_attention_norm, H, base + 2u);
    fill_hc_pattern(owned->attention_hc_fn, HC_MIX * HC_DIM, base + 11u, 128u);
    fill_hc_pattern(owned->attention_hc_base, HC_MIX, base + 12u, 256u);
    fill_hc_scale(owned->attention_hc_scale, base + 13u);
    fill_hc_pattern(owned->ffn_hc_fn, HC_MIX * HC_DIM, base + 21u, 128u);
    fill_hc_pattern(owned->ffn_hc_base, HC_MIX, base + 22u, 256u);
    fill_hc_scale(owned->ffn_hc_scale, base + 23u);

    weights.attention_hc_fn = owned->attention_hc_fn;
    weights.attention_hc_base = owned->attention_hc_base;
    weights.attention_hc_scale = owned->attention_hc_scale;
    weights.ffn_hc_fn = owned->ffn_hc_fn;
    weights.ffn_hc_base = owned->ffn_hc_base;
    weights.ffn_hc_scale = owned->ffn_hc_scale;
    weights.input_norm_weight = owned->input_norm;
    weights.post_attention_norm_weight = owned->post_attention_norm;
    weights.attention = fill_attention_weights(&owned->attention, layer_idx, rate);
    weights.moe = fill_moe_weights(&owned->moe, layer_idx);
    if (layer_idx < DSV4_DECODER_HASH_LAYERS)
        weights.moe.correction_bias = NULL;
    else
        weights.moe.hash_table = NULL;
    return weights;
}

static void fill_streams(float streams[HC * H], unsigned layer_idx, size_t position) {
    size_t stream;
    size_t dim;
    for (stream = 0u; stream < HC; ++stream) {
        for (dim = 0u; dim < H; ++dim) {
            int integer = (int)((stream * 13u + dim * 7u + position * 5u +
                                (size_t)layer_idx * 17u) % 31u) - 15;
            streams[stream * H + dim] = (float)integer / 32.0f;
        }
    }
}

static uint64_t token_for(unsigned layer_idx, size_t position) {
    return (uint64_t)((3u + layer_idx * 5u + position * 7u) % V);
}

static void sort_route_pairs(uint32_t indices[K], float weights[K]) {
    size_t i;
    size_t j;
    for (i = 0u; i < K; ++i) {
        for (j = i + 1u; j < K; ++j) {
            if (indices[j] < indices[i]) {
                uint32_t index_tmp = indices[i];
                float weight_tmp = weights[i];
                indices[i] = indices[j];
                weights[i] = weights[j];
                indices[j] = index_tmp;
                weights[j] = weight_tmp;
            }
        }
    }
}

static void assert_close(const char *label, size_t position, size_t slot,
                         float got, float expected, float tolerance) {
    if (!isfinite(got) || fabsf(got - expected) > tolerance) {
        fprintf(stderr, "FAIL %s pos=%zu slot=%zu got=%g expected=%g\n",
                label, position, slot, (double)got, (double)expected);
        ++failures;
    }
}

static void run_case(const char *name,
                     unsigned layer_idx,
                     size_t rate,
                     size_t steps,
                     const float *expected_output,
                     const unsigned *expected_raw,
                     const unsigned *expected_compressed,
                     const unsigned *expected_indexer,
                     const unsigned *expected_attention_count,
                     const unsigned *expected_attention_indices,
                     const unsigned *expected_route_indices,
                     const float *expected_route_weights) {
    DSV4DecoderLayerConfig config = make_layer_config(layer_idx, rate);
    DSV4DecoderLayerStatus status = DSV4_DL_OK;
    DSV4DecoderLayer *layer = dsv4_decoder_layer_create(&config, &status);
    OwnedLayerWeights owned;
    DSV4DecoderLayerWeights weights = fill_layer_weights(&owned, layer_idx, rate);
    size_t position;
    if (layer == NULL || status != DSV4_DL_OK) {
        fprintf(stderr, "FAIL %s create status=%d\n", name, (int)status);
        ++failures;
        return;
    }

    for (position = 0u; position < steps; ++position) {
        float streams[HC * H];
        float output[HC * H];
        size_t attention_indices[IK] = {DSV4_DECODER_SENTINEL, DSV4_DECODER_SENTINEL};
        size_t attention_count = 777u;
        uint32_t route_indices[K] = {UINT32_MAX, UINT32_MAX};
        float route_weights[K] = {-77.0f, -77.0f};
        DSV4DecoderLayerTrace trace = {{999u, 999u, 999u}};
        size_t i;

        fill_streams(streams, layer_idx, position);
        for (i = 0u; i < HC * H; ++i) output[i] = 99.0f;
        status = dsv4_decoder_layer_step_f32(
            layer, &weights, (int64_t)position, token_for(layer_idx, position),
            streams, output,
            attention_indices, IK, &attention_count,
            route_indices, route_weights, K, &trace);
        if (status != DSV4_DL_OK) {
            fprintf(stderr, "FAIL %s step=%zu status=%d\n", name, position, (int)status);
            ++failures;
            break;
        }
        for (i = 0u; i < HC * H; ++i)
            assert_close(name, position, i, output[i],
                         expected_output[position * HC * H + i], 3e-4f);
        if (trace.attention.raw_cache_length != expected_raw[position] ||
            trace.attention.compressed_count != expected_compressed[position] ||
            trace.attention.indexer_compressed_count != expected_indexer[position]) {
            fprintf(stderr,
                    "FAIL %s trace pos=%zu raw=%zu/%u comp=%zu/%u idx=%zu/%u\n",
                    name, position,
                    trace.attention.raw_cache_length, expected_raw[position],
                    trace.attention.compressed_count, expected_compressed[position],
                    trace.attention.indexer_compressed_count, expected_indexer[position]);
            ++failures;
        }
        if (attention_count != expected_attention_count[position]) {
            fprintf(stderr, "FAIL %s attention-count pos=%zu got=%zu expected=%u\n",
                    name, position, attention_count, expected_attention_count[position]);
            ++failures;
        }
        for (i = 0u; i < attention_count; ++i) {
            unsigned expected = expected_attention_indices[position * IK + i];
            if (attention_indices[i] != (size_t)expected) {
                fprintf(stderr, "FAIL %s attention-index pos=%zu slot=%zu got=%zu expected=%u\n",
                        name, position, i, attention_indices[i], expected);
                ++failures;
            }
        }
        sort_route_pairs(route_indices, route_weights);
        for (i = 0u; i < K; ++i) {
            unsigned expected_index = expected_route_indices[position * K + i];
            if (route_indices[i] != expected_index) {
                fprintf(stderr, "FAIL %s route-index pos=%zu slot=%zu got=%u expected=%u\n",
                        name, position, i, route_indices[i], expected_index);
                ++failures;
            }
            assert_close("route-weight", position, i, route_weights[i],
                         expected_route_weights[position * K + i], 3e-5f);
        }
    }
    dsv4_decoder_layer_destroy(layer);
}

static void test_late_failure_rolls_back_attention(void) {
    DSV4DecoderLayerConfig config = make_layer_config(1u, 4u);
    DSV4DecoderLayerStatus status = DSV4_DL_OK;
    DSV4DecoderLayer *layer = dsv4_decoder_layer_create(&config, &status);
    OwnedLayerWeights owned;
    DSV4DecoderLayerWeights weights = fill_layer_weights(&owned, 1u, 4u);
    float streams[HC * H];
    float output[HC * H];
    size_t attention_indices[IK] = {41u, 42u};
    size_t attention_count = 43u;
    uint32_t route_indices[K] = {44u, 45u};
    float route_weights[K] = {46.0f, 47.0f};
    DSV4DecoderLayerTrace trace = {{48u, 49u, 50u}};
    size_t i;

    if (layer == NULL || status != DSV4_DL_OK) {
        fprintf(stderr, "FAIL rollback create status=%d\n", (int)status);
        ++failures;
        return;
    }
    fill_streams(streams, 1u, 0u);
    for (i = 0u; i < HC * H; ++i) output[i] = 55.0f;
    /* Guaranteed late failure: learned routing requires correction_bias,
     * and DecoderLayer calls MoE only after the cloned attention step. */
    weights.moe.correction_bias = NULL;

    status = dsv4_decoder_layer_step_f32(
        layer, &weights, 0, token_for(1u, 0u), streams, output,
        attention_indices, IK, &attention_count,
        route_indices, route_weights, K, &trace);
    if (status != DSV4_DL_INVALID_ARGUMENT) {
        fprintf(stderr, "FAIL rollback late status=%d\n", (int)status);
        ++failures;
    }
    for (i = 0u; i < HC * H; ++i)
        if (output[i] != 55.0f) {
            fprintf(stderr, "FAIL rollback output mutated\n");
            ++failures;
            break;
        }
    if (attention_indices[0] != 41u || attention_indices[1] != 42u ||
        attention_count != 43u || route_indices[0] != 44u || route_indices[1] != 45u ||
        route_weights[0] != 46.0f || route_weights[1] != 47.0f ||
        trace.attention.raw_cache_length != 48u ||
        trace.attention.compressed_count != 49u ||
        trace.attention.indexer_compressed_count != 50u) {
        fprintf(stderr, "FAIL rollback diagnostics mutated\n");
        ++failures;
    }

    weights = fill_layer_weights(&owned, 1u, 4u);
    status = dsv4_decoder_layer_step_f32(
        layer, &weights, 0, token_for(1u, 0u), streams, output,
        attention_indices, IK, &attention_count,
        route_indices, route_weights, K, &trace);
    if (status != DSV4_DL_OK) {
        fprintf(stderr, "FAIL rollback retry status=%d\n", (int)status);
        ++failures;
    } else {
        for (i = 0u; i < HC * H; ++i)
            assert_close("rollback-retry", 0u, i, output[i], dsv4_decoder_csa_output[i], 3e-4f);
    }
    dsv4_decoder_layer_destroy(layer);
}

int main(void) {
    run_case("sliding", 0u, 0u, DSV4_DECODER_SLIDING_STEPS,
             dsv4_decoder_sliding_output,
             dsv4_decoder_sliding_raw_count,
             dsv4_decoder_sliding_compressed_count,
             dsv4_decoder_sliding_indexer_count,
             dsv4_decoder_sliding_attention_selected_count,
             dsv4_decoder_sliding_attention_selected,
             dsv4_decoder_sliding_route_indices,
             dsv4_decoder_sliding_route_weights);
    run_case("csa", 1u, 4u, DSV4_DECODER_CSA_STEPS,
             dsv4_decoder_csa_output,
             dsv4_decoder_csa_raw_count,
             dsv4_decoder_csa_compressed_count,
             dsv4_decoder_csa_indexer_count,
             dsv4_decoder_csa_attention_selected_count,
             dsv4_decoder_csa_attention_selected,
             dsv4_decoder_csa_route_indices,
             dsv4_decoder_csa_route_weights);
    run_case("hca", 2u, 128u, DSV4_DECODER_HCA_STEPS,
             dsv4_decoder_hca_output,
             dsv4_decoder_hca_raw_count,
             dsv4_decoder_hca_compressed_count,
             dsv4_decoder_hca_indexer_count,
             dsv4_decoder_hca_attention_selected_count,
             dsv4_decoder_hca_attention_selected,
             dsv4_decoder_hca_route_indices,
             dsv4_decoder_hca_route_weights);
    test_late_failure_rolls_back_attention();
    if (failures != 0) {
        fprintf(stderr, "%d decoder-layer parity failures\n", failures);
        return 1;
    }
    puts("DeepSeek native decoder-layer parity: OK");
    return 0;
}

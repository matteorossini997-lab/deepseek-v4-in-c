#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsv4_moe_vectors.h"

#include "dsv4_moe.h"

#define H DSV4_MOE_HIDDEN
#define M DSV4_MOE_INTERMEDIATE
#define E DSV4_MOE_EXPERTS
#define K DSV4_MOE_TOPK
#define V DSV4_MOE_VOCAB

typedef struct OwnedWeights {
    float router[E * H];
    float bias[E];
    uint32_t hash[V * K];
    float expert_gate[E * M * H];
    float expert_up[E * M * H];
    float expert_down[E * H * M];
    float shared_gate[M * H];
    float shared_up[M * H];
    float shared_down[H * M];
} OwnedWeights;

static int failures = 0;

static float pattern_value(size_t index, unsigned tag, unsigned denominator) {
    int integer = (int)((index * 13u + (size_t)tag * 17u) % 29u) - 14;
    return (float)integer / (float)denominator;
}

static void fill_pattern(float *values, size_t count, unsigned tag, unsigned denominator) {
    size_t i;
    for (i = 0u; i < count; ++i) values[i] = pattern_value(i, tag, denominator);
}

static void fill_hidden(float hidden[H], unsigned layer_idx, unsigned case_idx) {
    size_t i;
    for (i = 0u; i < H; ++i) {
        int integer = (int)((i * 7u + (size_t)case_idx * 5u +
                            (size_t)layer_idx * 11u) % 19u) - 9;
        hidden[i] = (float)integer / 16.0f;
    }
}

static DSV4MoEConfig make_config(void) {
    DSV4MoEConfig config;
    config.hidden_size = H;
    config.intermediate_size = M;
    config.num_routed_experts = E;
    config.top_k = K;
    config.num_hash_layers = DSV4_MOE_HASH_LAYERS;
    config.vocab_size = V;
    config.route_scale = DSV4_MOE_ROUTE_SCALE;
    config.swiglu_limit = DSV4_MOE_SWIGLU_LIMIT;
    return config;
}

static DSV4MoEWeights fill_weights(OwnedWeights *owned, unsigned layer_idx) {
    DSV4MoEWeights weights;
    unsigned base = 50u * layer_idx + 3u;
    size_t expert;
    size_t token;
    size_t slot;
    memset(owned, 0, sizeof(*owned));
    memset(&weights, 0, sizeof(weights));

    fill_pattern(owned->router, E * H, base + 1u, 128u);
    fill_pattern(owned->bias, E, base + 2u, 64u);
    for (expert = 0u; expert < E; ++expert) {
        fill_pattern(owned->expert_gate + expert * M * H, M * H,
                     base + 10u + (unsigned)(3u * expert), 128u);
        fill_pattern(owned->expert_up + expert * M * H, M * H,
                     base + 11u + (unsigned)(3u * expert), 128u);
        fill_pattern(owned->expert_down + expert * H * M, H * M,
                     base + 12u + (unsigned)(3u * expert), 128u);
    }
    fill_pattern(owned->shared_gate, M * H, base + 40u, 128u);
    fill_pattern(owned->shared_up, M * H, base + 41u, 128u);
    fill_pattern(owned->shared_down, H * M, base + 42u, 128u);

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

static void sort_pairs(uint32_t indices[K], float weights[K]) {
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

static void check_close(const char *label, size_t case_idx, size_t slot,
                        float got, float expected, float tolerance) {
    if (!isfinite(got) || fabsf(got - expected) > tolerance) {
        fprintf(stderr, "FAIL %s case=%zu slot=%zu got=%g expected=%g\n",
                label, case_idx, slot, (double)got, (double)expected);
        ++failures;
    }
}

static void run_cases(void) {
    DSV4MoEConfig config = make_config();
    size_t case_index;
    for (case_index = 0u; case_index < DSV4_MOE_CASES; ++case_index) {
        const unsigned layer = dsv4_moe_case_layer[case_index];
        const unsigned logical_case = dsv4_moe_case_id[case_index];
        const unsigned token = dsv4_moe_case_token[case_index];
        OwnedWeights owned;
        DSV4MoEWeights weights = fill_weights(&owned, layer);
        float hidden[H];
        float output[H];
        uint32_t indices[K] = {UINT32_MAX, UINT32_MAX};
        float route_weights[K] = {-77.0f, -77.0f};
        size_t i;
        DSV4MoEStatus status;

        fill_hidden(hidden, layer, logical_case);
        for (i = 0u; i < H; ++i) output[i] = 99.0f;
        if (layer >= DSV4_MOE_HASH_LAYERS) weights.hash_table = NULL;
        else weights.correction_bias = NULL;

        status = dsv4_moe_forward_f32(&config, &weights, layer, token, hidden,
                                      output, indices, route_weights, K);
        if (status != DSV4_MOE_OK) {
            fprintf(stderr, "FAIL case=%zu status=%d\n", case_index, (int)status);
            ++failures;
            continue;
        }
        for (i = 0u; i < H; ++i)
            check_close("output", case_index, i, output[i],
                        dsv4_moe_expected_output[case_index * H + i], 5e-5f);
        sort_pairs(indices, route_weights);
        for (i = 0u; i < K; ++i) {
            const unsigned expected_index = dsv4_moe_expected_indices[case_index * K + i];
            if (indices[i] != expected_index) {
                fprintf(stderr, "FAIL route index case=%zu slot=%zu got=%u expected=%u\n",
                        case_index, i, indices[i], expected_index);
                ++failures;
            }
            check_close("route weight", case_index, i, route_weights[i],
                        dsv4_moe_expected_weights[case_index * K + i], 2e-5f);
        }
    }
}

static void test_transactional_refusal(void) {
    DSV4MoEConfig config = make_config();
    OwnedWeights owned;
    DSV4MoEWeights weights = fill_weights(&owned, 1u);
    float hidden[H];
    float output[H];
    uint32_t indices[K] = {41u, 42u};
    float route_weights[K] = {43.0f, 44.0f};
    size_t i;
    DSV4MoEStatus status;

    fill_hidden(hidden, 1u, 2u);
    weights.hash_table = NULL;
    weights.correction_bias = NULL;
    for (i = 0u; i < H; ++i) output[i] = 55.0f;
    status = dsv4_moe_forward_f32(&config, &weights, 1u, 5u, hidden,
                                  output, indices, route_weights, K);
    if (status != DSV4_MOE_INVALID_ARGUMENT) {
        fprintf(stderr, "FAIL missing learned bias status=%d\n", (int)status);
        ++failures;
    }
    for (i = 0u; i < H; ++i)
        if (output[i] != 55.0f) {
            fprintf(stderr, "FAIL output mutated on missing bias\n");
            ++failures;
            break;
        }
    if (indices[0] != 41u || indices[1] != 42u ||
        route_weights[0] != 43.0f || route_weights[1] != 44.0f) {
        fprintf(stderr, "FAIL route mutated on missing bias\n");
        ++failures;
    }

    weights = fill_weights(&owned, 0u);
    weights.correction_bias = NULL;
    status = dsv4_moe_forward_f32(&config, &weights, 0u, V, hidden,
                                  output, indices, route_weights, K);
    if (status != DSV4_MOE_INVALID_ARGUMENT) {
        fprintf(stderr, "FAIL hash token range status=%d\n", (int)status);
        ++failures;
    }

    weights = fill_weights(&owned, 1u);
    weights.hash_table = NULL;
    owned.shared_down[3] = NAN;
    status = dsv4_moe_forward_f32(&config, &weights, 1u, 5u, hidden,
                                  output, indices, route_weights, K);
    if (status != DSV4_MOE_NONFINITE) {
        fprintf(stderr, "FAIL nonfinite shared weight status=%d\n", (int)status);
        ++failures;
    }
}

int main(void) {
    run_cases();
    test_transactional_refusal();
    if (failures != 0) {
        fprintf(stderr, "%d MoE parity failures\n", failures);
        return 1;
    }
    puts("DeepSeek native MoE parity: OK");
    return 0;
}

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsv4_mtp.h"
#include "dsv4_model_shell_vectors.h"
#include "dsv4_mtp_vectors.h"

#define H DSV4_MTP_HIDDEN
#define HC DSV4_MTP_HC
#define V DSV4_MTP_VOCAB
#define K DSV4_MTP_TOPK

static int failures = 0;

static DSV4AttentionRuntimeConfig make_attention_config(void)
{
    DSV4AttentionRuntimeConfig config;
    memset(&config, 0, sizeof(config));
    config.hidden_size = H;
    config.q_lora_rank = DSV4_MTP_Q_RANK;
    config.num_heads = DSV4_MTP_HEADS;
    config.head_dim = DSV4_MTP_HEAD_DIM;
    config.rope_dim = DSV4_MTP_ROPE_DIM;
    config.output_groups = DSV4_MTP_GROUPS;
    config.output_rank = DSV4_MTP_O_RANK;
    config.sliding_window = DSV4_MTP_WINDOW;
    config.compression_rate = 0u;
    config.rms_eps = DSV4_MTP_RMS_EPS;
    config.rope_theta = DSV4_MTP_ROPE_THETA;
    config.compress_rope_theta = DSV4_MTP_COMPRESS_THETA;
    return config;
}

static DSV4MoEConfig make_moe_config(void)
{
    DSV4MoEConfig config;
    config.hidden_size = H;
    config.intermediate_size = DSV4_MTP_INTERMEDIATE;
    config.num_routed_experts = DSV4_MTP_EXPERTS;
    config.top_k = K;
    config.num_hash_layers = DSV4_MTP_HASH_LAYERS;
    config.vocab_size = V;
    config.route_scale = DSV4_MTP_ROUTE_SCALE;
    config.swiglu_limit = DSV4_MTP_SWIGLU_LIMIT;
    return config;
}

static DSV4DecoderLayerConfig make_decoder_config(void)
{
    DSV4DecoderLayerConfig config;
    memset(&config, 0, sizeof(config));
    config.layer_idx = DSV4_MTP_LAYER_IDX;
    config.hidden_size = H;
    config.hc_mult = HC;
    config.sinkhorn_iterations = DSV4_MTP_SINKHORN_ITERS;
    config.hc_eps = DSV4_MTP_HC_EPS;
    config.norm_eps = DSV4_MTP_RMS_EPS;
    config.attention = make_attention_config();
    config.moe = make_moe_config();
    return config;
}

static DSV4MTPConfig make_config(void)
{
    DSV4MTPConfig config;
    config.vocab_size = V;
    config.hidden_size = H;
    config.hc_mult = HC;
    config.max_seq_len = DSV4_MTP_MAX_SEQ;
    config.hc_eps = DSV4_MTP_HC_EPS;
    config.norm_eps = DSV4_MTP_RMS_EPS;
    config.decoder = make_decoder_config();
    return config;
}

static DSV4DecoderLayerWeights make_decoder_weights(void)
{
    DSV4DecoderLayerWeights weights;
    memset(&weights, 0, sizeof(weights));
    weights.attention_hc_fn = dsv4_mtp_decoder_attn_hc_fn;
    weights.attention_hc_base = dsv4_mtp_decoder_attn_hc_base;
    weights.attention_hc_scale = dsv4_mtp_decoder_attn_hc_scale;
    weights.ffn_hc_fn = dsv4_mtp_decoder_ffn_hc_fn;
    weights.ffn_hc_base = dsv4_mtp_decoder_ffn_hc_base;
    weights.ffn_hc_scale = dsv4_mtp_decoder_ffn_hc_scale;
    weights.input_norm_weight = dsv4_mtp_decoder_input_norm;
    weights.post_attention_norm_weight = dsv4_mtp_decoder_post_attn_norm;

    weights.attention.q_a_weight = dsv4_mtp_decoder_q_a;
    weights.attention.q_a_norm_weight = dsv4_mtp_decoder_q_a_norm;
    weights.attention.q_b_weight = dsv4_mtp_decoder_q_b;
    weights.attention.kv_weight = dsv4_mtp_decoder_kv;
    weights.attention.kv_norm_weight = dsv4_mtp_decoder_kv_norm;
    weights.attention.sinks = dsv4_mtp_decoder_sinks;
    weights.attention.o_a_weight = dsv4_mtp_decoder_o_a;
    weights.attention.o_b_weight = dsv4_mtp_decoder_o_b;

    weights.moe.router_weight = dsv4_mtp_decoder_router;
    weights.moe.correction_bias = dsv4_mtp_decoder_bias;
    weights.moe.hash_table = NULL;
    weights.moe.expert_gate_weight = dsv4_mtp_decoder_expert_gate;
    weights.moe.expert_up_weight = dsv4_mtp_decoder_expert_up;
    weights.moe.expert_down_weight = dsv4_mtp_decoder_expert_down;
    weights.moe.shared_gate_weight = dsv4_mtp_decoder_shared_gate;
    weights.moe.shared_up_weight = dsv4_mtp_decoder_shared_up;
    weights.moe.shared_down_weight = dsv4_mtp_decoder_shared_down;
    return weights;
}

static DSV4MTPWeights make_weights(void)
{
    DSV4MTPWeights weights;
    memset(&weights, 0, sizeof(weights));
    weights.embedding_weight = dsv4_model_embedding;
    weights.lm_head_weight = dsv4_model_lm_head;
    weights.embedding_norm_weight = dsv4_mtp_embedding_norm;
    weights.hidden_norm_weight = dsv4_mtp_hidden_norm;
    weights.embedding_proj_weight = dsv4_mtp_embedding_proj;
    weights.hidden_proj_weight = dsv4_mtp_hidden_proj;
    weights.decoder = make_decoder_weights();
    weights.hc_head_fn = dsv4_mtp_hc_head_fn;
    weights.hc_head_base = dsv4_mtp_hc_head_base;
    weights.hc_head_scale = dsv4_mtp_hc_head_scale;
    weights.output_norm_weight = dsv4_mtp_output_norm;
    return weights;
}

static void assert_close(const char *name, size_t position, size_t index,
                         float got, float expected, float tolerance)
{
    float error = fabsf(got - expected);
    if (!isfinite(got) || error > tolerance) {
        fprintf(stderr,
                "FAIL %s pos=%zu idx=%zu got=%.9g expected=%.9g error=%.9g tol=%.9g\n",
                name, position, index, (double)got, (double)expected,
                (double)error, (double)tolerance);
        ++failures;
    }
}

static unsigned argmax(const float *values, size_t count)
{
    size_t i;
    size_t best = 0u;
    for (i = 1u; i < count; ++i)
        if (values[i] > values[best]) best = i;
    return (unsigned)best;
}

static void assert_route_pairs(size_t position,
                               const uint32_t got_index[K],
                               const float got_weight[K])
{
    size_t expected_slot;
    for (expected_slot = 0u; expected_slot < K; ++expected_slot) {
        uint32_t expected_index =
            dsv4_mtp_expected_route_indices[position * K + expected_slot];
        float expected_weight =
            dsv4_mtp_expected_route_weights[position * K + expected_slot];
        size_t got_slot;
        int found = 0;
        for (got_slot = 0u; got_slot < K; ++got_slot) {
            if (got_index[got_slot] == expected_index) {
                assert_close("route-weight", position, expected_slot,
                             got_weight[got_slot], expected_weight, 2e-5f);
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "FAIL route expert pos=%zu missing=%u\n",
                    position, expected_index);
            ++failures;
        }
    }
}

static void run_fixture(void)
{
    DSV4MTPConfig config = make_config();
    DSV4MTPWeights weights = make_weights();
    DSV4MTPStatus status = DSV4_MTP_OK;
    DSV4MTP *mtp = dsv4_mtp_create(&config, &status);
    size_t position;

    if (mtp == NULL || status != DSV4_MTP_OK) {
        fprintf(stderr, "FAIL MTP create status=%d\n", (int)status);
        ++failures;
        return;
    }

    for (position = 0u; position < DSV4_MTP_STEPS; ++position) {
        float streams[HC * H];
        float logits[V];
        uint32_t route_indices[K];
        float route_weights[K];
        DSV4MTPTrace trace;
        size_t i;

        for (i = 0u; i < HC * H; ++i) streams[i] = 91.0f;
        for (i = 0u; i < V; ++i) logits[i] = 92.0f;
        for (i = 0u; i < K; ++i) {
            route_indices[i] = UINT32_MAX;
            route_weights[i] = 93.0f;
        }
        memset(&trace, 0, sizeof(trace));

        status = dsv4_mtp_step_f32(
            mtp, &weights, dsv4_mtp_token[position],
            &dsv4_mtp_base_streams[position * HC * H],
            streams, logits, route_indices, route_weights, K, &trace);
        if (status != DSV4_MTP_OK) {
            fprintf(stderr, "FAIL MTP step=%zu status=%d\n", position, (int)status);
            ++failures;
            break;
        }
        for (i = 0u; i < HC * H; ++i)
            assert_close("streams", position, i, streams[i],
                         dsv4_mtp_expected_streams[position * HC * H + i], 8e-4f);
        for (i = 0u; i < V; ++i)
            assert_close("logits", position, i, logits[i],
                         dsv4_mtp_expected_logits[position * V + i], 2e-3f);
        if (argmax(logits, V) != dsv4_mtp_expected_argmax[position]) {
            fprintf(stderr, "FAIL argmax pos=%zu got=%u expected=%u\n",
                    position, argmax(logits, V), dsv4_mtp_expected_argmax[position]);
            ++failures;
        }
        assert_route_pairs(position, route_indices, route_weights);
        if (trace.decoder.attention.raw_cache_length !=
            dsv4_mtp_expected_raw_cache[position]) {
            fprintf(stderr, "FAIL raw-cache pos=%zu got=%zu expected=%u\n",
                    position, trace.decoder.attention.raw_cache_length,
                    dsv4_mtp_expected_raw_cache[position]);
            ++failures;
        }
    }
    dsv4_mtp_destroy(mtp);
}

static void test_rollback(void)
{
    DSV4MTPConfig config = make_config();
    DSV4MTPWeights weights = make_weights();
    DSV4MTPStatus status = DSV4_MTP_OK;
    DSV4MTP *mtp = dsv4_mtp_create(&config, &status);
    const float *saved_bias;
    float streams[HC * H];
    float logits[V];
    uint32_t route_indices[K];
    float route_weights[K];
    DSV4MTPTrace trace;
    DSV4MTPTrace sentinel_trace;
    size_t i;

    if (mtp == NULL || status != DSV4_MTP_OK) {
        fprintf(stderr, "FAIL rollback create status=%d\n", (int)status);
        ++failures;
        return;
    }

    saved_bias = weights.decoder.moe.correction_bias;
    weights.decoder.moe.correction_bias = NULL;
    for (i = 0u; i < HC * H; ++i) streams[i] = 81.0f;
    for (i = 0u; i < V; ++i) logits[i] = 82.0f;
    for (i = 0u; i < K; ++i) {
        route_indices[i] = UINT32_MAX - 1u;
        route_weights[i] = 83.0f;
    }
    memset(&trace, 0xA5, sizeof(trace));
    sentinel_trace = trace;

    status = dsv4_mtp_step_f32(
        mtp, &weights, dsv4_mtp_token[0], dsv4_mtp_base_streams,
        streams, logits, route_indices, route_weights, K, &trace);
    if (status != DSV4_MTP_INVALID_ARGUMENT) {
        fprintf(stderr, "FAIL rollback late status=%d\n", (int)status);
        ++failures;
    }
    for (i = 0u; i < HC * H; ++i)
        if (streams[i] != 81.0f) { ++failures; fprintf(stderr, "FAIL rollback streams mutated\n"); break; }
    for (i = 0u; i < V; ++i)
        if (logits[i] != 82.0f) { ++failures; fprintf(stderr, "FAIL rollback logits mutated\n"); break; }
    for (i = 0u; i < K; ++i)
        if (route_indices[i] != UINT32_MAX - 1u || route_weights[i] != 83.0f) {
            ++failures; fprintf(stderr, "FAIL rollback routes mutated\n"); break;
        }
    if (memcmp(&trace, &sentinel_trace, sizeof(trace)) != 0) {
        ++failures;
        fprintf(stderr, "FAIL rollback trace mutated\n");
    }

    weights.decoder.moe.correction_bias = saved_bias;
    status = dsv4_mtp_step_f32(
        mtp, &weights, dsv4_mtp_token[0], dsv4_mtp_base_streams,
        streams, logits, route_indices, route_weights, K, &trace);
    if (status != DSV4_MTP_OK) {
        fprintf(stderr, "FAIL rollback retry status=%d\n", (int)status);
        ++failures;
    } else {
        for (i = 0u; i < HC * H; ++i)
            assert_close("rollback-streams", 0u, i, streams[i],
                         dsv4_mtp_expected_streams[i], 8e-4f);
        for (i = 0u; i < V; ++i)
            assert_close("rollback-logits", 0u, i, logits[i],
                         dsv4_mtp_expected_logits[i], 2e-3f);
        assert_route_pairs(0u, route_indices, route_weights);
    }
    dsv4_mtp_destroy(mtp);
}

int main(void)
{
    run_fixture();
    test_rollback();
    if (failures != 0) {
        fprintf(stderr, "%d MTP parity failures\n", failures);
        return 1;
    }
    puts("DeepSeek native MTP parity: OK");
    return 0;
}

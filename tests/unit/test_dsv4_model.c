#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsv4_model.h"
#include "dsv4_model_shell_vectors.h"

#define H DSV4_MODEL_HIDDEN
#define HC DSV4_MODEL_HC
#define L DSV4_MODEL_LAYERS
#define V DSV4_MODEL_VOCAB
#define K DSV4_MODEL_MOE_TOPK

static int failures = 0;

static DSV4AttentionRuntimeConfig make_attention_config(size_t rate)
{
    DSV4AttentionRuntimeConfig config;
    memset(&config, 0, sizeof(config));
    config.hidden_size = H;
    config.q_lora_rank = DSV4_MODEL_Q_RANK;
    config.num_heads = DSV4_MODEL_HEADS;
    config.head_dim = DSV4_MODEL_HEAD_DIM;
    config.rope_dim = DSV4_MODEL_ROPE_DIM;
    config.output_groups = DSV4_MODEL_GROUPS;
    config.output_rank = DSV4_MODEL_O_RANK;
    config.sliding_window = DSV4_MODEL_WINDOW;
    config.compression_rate = rate;
    config.index_num_heads = DSV4_MODEL_INDEX_HEADS;
    config.index_head_dim = DSV4_MODEL_INDEX_DIM;
    config.index_rope_dim = DSV4_MODEL_ROPE_DIM;
    config.index_top_k = DSV4_MODEL_INDEX_TOPK;
    config.rms_eps = DSV4_MODEL_RMS_EPS;
    config.rope_theta = DSV4_MODEL_ROPE_THETA;
    config.compress_rope_theta = DSV4_MODEL_COMPRESS_THETA;
    return config;
}

static DSV4MoEConfig make_moe_config(void)
{
    DSV4MoEConfig config;
    config.hidden_size = H;
    config.intermediate_size = DSV4_MODEL_INTERMEDIATE;
    config.num_routed_experts = DSV4_MODEL_EXPERTS;
    config.top_k = DSV4_MODEL_MOE_TOPK;
    config.num_hash_layers = DSV4_MODEL_HASH_LAYERS;
    config.vocab_size = V;
    config.route_scale = DSV4_MODEL_ROUTE_SCALE;
    config.swiglu_limit = DSV4_MODEL_SWIGLU_LIMIT;
    return config;
}

static DSV4DecoderLayerConfig make_layer_config(uint32_t layer_idx, size_t rate)
{
    DSV4DecoderLayerConfig config;
    memset(&config, 0, sizeof(config));
    config.layer_idx = layer_idx;
    config.hidden_size = H;
    config.hc_mult = HC;
    config.sinkhorn_iterations = DSV4_MODEL_SINKHORN_ITERS;
    config.hc_eps = DSV4_MODEL_HC_EPS;
    config.norm_eps = DSV4_MODEL_RMS_EPS;
    config.attention = make_attention_config(rate);
    config.moe = make_moe_config();
    return config;
}

static void fill_layer_configs(DSV4DecoderLayerConfig configs[L])
{
    configs[0] = make_layer_config(0u, 0u);
    configs[1] = make_layer_config(1u, 4u);
    configs[2] = make_layer_config(2u, 128u);
}

static void fill_common_attention(DSV4AttentionRuntimeWeights *weights,
                                  const float *q_a,
                                  const float *q_a_norm,
                                  const float *q_b,
                                  const float *kv,
                                  const float *kv_norm,
                                  const float *sinks,
                                  const float *o_a,
                                  const float *o_b)
{
    memset(weights, 0, sizeof(*weights));
    weights->q_a_weight = q_a;
    weights->q_a_norm_weight = q_a_norm;
    weights->q_b_weight = q_b;
    weights->kv_weight = kv;
    weights->kv_norm_weight = kv_norm;
    weights->sinks = sinks;
    weights->o_a_weight = o_a;
    weights->o_b_weight = o_b;
}

static void fill_common_moe(DSV4MoEWeights *weights,
                            const float *router,
                            const float *expert_gate,
                            const float *expert_up,
                            const float *expert_down,
                            const float *shared_gate,
                            const float *shared_up,
                            const float *shared_down)
{
    memset(weights, 0, sizeof(*weights));
    weights->router_weight = router;
    weights->expert_gate_weight = expert_gate;
    weights->expert_up_weight = expert_up;
    weights->expert_down_weight = expert_down;
    weights->shared_gate_weight = shared_gate;
    weights->shared_up_weight = shared_up;
    weights->shared_down_weight = shared_down;
}

static void fill_layer_weights(DSV4DecoderLayerWeights weights[L])
{
    memset(weights, 0, L * sizeof(*weights));

    weights[0].attention_hc_fn = dsv4_model_l0_attn_hc_fn;
    weights[0].attention_hc_base = dsv4_model_l0_attn_hc_base;
    weights[0].attention_hc_scale = dsv4_model_l0_attn_hc_scale;
    weights[0].ffn_hc_fn = dsv4_model_l0_ffn_hc_fn;
    weights[0].ffn_hc_base = dsv4_model_l0_ffn_hc_base;
    weights[0].ffn_hc_scale = dsv4_model_l0_ffn_hc_scale;
    weights[0].input_norm_weight = dsv4_model_l0_input_norm;
    weights[0].post_attention_norm_weight = dsv4_model_l0_post_attn_norm;
    fill_common_attention(&weights[0].attention,
                          dsv4_model_l0_q_a, dsv4_model_l0_q_a_norm,
                          dsv4_model_l0_q_b, dsv4_model_l0_kv,
                          dsv4_model_l0_kv_norm, dsv4_model_l0_sinks,
                          dsv4_model_l0_o_a, dsv4_model_l0_o_b);
    fill_common_moe(&weights[0].moe,
                    dsv4_model_l0_router,
                    dsv4_model_l0_expert_gate,
                    dsv4_model_l0_expert_up,
                    dsv4_model_l0_expert_down,
                    dsv4_model_l0_shared_gate,
                    dsv4_model_l0_shared_up,
                    dsv4_model_l0_shared_down);
    weights[0].moe.hash_table = (const uint32_t *)dsv4_model_l0_hash;

    weights[1].attention_hc_fn = dsv4_model_l1_attn_hc_fn;
    weights[1].attention_hc_base = dsv4_model_l1_attn_hc_base;
    weights[1].attention_hc_scale = dsv4_model_l1_attn_hc_scale;
    weights[1].ffn_hc_fn = dsv4_model_l1_ffn_hc_fn;
    weights[1].ffn_hc_base = dsv4_model_l1_ffn_hc_base;
    weights[1].ffn_hc_scale = dsv4_model_l1_ffn_hc_scale;
    weights[1].input_norm_weight = dsv4_model_l1_input_norm;
    weights[1].post_attention_norm_weight = dsv4_model_l1_post_attn_norm;
    fill_common_attention(&weights[1].attention,
                          dsv4_model_l1_q_a, dsv4_model_l1_q_a_norm,
                          dsv4_model_l1_q_b, dsv4_model_l1_kv,
                          dsv4_model_l1_kv_norm, dsv4_model_l1_sinks,
                          dsv4_model_l1_o_a, dsv4_model_l1_o_b);
    weights[1].attention.compress_kv_weight = dsv4_model_l1_compress_kv;
    weights[1].attention.compress_gate_weight = dsv4_model_l1_compress_gate;
    weights[1].attention.compress_position_bias = dsv4_model_l1_compress_bias;
    weights[1].attention.compress_norm_weight = dsv4_model_l1_compress_norm;
    weights[1].attention.index_q_weight = dsv4_model_l1_index_q;
    weights[1].attention.index_head_weight = dsv4_model_l1_index_head;
    weights[1].attention.index_compress_kv_weight = dsv4_model_l1_index_compress_kv;
    weights[1].attention.index_compress_gate_weight = dsv4_model_l1_index_compress_gate;
    weights[1].attention.index_compress_position_bias = dsv4_model_l1_index_compress_bias;
    weights[1].attention.index_compress_norm_weight = dsv4_model_l1_index_compress_norm;
    fill_common_moe(&weights[1].moe,
                    dsv4_model_l1_router,
                    dsv4_model_l1_expert_gate,
                    dsv4_model_l1_expert_up,
                    dsv4_model_l1_expert_down,
                    dsv4_model_l1_shared_gate,
                    dsv4_model_l1_shared_up,
                    dsv4_model_l1_shared_down);
    weights[1].moe.correction_bias = dsv4_model_l1_bias;

    weights[2].attention_hc_fn = dsv4_model_l2_attn_hc_fn;
    weights[2].attention_hc_base = dsv4_model_l2_attn_hc_base;
    weights[2].attention_hc_scale = dsv4_model_l2_attn_hc_scale;
    weights[2].ffn_hc_fn = dsv4_model_l2_ffn_hc_fn;
    weights[2].ffn_hc_base = dsv4_model_l2_ffn_hc_base;
    weights[2].ffn_hc_scale = dsv4_model_l2_ffn_hc_scale;
    weights[2].input_norm_weight = dsv4_model_l2_input_norm;
    weights[2].post_attention_norm_weight = dsv4_model_l2_post_attn_norm;
    fill_common_attention(&weights[2].attention,
                          dsv4_model_l2_q_a, dsv4_model_l2_q_a_norm,
                          dsv4_model_l2_q_b, dsv4_model_l2_kv,
                          dsv4_model_l2_kv_norm, dsv4_model_l2_sinks,
                          dsv4_model_l2_o_a, dsv4_model_l2_o_b);
    weights[2].attention.compress_kv_weight = dsv4_model_l2_compress_kv;
    weights[2].attention.compress_gate_weight = dsv4_model_l2_compress_gate;
    weights[2].attention.compress_position_bias = dsv4_model_l2_compress_bias;
    weights[2].attention.compress_norm_weight = dsv4_model_l2_compress_norm;
    fill_common_moe(&weights[2].moe,
                    dsv4_model_l2_router,
                    dsv4_model_l2_expert_gate,
                    dsv4_model_l2_expert_up,
                    dsv4_model_l2_expert_down,
                    dsv4_model_l2_shared_gate,
                    dsv4_model_l2_shared_up,
                    dsv4_model_l2_shared_down);
    weights[2].moe.correction_bias = dsv4_model_l2_bias;
}

static DSV4ModelConfig make_model_config(DSV4DecoderLayerConfig layer_configs[L])
{
    DSV4ModelConfig config;
    config.vocab_size = V;
    config.hidden_size = H;
    config.hc_mult = HC;
    config.num_layers = L;
    config.max_seq_len = DSV4_MODEL_MAX_SEQ;
    config.hc_eps = DSV4_MODEL_HC_EPS;
    config.norm_eps = DSV4_MODEL_RMS_EPS;
    config.layer_configs = layer_configs;
    return config;
}

static DSV4ModelWeights make_model_weights(DSV4DecoderLayerWeights layer_weights[L])
{
    DSV4ModelWeights weights;
    weights.embedding_weight = dsv4_model_embedding;
    weights.layer_weights = layer_weights;
    weights.hc_head_fn = dsv4_model_hc_head_fn;
    weights.hc_head_base = dsv4_model_hc_head_base;
    weights.hc_head_scale = dsv4_model_hc_head_scale;
    weights.output_norm_weight = dsv4_model_output_norm;
    weights.lm_head_weight = dsv4_model_lm_head;
    return weights;
}

static void assert_close(const char *label, size_t position, size_t slot,
                         float got, float expected, float tolerance)
{
    if (!isfinite(got) || fabsf(got - expected) > tolerance) {
        fprintf(stderr, "FAIL %s pos=%zu slot=%zu got=%g expected=%g\n",
                label, position, slot, (double)got, (double)expected);
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

static void run_full_fixture(void)
{
    DSV4DecoderLayerConfig layer_configs[L];
    DSV4DecoderLayerWeights layer_weights[L];
    DSV4ModelConfig config;
    DSV4ModelWeights weights;
    DSV4ModelStatus status = DSV4_MODEL_OK;
    DSV4Model *model;
    size_t position;

    fill_layer_configs(layer_configs);
    fill_layer_weights(layer_weights);
    config = make_model_config(layer_configs);
    weights = make_model_weights(layer_weights);
    model = dsv4_model_create(&config, &status);
    if (model == NULL || status != DSV4_MODEL_OK) {
        fprintf(stderr, "FAIL model create status=%d\n", (int)status);
        ++failures;
        return;
    }

    for (position = 0u; position < DSV4_MODEL_STEPS; ++position) {
        float streams[HC * H];
        float logits[V];
        size_t i;
        for (i = 0u; i < HC * H; ++i) streams[i] = 99.0f;
        for (i = 0u; i < V; ++i) logits[i] = 98.0f;
        status = dsv4_model_step_f32(model, &weights,
                                     dsv4_model_token[position], streams, logits);
        if (status != DSV4_MODEL_OK) {
            fprintf(stderr, "FAIL model step=%zu status=%d\n", position, (int)status);
            ++failures;
            break;
        }
        for (i = 0u; i < HC * H; ++i)
            assert_close("streams", position, i, streams[i],
                         dsv4_model_expected_streams[position * HC * H + i], 8e-4f);
        for (i = 0u; i < V; ++i)
            assert_close("logits", position, i, logits[i],
                         dsv4_model_expected_logits[position * V + i], 2e-3f);
        if (argmax(logits, V) != dsv4_model_expected_argmax[position]) {
            fprintf(stderr, "FAIL argmax pos=%zu got=%u expected=%u\n",
                    position, argmax(logits, V), dsv4_model_expected_argmax[position]);
            ++failures;
        }
    }
    dsv4_model_destroy(model);
}

static void test_multilayer_rollback(void)
{
    DSV4DecoderLayerConfig layer_configs[L];
    DSV4DecoderLayerWeights layer_weights[L];
    DSV4ModelConfig config;
    DSV4ModelWeights weights;
    DSV4ModelStatus status = DSV4_MODEL_OK;
    DSV4Model *model;
    const float *saved_bias;
    float streams[HC * H];
    float logits[V];
    size_t i;

    fill_layer_configs(layer_configs);
    fill_layer_weights(layer_weights);
    config = make_model_config(layer_configs);
    weights = make_model_weights(layer_weights);
    model = dsv4_model_create(&config, &status);
    if (model == NULL || status != DSV4_MODEL_OK) {
        fprintf(stderr, "FAIL rollback create status=%d\n", (int)status);
        ++failures;
        return;
    }

    saved_bias = layer_weights[2].moe.correction_bias;
    layer_weights[2].moe.correction_bias = NULL;
    for (i = 0u; i < HC * H; ++i) streams[i] = 77.0f;
    for (i = 0u; i < V; ++i) logits[i] = 76.0f;
    status = dsv4_model_step_f32(model, &weights, dsv4_model_token[0], streams, logits);
    if (status != DSV4_MODEL_INVALID_ARGUMENT) {
        fprintf(stderr, "FAIL rollback late status=%d\n", (int)status);
        ++failures;
    }
    for (i = 0u; i < HC * H; ++i)
        if (streams[i] != 77.0f) {
            fprintf(stderr, "FAIL rollback streams mutated\n");
            ++failures;
            break;
        }
    for (i = 0u; i < V; ++i)
        if (logits[i] != 76.0f) {
            fprintf(stderr, "FAIL rollback logits mutated\n");
            ++failures;
            break;
        }

    layer_weights[2].moe.correction_bias = saved_bias;
    status = dsv4_model_step_f32(model, &weights, dsv4_model_token[0], streams, logits);
    if (status != DSV4_MODEL_OK) {
        fprintf(stderr, "FAIL rollback retry status=%d\n", (int)status);
        ++failures;
    } else {
        for (i = 0u; i < HC * H; ++i)
            assert_close("rollback-streams", 0u, i, streams[i],
                         dsv4_model_expected_streams[i], 8e-4f);
        for (i = 0u; i < V; ++i)
            assert_close("rollback-logits", 0u, i, logits[i],
                         dsv4_model_expected_logits[i], 2e-3f);
    }
    dsv4_model_destroy(model);
}

int main(void)
{
    run_full_fixture();
    test_multilayer_rollback();
    if (failures != 0) {
        fprintf(stderr, "%d model-shell parity failures\n", failures);
        return 1;
    }
    puts("DeepSeek native base-model shell parity: OK");
    return 0;
}

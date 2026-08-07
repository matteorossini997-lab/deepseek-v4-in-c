#ifndef DSV4_FULL_FIXTURE_BUILDERS_H
#define DSV4_FULL_FIXTURE_BUILDERS_H

#include "dsv4_full_model.h"
#include "dsv4_model_shell_vectors.h"
#include "dsv4_mtp_vectors.h"

#include <string.h>

static DSV4AttentionRuntimeConfig fixture_attention(size_t rate)
{
    DSV4AttentionRuntimeConfig c;
    memset(&c, 0, sizeof(c));
    c.hidden_size = DSV4_MODEL_HIDDEN;
    c.q_lora_rank = DSV4_MODEL_Q_RANK;
    c.num_heads = DSV4_MODEL_HEADS;
    c.head_dim = DSV4_MODEL_HEAD_DIM;
    c.rope_dim = DSV4_MODEL_ROPE_DIM;
    c.output_groups = DSV4_MODEL_GROUPS;
    c.output_rank = DSV4_MODEL_O_RANK;
    c.sliding_window = DSV4_MODEL_WINDOW;
    c.compression_rate = rate;
    c.index_num_heads = DSV4_MODEL_INDEX_HEADS;
    c.index_head_dim = DSV4_MODEL_INDEX_DIM;
    c.index_rope_dim = DSV4_MODEL_ROPE_DIM;
    c.index_top_k = DSV4_MODEL_INDEX_TOPK;
    c.rms_eps = DSV4_MODEL_RMS_EPS;
    c.rope_theta = DSV4_MODEL_ROPE_THETA;
    c.compress_rope_theta = DSV4_MODEL_COMPRESS_THETA;
    return c;
}

static DSV4MoEConfig fixture_moe(void)
{
    DSV4MoEConfig c;
    c.hidden_size = DSV4_MODEL_HIDDEN;
    c.intermediate_size = DSV4_MODEL_INTERMEDIATE;
    c.num_routed_experts = DSV4_MODEL_EXPERTS;
    c.top_k = DSV4_MODEL_MOE_TOPK;
    c.num_hash_layers = DSV4_MODEL_HASH_LAYERS;
    c.vocab_size = DSV4_MODEL_VOCAB;
    c.route_scale = DSV4_MODEL_ROUTE_SCALE;
    c.swiglu_limit = DSV4_MODEL_SWIGLU_LIMIT;
    return c;
}

static DSV4DecoderLayerConfig fixture_layer(uint32_t index, size_t rate)
{
    DSV4DecoderLayerConfig c;
    memset(&c, 0, sizeof(c));
    c.layer_idx = index;
    c.hidden_size = DSV4_MODEL_HIDDEN;
    c.hc_mult = DSV4_MODEL_HC;
    c.sinkhorn_iterations = DSV4_MODEL_SINKHORN_ITERS;
    c.hc_eps = DSV4_MODEL_HC_EPS;
    c.norm_eps = DSV4_MODEL_RMS_EPS;
    c.attention = fixture_attention(rate);
    c.moe = fixture_moe();
    return c;
}

static void fixture_attention_weights(DSV4AttentionRuntimeWeights *w,
                                      const float *qa, const float *qan,
                                      const float *qb, const float *kv,
                                      const float *kvn, const float *sinks,
                                      const float *oa, const float *ob)
{
    memset(w, 0, sizeof(*w));
    w->q_a_weight = qa; w->q_a_norm_weight = qan; w->q_b_weight = qb;
    w->kv_weight = kv; w->kv_norm_weight = kvn; w->sinks = sinks;
    w->o_a_weight = oa; w->o_b_weight = ob;
}

static void fixture_moe_weights(DSV4MoEWeights *w,
                                const float *router,
                                const float *gate, const float *up, const float *down,
                                const float *sgate, const float *sup, const float *sdown)
{
    memset(w, 0, sizeof(*w));
    w->router_weight = router;
    w->expert_gate_weight = gate; w->expert_up_weight = up; w->expert_down_weight = down;
    w->shared_gate_weight = sgate; w->shared_up_weight = sup; w->shared_down_weight = sdown;
}

static void fixture_base_layer_weights(DSV4DecoderLayerWeights w[DSV4_MODEL_LAYERS])
{
    memset(w, 0, DSV4_MODEL_LAYERS * sizeof(*w));

    w[0].attention_hc_fn=dsv4_model_l0_attn_hc_fn; w[0].attention_hc_base=dsv4_model_l0_attn_hc_base; w[0].attention_hc_scale=dsv4_model_l0_attn_hc_scale;
    w[0].ffn_hc_fn=dsv4_model_l0_ffn_hc_fn; w[0].ffn_hc_base=dsv4_model_l0_ffn_hc_base; w[0].ffn_hc_scale=dsv4_model_l0_ffn_hc_scale;
    w[0].input_norm_weight=dsv4_model_l0_input_norm; w[0].post_attention_norm_weight=dsv4_model_l0_post_attn_norm;
    fixture_attention_weights(&w[0].attention,dsv4_model_l0_q_a,dsv4_model_l0_q_a_norm,dsv4_model_l0_q_b,dsv4_model_l0_kv,dsv4_model_l0_kv_norm,dsv4_model_l0_sinks,dsv4_model_l0_o_a,dsv4_model_l0_o_b);
    fixture_moe_weights(&w[0].moe,dsv4_model_l0_router,dsv4_model_l0_expert_gate,dsv4_model_l0_expert_up,dsv4_model_l0_expert_down,dsv4_model_l0_shared_gate,dsv4_model_l0_shared_up,dsv4_model_l0_shared_down);
    w[0].moe.hash_table=(const uint32_t *)dsv4_model_l0_hash;

    w[1].attention_hc_fn=dsv4_model_l1_attn_hc_fn; w[1].attention_hc_base=dsv4_model_l1_attn_hc_base; w[1].attention_hc_scale=dsv4_model_l1_attn_hc_scale;
    w[1].ffn_hc_fn=dsv4_model_l1_ffn_hc_fn; w[1].ffn_hc_base=dsv4_model_l1_ffn_hc_base; w[1].ffn_hc_scale=dsv4_model_l1_ffn_hc_scale;
    w[1].input_norm_weight=dsv4_model_l1_input_norm; w[1].post_attention_norm_weight=dsv4_model_l1_post_attn_norm;
    fixture_attention_weights(&w[1].attention,dsv4_model_l1_q_a,dsv4_model_l1_q_a_norm,dsv4_model_l1_q_b,dsv4_model_l1_kv,dsv4_model_l1_kv_norm,dsv4_model_l1_sinks,dsv4_model_l1_o_a,dsv4_model_l1_o_b);
    w[1].attention.compress_kv_weight=dsv4_model_l1_compress_kv; w[1].attention.compress_gate_weight=dsv4_model_l1_compress_gate; w[1].attention.compress_position_bias=dsv4_model_l1_compress_bias; w[1].attention.compress_norm_weight=dsv4_model_l1_compress_norm;
    w[1].attention.index_q_weight=dsv4_model_l1_index_q; w[1].attention.index_head_weight=dsv4_model_l1_index_head;
    w[1].attention.index_compress_kv_weight=dsv4_model_l1_index_compress_kv; w[1].attention.index_compress_gate_weight=dsv4_model_l1_index_compress_gate; w[1].attention.index_compress_position_bias=dsv4_model_l1_index_compress_bias; w[1].attention.index_compress_norm_weight=dsv4_model_l1_index_compress_norm;
    fixture_moe_weights(&w[1].moe,dsv4_model_l1_router,dsv4_model_l1_expert_gate,dsv4_model_l1_expert_up,dsv4_model_l1_expert_down,dsv4_model_l1_shared_gate,dsv4_model_l1_shared_up,dsv4_model_l1_shared_down); w[1].moe.correction_bias=dsv4_model_l1_bias;

    w[2].attention_hc_fn=dsv4_model_l2_attn_hc_fn; w[2].attention_hc_base=dsv4_model_l2_attn_hc_base; w[2].attention_hc_scale=dsv4_model_l2_attn_hc_scale;
    w[2].ffn_hc_fn=dsv4_model_l2_ffn_hc_fn; w[2].ffn_hc_base=dsv4_model_l2_ffn_hc_base; w[2].ffn_hc_scale=dsv4_model_l2_ffn_hc_scale;
    w[2].input_norm_weight=dsv4_model_l2_input_norm; w[2].post_attention_norm_weight=dsv4_model_l2_post_attn_norm;
    fixture_attention_weights(&w[2].attention,dsv4_model_l2_q_a,dsv4_model_l2_q_a_norm,dsv4_model_l2_q_b,dsv4_model_l2_kv,dsv4_model_l2_kv_norm,dsv4_model_l2_sinks,dsv4_model_l2_o_a,dsv4_model_l2_o_b);
    w[2].attention.compress_kv_weight=dsv4_model_l2_compress_kv; w[2].attention.compress_gate_weight=dsv4_model_l2_compress_gate; w[2].attention.compress_position_bias=dsv4_model_l2_compress_bias; w[2].attention.compress_norm_weight=dsv4_model_l2_compress_norm;
    fixture_moe_weights(&w[2].moe,dsv4_model_l2_router,dsv4_model_l2_expert_gate,dsv4_model_l2_expert_up,dsv4_model_l2_expert_down,dsv4_model_l2_shared_gate,dsv4_model_l2_shared_up,dsv4_model_l2_shared_down); w[2].moe.correction_bias=dsv4_model_l2_bias;
}

static DSV4DecoderLayerWeights fixture_mtp_layer_weights(void)
{
    DSV4DecoderLayerWeights w;
    memset(&w,0,sizeof(w));
    w.attention_hc_fn=dsv4_mtp_decoder_attn_hc_fn; w.attention_hc_base=dsv4_mtp_decoder_attn_hc_base; w.attention_hc_scale=dsv4_mtp_decoder_attn_hc_scale;
    w.ffn_hc_fn=dsv4_mtp_decoder_ffn_hc_fn; w.ffn_hc_base=dsv4_mtp_decoder_ffn_hc_base; w.ffn_hc_scale=dsv4_mtp_decoder_ffn_hc_scale;
    w.input_norm_weight=dsv4_mtp_decoder_input_norm; w.post_attention_norm_weight=dsv4_mtp_decoder_post_attn_norm;
    fixture_attention_weights(&w.attention,dsv4_mtp_decoder_q_a,dsv4_mtp_decoder_q_a_norm,dsv4_mtp_decoder_q_b,dsv4_mtp_decoder_kv,dsv4_mtp_decoder_kv_norm,dsv4_mtp_decoder_sinks,dsv4_mtp_decoder_o_a,dsv4_mtp_decoder_o_b);
    fixture_moe_weights(&w.moe,dsv4_mtp_decoder_router,dsv4_mtp_decoder_expert_gate,dsv4_mtp_decoder_expert_up,dsv4_mtp_decoder_expert_down,dsv4_mtp_decoder_shared_gate,dsv4_mtp_decoder_shared_up,dsv4_mtp_decoder_shared_down);
    w.moe.correction_bias=dsv4_mtp_decoder_bias;
    return w;
}

static DSV4FullModelConfig fixture_full_config(DSV4DecoderLayerConfig base_layers[DSV4_MODEL_LAYERS])
{
    DSV4FullModelConfig c;
    base_layers[0]=fixture_layer(0u,0u); base_layers[1]=fixture_layer(1u,4u); base_layers[2]=fixture_layer(2u,128u);
    c.base.vocab_size=DSV4_MODEL_VOCAB; c.base.hidden_size=DSV4_MODEL_HIDDEN; c.base.hc_mult=DSV4_MODEL_HC; c.base.num_layers=DSV4_MODEL_LAYERS; c.base.max_seq_len=DSV4_MODEL_MAX_SEQ; c.base.hc_eps=DSV4_MODEL_HC_EPS; c.base.norm_eps=DSV4_MODEL_RMS_EPS; c.base.layer_configs=base_layers;
    c.mtp.vocab_size=DSV4_MTP_VOCAB; c.mtp.hidden_size=DSV4_MTP_HIDDEN; c.mtp.hc_mult=DSV4_MTP_HC; c.mtp.max_seq_len=DSV4_MTP_MAX_SEQ; c.mtp.hc_eps=DSV4_MTP_HC_EPS; c.mtp.norm_eps=DSV4_MTP_RMS_EPS; c.mtp.decoder=fixture_layer(DSV4_MTP_LAYER_IDX,0u);
    return c;
}

static DSV4FullModelWeights fixture_full_weights(DSV4DecoderLayerWeights base_layers[DSV4_MODEL_LAYERS])
{
    DSV4FullModelWeights w;
    memset(&w,0,sizeof(w));
    fixture_base_layer_weights(base_layers);
    w.base.embedding_weight=dsv4_model_embedding; w.base.layer_weights=base_layers; w.base.hc_head_fn=dsv4_model_hc_head_fn; w.base.hc_head_base=dsv4_model_hc_head_base; w.base.hc_head_scale=dsv4_model_hc_head_scale; w.base.output_norm_weight=dsv4_model_output_norm; w.base.lm_head_weight=dsv4_model_lm_head;
    w.mtp.embedding_weight=dsv4_model_embedding; w.mtp.lm_head_weight=dsv4_model_lm_head; w.mtp.embedding_norm_weight=dsv4_mtp_embedding_norm; w.mtp.hidden_norm_weight=dsv4_mtp_hidden_norm; w.mtp.embedding_proj_weight=dsv4_mtp_embedding_proj; w.mtp.hidden_proj_weight=dsv4_mtp_hidden_proj; w.mtp.decoder=fixture_mtp_layer_weights(); w.mtp.hc_head_fn=dsv4_mtp_hc_head_fn; w.mtp.hc_head_base=dsv4_mtp_hc_head_base; w.mtp.hc_head_scale=dsv4_mtp_hc_head_scale; w.mtp.output_norm_weight=dsv4_mtp_output_norm;
    return w;
}

#endif

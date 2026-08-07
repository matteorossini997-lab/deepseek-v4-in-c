#ifndef DSV4_ATTENTION_RUNTIME_H
#define DSV4_ATTENTION_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4AttentionRuntimeStatus {
    DSV4_AR_OK = 0,
    DSV4_AR_INVALID_ARGUMENT = 1,
    DSV4_AR_NONFINITE = 2,
    DSV4_AR_OVERFLOW = 3,
    DSV4_AR_ALLOCATION_FAILED = 4,
    DSV4_AR_DEPENDENCY_FAILED = 5
} DSV4AttentionRuntimeStatus;

typedef struct DSV4AttentionRuntimeConfig {
    size_t hidden_size;
    size_t q_lora_rank;
    size_t num_heads;
    size_t head_dim;
    size_t rope_dim;
    size_t output_groups;
    size_t output_rank;
    size_t sliding_window;
    size_t compression_rate;
    size_t index_num_heads;
    size_t index_head_dim;
    size_t index_rope_dim;
    size_t index_top_k;
    float rms_eps;
    float rope_theta;
    float compress_rope_theta;
} DSV4AttentionRuntimeConfig;

typedef struct DSV4AttentionRuntimeWeights {
    const float *q_a_weight;
    const float *q_a_norm_weight;
    const float *q_b_weight;
    const float *kv_weight;
    const float *kv_norm_weight;
    const float *sinks;
    const float *o_a_weight;
    const float *o_b_weight;

    /* Required for compression_rate 4 or 128. */
    const float *compress_kv_weight;
    const float *compress_gate_weight;
    const float *compress_position_bias;
    const float *compress_norm_weight;

    /* Required only for compression_rate 4 (CSA). */
    const float *index_q_weight;
    const float *index_head_weight;
    const float *index_compress_kv_weight;
    const float *index_compress_gate_weight;
    const float *index_compress_position_bias;
    const float *index_compress_norm_weight;
} DSV4AttentionRuntimeWeights;

typedef struct DSV4AttentionRuntimeTrace {
    size_t raw_cache_length;
    size_t compressed_count;
    size_t indexer_compressed_count;
} DSV4AttentionRuntimeTrace;

typedef struct DSV4AttentionRuntime DSV4AttentionRuntime;

/*
 * Creates a correctness-first FP32 runtime for batch size one and one token per
 * step. The runtime owns mutable KV/compressor/indexer state; weight pointers
 * passed to step remain caller-owned and immutable.
 *
 * compression_rate must be one of 0 (sliding), 4 (CSA) or 128 (HCA).
 */
DSV4AttentionRuntime *dsv4_attention_runtime_create(
    const DSV4AttentionRuntimeConfig *config,
    DSV4AttentionRuntimeStatus *out_status);

void dsv4_attention_runtime_destroy(DSV4AttentionRuntime *runtime);

/* Resets position and every owned cache. On allocation failure the old state is
 * retained unchanged. */
DSV4AttentionRuntimeStatus dsv4_attention_runtime_reset(
    DSV4AttentionRuntime *runtime);

/*
 * Executes the learned-projection + incremental-state equivalent of
 * MiniAttention.step for one [1, 1, hidden_size] token represented here as one
 * hidden_size vector.
 *
 * In CSA mode out_selected_indices must have capacity >= index_top_k. In the
 * other modes it may be NULL and selected_count is published as zero.
 *
 * Publication is transactional: on any non-OK status runtime state,
 * out_hidden, selected outputs and out_trace remain unchanged.
 */
DSV4AttentionRuntimeStatus dsv4_attention_runtime_step_f32(
    DSV4AttentionRuntime *runtime,
    const DSV4AttentionRuntimeWeights *weights,
    int64_t position,
    const float *hidden,
    float *out_hidden,
    size_t *out_selected_indices,
    size_t out_selected_capacity,
    size_t *out_selected_count,
    DSV4AttentionRuntimeTrace *out_trace);

const char *dsv4_attention_runtime_status_string(
    DSV4AttentionRuntimeStatus status);

#ifdef __cplusplus
}
#endif

#endif

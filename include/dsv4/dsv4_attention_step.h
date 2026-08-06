#ifndef DSV4_ATTENTION_STEP_H
#define DSV4_ATTENTION_STEP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4AttentionStepStatus {
    DSV4_AS_OK = 0,
    DSV4_AS_INVALID_ARGUMENT = 1,
    DSV4_AS_NONFINITE = 2,
    DSV4_AS_OVERFLOW = 3,
    DSV4_AS_ALLOCATION_FAILED = 4,
    DSV4_AS_DEPENDENCY_FAILED = 5
} DSV4AttentionStepStatus;

typedef struct DSV4AttentionStepConfig {
    size_t num_heads;
    size_t head_dim;
    size_t rope_dim;
    size_t raw_tokens;
    size_t compressed_tokens;
    size_t index_num_heads;
    size_t index_head_dim;
    size_t index_rope_dim;
    size_t index_top_k;
    size_t output_groups;
    size_t output_rank;
    size_t hidden_size;
    float rope_theta;
    int use_sparse_index;
} DSV4AttentionStepConfig;

/*
 * Executes the post-projection portion of one attention step.
 *
 * q and index_q are unrotated. raw_kv, compressed_kv and index_compressed
 * already carry their canonical per-entry RoPE. CSA mode selects compressed
 * entries through the sparse index. Non-indexed mode appends all compressed
 * entries. The result is inverse-rotated, passed through grouped o_a and dense
 * o_b projections, and published atomically with selected indices.
 *
 * Shapes:
 *   q                  [num_heads, head_dim]
 *   raw_kv             [raw_tokens, head_dim]
 *   compressed_kv      [compressed_tokens, head_dim]
 *   index_q            [index_num_heads, index_head_dim]
 *   index_compressed   [compressed_tokens, index_head_dim]
 *   index_head_weights [index_num_heads]
 *   sinks              [num_heads]
 *   o_a_weight         [output_groups, output_rank,
 *                       num_heads * head_dim / output_groups]
 *   o_b_weight         [hidden_size, output_groups * output_rank]
 */
DSV4AttentionStepStatus dsv4_attention_postprojected_step_f32(
    const DSV4AttentionStepConfig *config,
    int64_t position,
    const float *q,
    const float *raw_kv,
    const float *compressed_kv,
    const float *index_q,
    const float *index_compressed,
    const float *index_head_weights,
    const float *sinks,
    const float *o_a_weight,
    const float *o_b_weight,
    float *out_hidden,
    size_t *out_selected_indices,
    size_t *out_selected_count);

const char *dsv4_attention_step_status_string(DSV4AttentionStepStatus status);

#ifdef __cplusplus
}
#endif

#endif

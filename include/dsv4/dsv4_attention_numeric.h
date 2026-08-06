#ifndef DSV4_ATTENTION_NUMERIC_H
#define DSV4_ATTENTION_NUMERIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4AttentionNumericStatus {
    DSV4_AN_OK = 0,
    DSV4_AN_INVALID_ARGUMENT = 1,
    DSV4_AN_NONFINITE = 2,
    DSV4_AN_OVERFLOW = 3,
    DSV4_AN_ALLOCATION_FAILED = 4
} DSV4AttentionNumericStatus;

typedef struct DSV4CompressorNumericState {
    size_t rate;
    size_t output_dim;
    size_t rope_dim;
    int overlap;
    float rms_eps;
    float theta;
    uint64_t emitted;
    int has_previous;
    float *previous_ca_kv;
    float *previous_ca_gate;
} DSV4CompressorNumericState;

/* Applies interleaved RoPE to the trailing rope_dim values of every row. */
DSV4AttentionNumericStatus dsv4_partial_rope_f32(
    float *values,
    size_t row_count,
    size_t dim,
    size_t rope_dim,
    int64_t position,
    float theta);

/*
 * Computes sink-aware attention for one query position. q is [heads, head_dim],
 * kv is shared across heads with shape [tokens, head_dim], and out is
 * [heads, head_dim]. The sink contributes probability mass but no value vector.
 */
DSV4AttentionNumericStatus dsv4_sink_attention_shared_kv_f32(
    const float *q,
    const float *kv,
    const float *sinks,
    size_t heads,
    size_t tokens,
    size_t head_dim,
    float scale,
    float *out);

/* Initializes a caller-owned, allocation-free compressor state. */
DSV4AttentionNumericStatus dsv4_compressor_numeric_init(
    DSV4CompressorNumericState *state,
    size_t rate,
    size_t output_dim,
    size_t rope_dim,
    int overlap,
    float rms_eps,
    float theta,
    float *previous_ca_kv,
    float *previous_ca_gate);

/*
 * Compresses one complete projected window. For overlap mode, projected arrays
 * are [rate, 2 * output_dim] split into CA then CB. Otherwise they are
 * [rate, output_dim]. position_bias has the same shape. State publication is
 * transactional: emitted/previous CA are updated only after a valid output.
 */
DSV4AttentionNumericStatus dsv4_compress_projected_window_f32(
    DSV4CompressorNumericState *state,
    const float *projected_kv,
    const float *projected_gate,
    const float *position_bias,
    const float *norm_weight,
    float *out);

const char *dsv4_attention_numeric_status_string(DSV4AttentionNumericStatus status);

#ifdef __cplusplus
}
#endif

#endif

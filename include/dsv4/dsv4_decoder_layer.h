#ifndef DSV4_DECODER_LAYER_H
#define DSV4_DECODER_LAYER_H

#include "dsv4_attention_runtime.h"
#include "dsv4_moe.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4DecoderLayerStatus {
    DSV4_DL_OK = 0,
    DSV4_DL_INVALID_ARGUMENT = 1,
    DSV4_DL_NONFINITE = 2,
    DSV4_DL_OVERFLOW = 3,
    DSV4_DL_ALLOCATION_FAILED = 4,
    DSV4_DL_DEPENDENCY_FAILED = 5
} DSV4DecoderLayerStatus;

typedef struct DSV4DecoderLayerConfig {
    uint32_t layer_idx;
    size_t hidden_size;
    size_t hc_mult;
    int sinkhorn_iterations;
    float hc_eps;
    float norm_eps;
    DSV4AttentionRuntimeConfig attention;
    DSV4MoEConfig moe;
} DSV4DecoderLayerConfig;

typedef struct DSV4DecoderLayerWeights {
    /* Hyper-Connection fn is [(2 + hc_mult) * hc_mult,
     * hc_mult * hidden_size], base has (2 + hc_mult) * hc_mult values and
     * scale has three values. */
    const float *attention_hc_fn;
    const float *attention_hc_base;
    const float *attention_hc_scale;
    const float *ffn_hc_fn;
    const float *ffn_hc_base;
    const float *ffn_hc_scale;

    /* [hidden_size] weighted RMSNorm scales. */
    const float *input_norm_weight;
    const float *post_attention_norm_weight;

    DSV4AttentionRuntimeWeights attention;
    DSV4MoEWeights moe;
} DSV4DecoderLayerWeights;

typedef struct DSV4DecoderLayerTrace {
    DSV4AttentionRuntimeTrace attention;
} DSV4DecoderLayerTrace;

typedef struct DSV4DecoderLayer DSV4DecoderLayer;

/* Owns one mutable attention runtime. All weights remain caller-owned. */
DSV4DecoderLayer *dsv4_decoder_layer_create(
    const DSV4DecoderLayerConfig *config,
    DSV4DecoderLayerStatus *out_status);

void dsv4_decoder_layer_destroy(DSV4DecoderLayer *layer);

DSV4DecoderLayerStatus dsv4_decoder_layer_reset(DSV4DecoderLayer *layer);

/*
 * Executes one DecoderLayer.step for [hc_mult, hidden_size] streams.
 *
 * The whole layer is transactional. Attention runs on a deep clone and that
 * clone is installed only after the second HC mix succeeds, so a late MoE
 * failure cannot advance KV/compressor/indexer state.
 *
 * CSA layers require out_attention_indices capacity >= attention.index_top_k.
 * Route buffers always require capacity >= moe.top_k.
 */
DSV4DecoderLayerStatus dsv4_decoder_layer_step_f32(
    DSV4DecoderLayer *layer,
    const DSV4DecoderLayerWeights *weights,
    int64_t position,
    uint64_t token_id,
    const float *streams,
    float *out_streams,
    size_t *out_attention_indices,
    size_t attention_capacity,
    size_t *out_attention_count,
    uint32_t *out_route_indices,
    float *out_route_weights,
    size_t route_capacity,
    DSV4DecoderLayerTrace *out_trace);

const char *dsv4_decoder_layer_status_string(DSV4DecoderLayerStatus status);

#ifdef __cplusplus
}
#endif

#endif

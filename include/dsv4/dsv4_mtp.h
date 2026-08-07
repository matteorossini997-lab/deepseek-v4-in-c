#ifndef DSV4_MTP_H
#define DSV4_MTP_H

#include "dsv4_decoder_layer.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4MTPStatus {
    DSV4_MTP_OK = 0,
    DSV4_MTP_INVALID_ARGUMENT = 1,
    DSV4_MTP_NONFINITE = 2,
    DSV4_MTP_OVERFLOW = 3,
    DSV4_MTP_ALLOCATION_FAILED = 4,
    DSV4_MTP_DEPENDENCY_FAILED = 5
} DSV4MTPStatus;

typedef struct DSV4MTPConfig {
    size_t vocab_size;
    size_t hidden_size;
    size_t hc_mult;
    size_t max_seq_len;
    float hc_eps;
    float norm_eps;
    DSV4DecoderLayerConfig decoder;
} DSV4MTPConfig;

typedef struct DSV4MTPWeights {
    /* Shared with the base model. */
    const float *embedding_weight; /* [vocab_size, hidden_size] */
    const float *lm_head_weight;   /* [vocab_size, hidden_size] */

    /* MTP-owned pre-block projections. */
    const float *embedding_norm_weight; /* [hidden_size] */
    const float *hidden_norm_weight;    /* [hidden_size] */
    const float *embedding_proj_weight; /* [hidden_size, hidden_size] */
    const float *hidden_proj_weight;    /* [hidden_size, hidden_size] */

    DSV4DecoderLayerWeights decoder;

    /* MTP HyperHead and final norm. */
    const float *hc_head_fn;    /* [hc_mult, hc_mult * hidden_size] */
    const float *hc_head_base;  /* [hc_mult] */
    const float *hc_head_scale; /* [1] */
    const float *output_norm_weight; /* [hidden_size] */
} DSV4MTPWeights;

typedef struct DSV4MTPTrace {
    DSV4DecoderLayerTrace decoder;
} DSV4MTPTrace;

typedef struct DSV4MTP DSV4MTP;

DSV4MTP *dsv4_mtp_create(const DSV4MTPConfig *config, DSV4MTPStatus *out_status);
void dsv4_mtp_destroy(DSV4MTP *mtp);
DSV4MTPStatus dsv4_mtp_reset(DSV4MTP *mtp);

/*
 * Executes one incremental MTP token using final base-model HC streams.
 * The runtime owns only the extra decoder state and absolute position.
 *
 * On success:
 *   out_streams: [hc_mult, hidden_size] post-MTP decoder streams;
 *   out_logits:  [vocab_size] shared LM-head logits;
 *   route arrays: decoder learned-router expert/weight pairs.
 *
 * Publication is transactional: any failure leaves runtime position/state and
 * all caller outputs unchanged, so the same token can be retried.
 */
DSV4MTPStatus dsv4_mtp_step_f32(
    DSV4MTP *mtp,
    const DSV4MTPWeights *weights,
    uint64_t token_id,
    const float *base_streams,
    float *out_streams,
    float *out_logits,
    uint32_t *out_route_indices,
    float *out_route_weights,
    size_t route_capacity,
    DSV4MTPTrace *out_trace);

const char *dsv4_mtp_status_string(DSV4MTPStatus status);

#ifdef __cplusplus
}
#endif

#endif

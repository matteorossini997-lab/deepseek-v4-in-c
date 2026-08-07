#ifndef DSV4_MODEL_H
#define DSV4_MODEL_H

#include "dsv4_decoder_layer.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4ModelStatus {
    DSV4_MODEL_OK = 0,
    DSV4_MODEL_INVALID_ARGUMENT = 1,
    DSV4_MODEL_NONFINITE = 2,
    DSV4_MODEL_OVERFLOW = 3,
    DSV4_MODEL_ALLOCATION_FAILED = 4,
    DSV4_MODEL_DEPENDENCY_FAILED = 5
} DSV4ModelStatus;

typedef struct DSV4ModelConfig {
    size_t vocab_size;
    size_t hidden_size;
    size_t hc_mult;
    size_t num_layers;
    size_t max_seq_len;
    float hc_eps;
    float norm_eps;
    const DSV4DecoderLayerConfig *layer_configs;
} DSV4ModelConfig;

typedef struct DSV4ModelWeights {
    /* [vocab_size, hidden_size] */
    const float *embedding_weight;

    /* [num_layers], with each nested pointer caller-owned and immutable. */
    const DSV4DecoderLayerWeights *layer_weights;

    /* HyperHead: fn [hc_mult, hc_mult * hidden_size], base [hc_mult], scale [1]. */
    const float *hc_head_fn;
    const float *hc_head_base;
    const float *hc_head_scale;

    /* Final RMSNorm [hidden_size] and LM head [vocab_size, hidden_size]. */
    const float *output_norm_weight;
    const float *lm_head_weight;
} DSV4ModelWeights;

typedef struct DSV4Model DSV4Model;

/* Copies layer configs and creates one owned decoder runtime per base layer. */
DSV4Model *dsv4_model_create(
    const DSV4ModelConfig *config,
    DSV4ModelStatus *out_status);

void dsv4_model_destroy(DSV4Model *model);

/* Transactionally recreates all base-layer state at position zero. */
DSV4ModelStatus dsv4_model_reset(DSV4Model *model);

/*
 * Executes one base-model token at the internally owned absolute position.
 * MTP is deliberately not executed here.
 *
 * On success:
 *   out_streams: [hc_mult, hidden_size], final base-layer streams for MTP reuse;
 *   out_logits:  [vocab_size], final normalized LM-head logits.
 *
 * The full model step is transactional. Every decoder layer runs on a deep
 * clone, and all cloned states plus next_position are committed only after
 * HyperHead, final RMSNorm and every LM-head logit are finite.
 */
DSV4ModelStatus dsv4_model_step_f32(
    DSV4Model *model,
    const DSV4ModelWeights *weights,
    uint64_t token_id,
    float *out_streams,
    float *out_logits);

const char *dsv4_model_status_string(DSV4ModelStatus status);

#ifdef __cplusplus
}
#endif

#endif

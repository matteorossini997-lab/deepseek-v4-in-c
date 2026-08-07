#ifndef DSV4_FULL_MODEL_H
#define DSV4_FULL_MODEL_H

#include "dsv4_model.h"
#include "dsv4_mtp.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4FullModelStatus {
    DSV4_FULL_OK = 0,
    DSV4_FULL_INVALID_ARGUMENT = 1,
    DSV4_FULL_NONFINITE = 2,
    DSV4_FULL_OVERFLOW = 3,
    DSV4_FULL_ALLOCATION_FAILED = 4,
    DSV4_FULL_DEPENDENCY_FAILED = 5
} DSV4FullModelStatus;

typedef struct DSV4FullModelConfig {
    DSV4ModelConfig base;
    DSV4MTPConfig mtp;
} DSV4FullModelConfig;

typedef struct DSV4FullModelWeights {
    DSV4ModelWeights base;
    DSV4MTPWeights mtp;
} DSV4FullModelWeights;

typedef struct DSV4FullModelTrace {
    DSV4MTPTrace mtp;
} DSV4FullModelTrace;

typedef struct DSV4FullModel DSV4FullModel;

DSV4FullModel *dsv4_full_model_create(
    const DSV4FullModelConfig *config,
    DSV4FullModelStatus *out_status);

void dsv4_full_model_destroy(DSV4FullModel *model);
DSV4FullModelStatus dsv4_full_model_reset(DSV4FullModel *model);

/* Atomically stages one base step and the matching MTP step. */
DSV4FullModelStatus dsv4_full_model_step_f32(
    DSV4FullModel *model,
    const DSV4FullModelWeights *weights,
    uint64_t token_id,
    float *out_base_streams,
    float *out_base_logits,
    float *out_mtp_streams,
    float *out_mtp_logits,
    uint32_t *out_mtp_route_indices,
    float *out_mtp_route_weights,
    size_t route_capacity,
    DSV4FullModelTrace *out_trace);

/*
 * Resets generation state transactionally, consumes the prompt, and appends up
 * to max_new_tokens chosen by argmax(base logits). MTP is evaluated but never
 * changes the standard token decision. out_tokens contains prompt+completion.
 * If stop_on_eos is nonzero, generation stops after appending eos_id.
 */
DSV4FullModelStatus dsv4_full_model_generate_greedy(
    DSV4FullModel *model,
    const DSV4FullModelWeights *weights,
    const uint64_t *prompt_tokens,
    size_t prompt_len,
    size_t max_new_tokens,
    int stop_on_eos,
    uint64_t eos_id,
    uint64_t *out_tokens,
    size_t out_capacity,
    size_t *out_length);

const char *dsv4_full_model_status_string(DSV4FullModelStatus status);

#ifdef __cplusplus
}
#endif

#endif

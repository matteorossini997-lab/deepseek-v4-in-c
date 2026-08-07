#include "dsv4_full_model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct DSV4FullModel {
    DSV4FullModelConfig config;
    DSV4DecoderLayerConfig *base_layer_configs;
    DSV4Model *base;
    DSV4MTP *mtp;
};

static int checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL) return 0;
    if (a != 0u && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int checked_add_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static DSV4FullModelStatus map_base(DSV4ModelStatus status)
{
    switch (status) {
        case DSV4_MODEL_OK: return DSV4_FULL_OK;
        case DSV4_MODEL_INVALID_ARGUMENT: return DSV4_FULL_INVALID_ARGUMENT;
        case DSV4_MODEL_NONFINITE: return DSV4_FULL_NONFINITE;
        case DSV4_MODEL_OVERFLOW: return DSV4_FULL_OVERFLOW;
        case DSV4_MODEL_ALLOCATION_FAILED: return DSV4_FULL_ALLOCATION_FAILED;
        case DSV4_MODEL_DEPENDENCY_FAILED: return DSV4_FULL_DEPENDENCY_FAILED;
        default: return DSV4_FULL_DEPENDENCY_FAILED;
    }
}

static DSV4FullModelStatus map_mtp(DSV4MTPStatus status)
{
    switch (status) {
        case DSV4_MTP_OK: return DSV4_FULL_OK;
        case DSV4_MTP_INVALID_ARGUMENT: return DSV4_FULL_INVALID_ARGUMENT;
        case DSV4_MTP_NONFINITE: return DSV4_FULL_NONFINITE;
        case DSV4_MTP_OVERFLOW: return DSV4_FULL_OVERFLOW;
        case DSV4_MTP_ALLOCATION_FAILED: return DSV4_FULL_ALLOCATION_FAILED;
        case DSV4_MTP_DEPENDENCY_FAILED: return DSV4_FULL_DEPENDENCY_FAILED;
        default: return DSV4_FULL_DEPENDENCY_FAILED;
    }
}

static DSV4FullModelStatus alloc_floats(float **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_FULL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_FULL_OK;
    if (!checked_mul_size(count, sizeof(float), &bytes)) return DSV4_FULL_OVERFLOW;
    *out = (float *)malloc(bytes);
    return *out == NULL ? DSV4_FULL_ALLOCATION_FAILED : DSV4_FULL_OK;
}

static DSV4FullModelStatus alloc_u32(uint32_t **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_FULL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_FULL_OK;
    if (!checked_mul_size(count, sizeof(uint32_t), &bytes)) return DSV4_FULL_OVERFLOW;
    *out = (uint32_t *)malloc(bytes);
    return *out == NULL ? DSV4_FULL_ALLOCATION_FAILED : DSV4_FULL_OK;
}

static DSV4FullModelStatus alloc_u64(uint64_t **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_FULL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_FULL_OK;
    if (!checked_mul_size(count, sizeof(uint64_t), &bytes)) return DSV4_FULL_OVERFLOW;
    *out = (uint64_t *)malloc(bytes);
    return *out == NULL ? DSV4_FULL_ALLOCATION_FAILED : DSV4_FULL_OK;
}

static int same_float(float a, float b)
{
    return isfinite(a) && isfinite(b) && a == b;
}

static DSV4FullModelStatus validate_config(const DSV4FullModelConfig *config)
{
    size_t layer_bytes;
    if (config == NULL || config->base.layer_configs == NULL ||
        config->base.num_layers == 0u || config->base.num_layers > UINT32_MAX)
        return DSV4_FULL_INVALID_ARGUMENT;

    if (config->base.vocab_size != config->mtp.vocab_size ||
        config->base.hidden_size != config->mtp.hidden_size ||
        config->base.hc_mult != config->mtp.hc_mult ||
        config->base.max_seq_len != config->mtp.max_seq_len ||
        !same_float(config->base.hc_eps, config->mtp.hc_eps) ||
        !same_float(config->base.norm_eps, config->mtp.norm_eps) ||
        config->mtp.decoder.layer_idx != (uint32_t)config->base.num_layers ||
        config->mtp.decoder.hidden_size != config->base.hidden_size ||
        config->mtp.decoder.hc_mult != config->base.hc_mult ||
        config->mtp.decoder.moe.vocab_size != config->base.vocab_size)
        return DSV4_FULL_INVALID_ARGUMENT;

    if (!checked_mul_size(config->base.num_layers,
                          sizeof(DSV4DecoderLayerConfig), &layer_bytes))
        return DSV4_FULL_OVERFLOW;
    (void)layer_bytes;
    return DSV4_FULL_OK;
}

static DSV4FullModelStatus validate_weights(const DSV4FullModel *model,
                                             const DSV4FullModelWeights *weights)
{
    if (model == NULL || weights == NULL ||
        weights->base.embedding_weight == NULL ||
        weights->base.lm_head_weight == NULL ||
        weights->mtp.embedding_weight == NULL ||
        weights->mtp.lm_head_weight == NULL)
        return DSV4_FULL_INVALID_ARGUMENT;
    if (weights->base.embedding_weight != weights->mtp.embedding_weight ||
        weights->base.lm_head_weight != weights->mtp.lm_head_weight)
        return DSV4_FULL_INVALID_ARGUMENT;
    return DSV4_FULL_OK;
}

DSV4FullModel *dsv4_full_model_create(const DSV4FullModelConfig *config,
                                      DSV4FullModelStatus *out_status)
{
    DSV4FullModel *model = NULL;
    DSV4FullModelStatus status = validate_config(config);
    DSV4ModelStatus base_status;
    DSV4MTPStatus mtp_status;
    size_t layer_bytes;

    if (status != DSV4_FULL_OK) goto done;
    model = (DSV4FullModel *)calloc(1u, sizeof(*model));
    if (model == NULL) {
        status = DSV4_FULL_ALLOCATION_FAILED;
        goto done;
    }
    if (!checked_mul_size(config->base.num_layers,
                          sizeof(*model->base_layer_configs), &layer_bytes)) {
        status = DSV4_FULL_OVERFLOW;
        goto fail;
    }
    model->base_layer_configs =
        (DSV4DecoderLayerConfig *)malloc(layer_bytes);
    if (model->base_layer_configs == NULL) {
        status = DSV4_FULL_ALLOCATION_FAILED;
        goto fail;
    }
    memcpy(model->base_layer_configs, config->base.layer_configs, layer_bytes);
    model->config = *config;
    model->config.base.layer_configs = model->base_layer_configs;

    model->base = dsv4_model_create(&model->config.base, &base_status);
    if (model->base == NULL) {
        status = map_base(base_status);
        goto fail;
    }
    model->mtp = dsv4_mtp_create(&model->config.mtp, &mtp_status);
    if (model->mtp == NULL) {
        status = map_mtp(mtp_status);
        goto fail;
    }
    status = DSV4_FULL_OK;
    goto done;

fail:
    dsv4_model_destroy(model->base);
    dsv4_mtp_destroy(model->mtp);
    free(model->base_layer_configs);
    free(model);
    model = NULL;
done:
    if (out_status != NULL) *out_status = status;
    return model;
}

void dsv4_full_model_destroy(DSV4FullModel *model)
{
    if (model == NULL) return;
    dsv4_model_destroy(model->base);
    dsv4_mtp_destroy(model->mtp);
    free(model->base_layer_configs);
    free(model);
}

DSV4FullModelStatus dsv4_full_model_reset(DSV4FullModel *model)
{
    DSV4Model *fresh_base = NULL;
    DSV4MTP *fresh_mtp = NULL;
    DSV4ModelStatus base_status;
    DSV4MTPStatus mtp_status;
    if (model == NULL || model->base_layer_configs == NULL)
        return DSV4_FULL_INVALID_ARGUMENT;

    fresh_base = dsv4_model_create(&model->config.base, &base_status);
    if (fresh_base == NULL) return map_base(base_status);
    fresh_mtp = dsv4_mtp_create(&model->config.mtp, &mtp_status);
    if (fresh_mtp == NULL) {
        DSV4FullModelStatus status = map_mtp(mtp_status);
        dsv4_model_destroy(fresh_base);
        return status;
    }

    dsv4_model_destroy(model->base);
    dsv4_mtp_destroy(model->mtp);
    model->base = fresh_base;
    model->mtp = fresh_mtp;
    return DSV4_FULL_OK;
}

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
    DSV4FullModelTrace *out_trace)
{
    DSV4FullModelStatus status;
    DSV4Model *base_work = NULL;
    DSV4MTP *mtp_work = NULL;
    DSV4ModelStatus base_status;
    DSV4MTPStatus mtp_status;
    float *base_streams = NULL;
    float *base_logits = NULL;
    float *mtp_streams = NULL;
    float *mtp_logits = NULL;
    uint32_t *route_indices = NULL;
    float *route_weights = NULL;
    DSV4MTPTrace mtp_trace;
    size_t stream_count;
    size_t route_count;

    if (model == NULL || model->base == NULL || model->mtp == NULL ||
        out_base_streams == NULL || out_base_logits == NULL ||
        out_mtp_streams == NULL || out_mtp_logits == NULL ||
        out_mtp_route_indices == NULL || out_mtp_route_weights == NULL ||
        out_trace == NULL || token_id >= model->config.base.vocab_size)
        return DSV4_FULL_INVALID_ARGUMENT;
    status = validate_weights(model, weights);
    if (status != DSV4_FULL_OK) return status;
    route_count = model->config.mtp.decoder.moe.top_k;
    if (route_capacity < route_count) return DSV4_FULL_INVALID_ARGUMENT;
    if (!checked_mul_size(model->config.base.hc_mult,
                          model->config.base.hidden_size, &stream_count))
        return DSV4_FULL_OVERFLOW;

    status = alloc_floats(&base_streams, stream_count);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&base_logits, model->config.base.vocab_size);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&mtp_streams, stream_count);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&mtp_logits, model->config.base.vocab_size);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_u32(&route_indices, route_count);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&route_weights, route_count);
    if (status != DSV4_FULL_OK) goto cleanup;

    base_work = dsv4_model_clone(model->base, &base_status);
    if (base_work == NULL) {
        status = map_base(base_status);
        goto cleanup;
    }
    mtp_work = dsv4_mtp_clone(model->mtp, &mtp_status);
    if (mtp_work == NULL) {
        status = map_mtp(mtp_status);
        goto cleanup;
    }

    base_status = dsv4_model_step_f32(
        base_work, &weights->base, token_id, base_streams, base_logits);
    status = map_base(base_status);
    if (status != DSV4_FULL_OK) goto cleanup;

    mtp_status = dsv4_mtp_step_f32(
        mtp_work, &weights->mtp, token_id, base_streams,
        mtp_streams, mtp_logits, route_indices, route_weights,
        route_count, &mtp_trace);
    status = map_mtp(mtp_status);
    if (status != DSV4_FULL_OK) goto cleanup;

    {
        DSV4Model *old_base = model->base;
        DSV4MTP *old_mtp = model->mtp;
        model->base = base_work;
        model->mtp = mtp_work;
        base_work = NULL;
        mtp_work = NULL;
        dsv4_model_destroy(old_base);
        dsv4_mtp_destroy(old_mtp);
    }

    memcpy(out_base_streams, base_streams, stream_count * sizeof(float));
    memcpy(out_base_logits, base_logits,
           model->config.base.vocab_size * sizeof(float));
    memcpy(out_mtp_streams, mtp_streams, stream_count * sizeof(float));
    memcpy(out_mtp_logits, mtp_logits,
           model->config.base.vocab_size * sizeof(float));
    memcpy(out_mtp_route_indices, route_indices,
           route_count * sizeof(uint32_t));
    memcpy(out_mtp_route_weights, route_weights,
           route_count * sizeof(float));
    out_trace->mtp = mtp_trace;
    status = DSV4_FULL_OK;

cleanup:
    dsv4_model_destroy(base_work);
    dsv4_mtp_destroy(mtp_work);
    free(base_streams);
    free(base_logits);
    free(mtp_streams);
    free(mtp_logits);
    free(route_indices);
    free(route_weights);
    return status;
}

static uint64_t argmax_token(const float *logits, size_t count)
{
    size_t i;
    size_t best = 0u;
    for (i = 1u; i < count; ++i)
        if (logits[i] > logits[best]) best = i;
    return (uint64_t)best;
}

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
    size_t *out_length)
{
    DSV4FullModelStatus status;
    uint64_t *sequence = NULL;
    float *base_streams = NULL;
    float *base_logits = NULL;
    float *mtp_streams = NULL;
    float *mtp_logits = NULL;
    uint32_t *route_indices = NULL;
    float *route_weights = NULL;
    DSV4FullModelTrace trace;
    size_t total_limit;
    size_t stream_count;
    size_t route_count;
    size_t position;
    size_t length;

    if (model == NULL || weights == NULL || prompt_tokens == NULL ||
        prompt_len == 0u || out_tokens == NULL || out_length == NULL)
        return DSV4_FULL_INVALID_ARGUMENT;
    status = validate_weights(model, weights);
    if (status != DSV4_FULL_OK) return status;
    if (!checked_add_size(prompt_len, max_new_tokens, &total_limit))
        return DSV4_FULL_OVERFLOW;
    if (total_limit > model->config.base.max_seq_len || out_capacity < total_limit)
        return DSV4_FULL_INVALID_ARGUMENT;
    if (stop_on_eos != 0 && eos_id >= model->config.base.vocab_size)
        return DSV4_FULL_INVALID_ARGUMENT;
    for (position = 0u; position < prompt_len; ++position)
        if (prompt_tokens[position] >= model->config.base.vocab_size)
            return DSV4_FULL_INVALID_ARGUMENT;

    if (!checked_mul_size(model->config.base.hc_mult,
                          model->config.base.hidden_size, &stream_count))
        return DSV4_FULL_OVERFLOW;
    route_count = model->config.mtp.decoder.moe.top_k;

    status = alloc_u64(&sequence, total_limit);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&base_streams, stream_count);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&base_logits, model->config.base.vocab_size);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&mtp_streams, stream_count);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&mtp_logits, model->config.base.vocab_size);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_u32(&route_indices, route_count);
    if (status != DSV4_FULL_OK) goto cleanup;
    status = alloc_floats(&route_weights, route_count);
    if (status != DSV4_FULL_OK) goto cleanup;

    status = dsv4_full_model_reset(model);
    if (status != DSV4_FULL_OK) goto cleanup;
    memcpy(sequence, prompt_tokens, prompt_len * sizeof(uint64_t));
    length = prompt_len;

    for (position = 0u; position < prompt_len; ++position) {
        status = dsv4_full_model_step_f32(
            model, weights, prompt_tokens[position],
            base_streams, base_logits, mtp_streams, mtp_logits,
            route_indices, route_weights, route_count, &trace);
        if (status != DSV4_FULL_OK) goto cleanup;
    }

    for (position = 0u; position < max_new_tokens; ++position) {
        const uint64_t next = argmax_token(base_logits, model->config.base.vocab_size);
        sequence[length++] = next;
        if (stop_on_eos != 0 && next == eos_id) break;
        status = dsv4_full_model_step_f32(
            model, weights, next,
            base_streams, base_logits, mtp_streams, mtp_logits,
            route_indices, route_weights, route_count, &trace);
        if (status != DSV4_FULL_OK) goto cleanup;
    }

    memcpy(out_tokens, sequence, length * sizeof(uint64_t));
    *out_length = length;
    status = DSV4_FULL_OK;

cleanup:
    free(sequence);
    free(base_streams);
    free(base_logits);
    free(mtp_streams);
    free(mtp_logits);
    free(route_indices);
    free(route_weights);
    return status;
}

const char *dsv4_full_model_status_string(DSV4FullModelStatus status)
{
    switch (status) {
        case DSV4_FULL_OK: return "ok";
        case DSV4_FULL_INVALID_ARGUMENT: return "invalid argument";
        case DSV4_FULL_NONFINITE: return "non-finite input, weight or result";
        case DSV4_FULL_OVERFLOW: return "size overflow";
        case DSV4_FULL_ALLOCATION_FAILED: return "allocation failed";
        case DSV4_FULL_DEPENDENCY_FAILED: return "dependency failed";
        default: return "unknown status";
    }
}

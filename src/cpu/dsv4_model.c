#include "dsv4_model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct DSV4Model {
    DSV4ModelConfig config;
    DSV4DecoderLayerConfig *layer_configs;
    DSV4DecoderLayer **layers;
    uint64_t next_position;
};

static int checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL) return 0;
    if (a != 0u && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int all_finite(const float *values, size_t count)
{
    size_t i;
    if (values == NULL) return 0;
    for (i = 0u; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

static DSV4ModelStatus alloc_floats(float **out, size_t count, int zero)
{
    size_t bytes;
    void *memory;
    if (out == NULL) return DSV4_MODEL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_MODEL_OK;
    if (!checked_mul_size(count, sizeof(float), &bytes)) return DSV4_MODEL_OVERFLOW;
    memory = zero != 0 ? calloc(1u, bytes) : malloc(bytes);
    if (memory == NULL) return DSV4_MODEL_ALLOCATION_FAILED;
    *out = (float *)memory;
    return DSV4_MODEL_OK;
}

static DSV4ModelStatus alloc_size_t(size_t **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_MODEL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_MODEL_OK;
    if (!checked_mul_size(count, sizeof(size_t), &bytes)) return DSV4_MODEL_OVERFLOW;
    *out = (size_t *)malloc(bytes);
    return *out == NULL ? DSV4_MODEL_ALLOCATION_FAILED : DSV4_MODEL_OK;
}

static DSV4ModelStatus alloc_u32(uint32_t **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_MODEL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_MODEL_OK;
    if (!checked_mul_size(count, sizeof(uint32_t), &bytes)) return DSV4_MODEL_OVERFLOW;
    *out = (uint32_t *)malloc(bytes);
    return *out == NULL ? DSV4_MODEL_ALLOCATION_FAILED : DSV4_MODEL_OK;
}

static DSV4ModelStatus map_decoder(DSV4DecoderLayerStatus status)
{
    switch (status) {
        case DSV4_DL_OK: return DSV4_MODEL_OK;
        case DSV4_DL_INVALID_ARGUMENT: return DSV4_MODEL_INVALID_ARGUMENT;
        case DSV4_DL_NONFINITE: return DSV4_MODEL_NONFINITE;
        case DSV4_DL_OVERFLOW: return DSV4_MODEL_OVERFLOW;
        case DSV4_DL_ALLOCATION_FAILED: return DSV4_MODEL_ALLOCATION_FAILED;
        case DSV4_DL_DEPENDENCY_FAILED: return DSV4_MODEL_DEPENDENCY_FAILED;
        default: return DSV4_MODEL_DEPENDENCY_FAILED;
    }
}

static DSV4ModelStatus validate_config(const DSV4ModelConfig *config)
{
    size_t stream_count;
    size_t ignored;
    size_t i;
    if (config == NULL || config->vocab_size == 0u || config->hidden_size == 0u ||
        config->hc_mult == 0u || config->num_layers == 0u ||
        config->max_seq_len == 0u || config->layer_configs == NULL ||
        config->max_seq_len > (size_t)INT64_MAX ||
        !isfinite(config->hc_eps) || config->hc_eps <= 0.0f ||
        !isfinite(config->norm_eps) || config->norm_eps <= 0.0f)
        return DSV4_MODEL_INVALID_ARGUMENT;
    if (!checked_mul_size(config->hc_mult, config->hidden_size, &stream_count) ||
        !checked_mul_size(config->vocab_size, config->hidden_size, &ignored) ||
        !checked_mul_size(config->hc_mult, stream_count, &ignored))
        return DSV4_MODEL_OVERFLOW;

    for (i = 0u; i < config->num_layers; ++i) {
        const DSV4DecoderLayerConfig *layer = &config->layer_configs[i];
        if (layer->layer_idx != (uint32_t)i || layer->hidden_size != config->hidden_size ||
            layer->hc_mult != config->hc_mult || layer->moe.vocab_size != config->vocab_size)
            return DSV4_MODEL_INVALID_ARGUMENT;
    }
    return DSV4_MODEL_OK;
}

static void destroy_layer_array(DSV4DecoderLayer **layers, size_t count)
{
    size_t i;
    if (layers == NULL) return;
    for (i = 0u; i < count; ++i) dsv4_decoder_layer_destroy(layers[i]);
    free(layers);
}

static DSV4ModelStatus create_layer_array(const DSV4DecoderLayerConfig *configs,
                                          size_t count,
                                          DSV4DecoderLayer ***out_layers)
{
    DSV4DecoderLayer **layers;
    size_t bytes;
    size_t i;
    if (configs == NULL || out_layers == NULL || count == 0u)
        return DSV4_MODEL_INVALID_ARGUMENT;
    *out_layers = NULL;
    if (!checked_mul_size(count, sizeof(*layers), &bytes)) return DSV4_MODEL_OVERFLOW;
    layers = (DSV4DecoderLayer **)calloc(1u, bytes);
    if (layers == NULL) return DSV4_MODEL_ALLOCATION_FAILED;
    for (i = 0u; i < count; ++i) {
        DSV4DecoderLayerStatus layer_status;
        layers[i] = dsv4_decoder_layer_create(&configs[i], &layer_status);
        if (layers[i] == NULL) {
            DSV4ModelStatus status = map_decoder(layer_status);
            destroy_layer_array(layers, count);
            return status;
        }
    }
    *out_layers = layers;
    return DSV4_MODEL_OK;
}

DSV4Model *dsv4_model_create(const DSV4ModelConfig *config,
                             DSV4ModelStatus *out_status)
{
    DSV4Model *model = NULL;
    DSV4ModelStatus status;
    size_t config_bytes;

    status = validate_config(config);
    if (status != DSV4_MODEL_OK) goto done;
    model = (DSV4Model *)calloc(1u, sizeof(*model));
    if (model == NULL) {
        status = DSV4_MODEL_ALLOCATION_FAILED;
        goto done;
    }
    if (!checked_mul_size(config->num_layers, sizeof(*model->layer_configs), &config_bytes)) {
        status = DSV4_MODEL_OVERFLOW;
        goto fail;
    }
    model->layer_configs = (DSV4DecoderLayerConfig *)malloc(config_bytes);
    if (model->layer_configs == NULL) {
        status = DSV4_MODEL_ALLOCATION_FAILED;
        goto fail;
    }
    memcpy(model->layer_configs, config->layer_configs, config_bytes);
    model->config = *config;
    model->config.layer_configs = model->layer_configs;
    status = create_layer_array(model->layer_configs, config->num_layers, &model->layers);
    if (status != DSV4_MODEL_OK) goto fail;
    status = DSV4_MODEL_OK;
    goto done;

fail:
    destroy_layer_array(model->layers, config != NULL ? config->num_layers : 0u);
    free(model->layer_configs);
    free(model);
    model = NULL;
done:
    if (out_status != NULL) *out_status = status;
    return model;
}

void dsv4_model_destroy(DSV4Model *model)
{
    if (model == NULL) return;
    destroy_layer_array(model->layers, model->config.num_layers);
    free(model->layer_configs);
    free(model);
}

DSV4ModelStatus dsv4_model_reset(DSV4Model *model)
{
    DSV4DecoderLayer **fresh = NULL;
    DSV4ModelStatus status;
    if (model == NULL || model->layer_configs == NULL) return DSV4_MODEL_INVALID_ARGUMENT;
    status = create_layer_array(model->layer_configs, model->config.num_layers, &fresh);
    if (status != DSV4_MODEL_OK) return status;
    destroy_layer_array(model->layers, model->config.num_layers);
    model->layers = fresh;
    model->next_position = 0u;
    return DSV4_MODEL_OK;
}

static DSV4ModelStatus clone_layer_array(DSV4DecoderLayer *const *source,
                                         size_t count,
                                         DSV4DecoderLayer ***out_layers)
{
    DSV4DecoderLayer **layers;
    size_t bytes;
    size_t i;
    if (source == NULL || out_layers == NULL || count == 0u)
        return DSV4_MODEL_INVALID_ARGUMENT;
    *out_layers = NULL;
    if (!checked_mul_size(count, sizeof(*layers), &bytes)) return DSV4_MODEL_OVERFLOW;
    layers = (DSV4DecoderLayer **)calloc(1u, bytes);
    if (layers == NULL) return DSV4_MODEL_ALLOCATION_FAILED;
    for (i = 0u; i < count; ++i) {
        DSV4DecoderLayerStatus layer_status;
        layers[i] = dsv4_decoder_layer_clone(source[i], &layer_status);
        if (layers[i] == NULL) {
            DSV4ModelStatus status = map_decoder(layer_status);
            destroy_layer_array(layers, count);
            return status;
        }
    }
    *out_layers = layers;
    return DSV4_MODEL_OK;
}

static DSV4ModelStatus unweighted_rms(const float *input,
                                      size_t count,
                                      float eps,
                                      float *output)
{
    double mean_square = 0.0;
    double inverse;
    size_t i;
    if (input == NULL || output == NULL || count == 0u || !isfinite(eps) || eps <= 0.0f)
        return DSV4_MODEL_INVALID_ARGUMENT;
    for (i = 0u; i < count; ++i) {
        if (!isfinite(input[i])) return DSV4_MODEL_NONFINITE;
        mean_square += (double)input[i] * (double)input[i];
    }
    mean_square /= (double)count;
    if (!isfinite(mean_square) || mean_square + (double)eps <= 0.0)
        return DSV4_MODEL_NONFINITE;
    inverse = 1.0 / sqrt(mean_square + (double)eps);
    if (!isfinite(inverse)) return DSV4_MODEL_NONFINITE;
    for (i = 0u; i < count; ++i) {
        output[i] = (float)((double)input[i] * inverse);
        if (!isfinite(output[i])) return DSV4_MODEL_NONFINITE;
    }
    return DSV4_MODEL_OK;
}

static DSV4ModelStatus weighted_rms(const float *input,
                                    const float *weight,
                                    size_t count,
                                    float eps,
                                    float *output)
{
    DSV4ModelStatus status;
    float *normalized = NULL;
    size_t i;
    if (weight == NULL) return DSV4_MODEL_INVALID_ARGUMENT;
    status = alloc_floats(&normalized, count, 0);
    if (status != DSV4_MODEL_OK) return status;
    status = unweighted_rms(input, count, eps, normalized);
    if (status != DSV4_MODEL_OK) goto cleanup;
    for (i = 0u; i < count; ++i) {
        if (!isfinite(weight[i])) {
            status = DSV4_MODEL_NONFINITE;
            goto cleanup;
        }
        output[i] = normalized[i] * weight[i];
        if (!isfinite(output[i])) {
            status = DSV4_MODEL_NONFINITE;
            goto cleanup;
        }
    }
    status = DSV4_MODEL_OK;
cleanup:
    free(normalized);
    return status;
}

static float sigmoid_stable(float x)
{
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

static DSV4ModelStatus hyper_head(const DSV4Model *model,
                                  const DSV4ModelWeights *weights,
                                  const float *streams,
                                  float *collapsed)
{
    DSV4ModelStatus status;
    float *flat_norm = NULL;
    float *pre = NULL;
    size_t stream_count;
    size_t stream;
    size_t input_slot;
    size_t dim;

    if (model == NULL || weights == NULL || streams == NULL || collapsed == NULL ||
        weights->hc_head_fn == NULL || weights->hc_head_base == NULL ||
        weights->hc_head_scale == NULL)
        return DSV4_MODEL_INVALID_ARGUMENT;
    if (!checked_mul_size(model->config.hc_mult, model->config.hidden_size, &stream_count))
        return DSV4_MODEL_OVERFLOW;
    if (!all_finite(weights->hc_head_fn, model->config.hc_mult * stream_count) ||
        !all_finite(weights->hc_head_base, model->config.hc_mult) ||
        !all_finite(weights->hc_head_scale, 1u))
        return DSV4_MODEL_NONFINITE;

    status = alloc_floats(&flat_norm, stream_count, 0);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_floats(&pre, model->config.hc_mult, 0);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = unweighted_rms(streams, stream_count, model->config.norm_eps, flat_norm);
    if (status != DSV4_MODEL_OK) goto cleanup;

    for (stream = 0u; stream < model->config.hc_mult; ++stream) {
        const float *row = weights->hc_head_fn + stream * stream_count;
        float mix = 0.0f;
        float arg;
        for (input_slot = 0u; input_slot < stream_count; ++input_slot)
            mix += row[input_slot] * flat_norm[input_slot];
        arg = mix * weights->hc_head_scale[0] + weights->hc_head_base[stream];
        if (!isfinite(arg)) {
            status = DSV4_MODEL_NONFINITE;
            goto cleanup;
        }
        pre[stream] = sigmoid_stable(arg) + model->config.hc_eps;
        if (!isfinite(pre[stream])) {
            status = DSV4_MODEL_NONFINITE;
            goto cleanup;
        }
    }

    for (dim = 0u; dim < model->config.hidden_size; ++dim) {
        float sum = 0.0f;
        for (stream = 0u; stream < model->config.hc_mult; ++stream)
            sum += pre[stream] * streams[stream * model->config.hidden_size + dim];
        if (!isfinite(sum)) {
            status = DSV4_MODEL_NONFINITE;
            goto cleanup;
        }
        collapsed[dim] = sum;
    }
    status = DSV4_MODEL_OK;
cleanup:
    free(flat_norm);
    free(pre);
    return status;
}

static DSV4ModelStatus lm_head(const DSV4Model *model,
                               const float *hidden,
                               const float *weight,
                               float *logits)
{
    size_t token;
    size_t dim;
    if (model == NULL || hidden == NULL || weight == NULL || logits == NULL)
        return DSV4_MODEL_INVALID_ARGUMENT;
    for (token = 0u; token < model->config.vocab_size; ++token) {
        const float *row = weight + token * model->config.hidden_size;
        float sum = 0.0f;
        for (dim = 0u; dim < model->config.hidden_size; ++dim) {
            if (!isfinite(row[dim]) || !isfinite(hidden[dim]))
                return DSV4_MODEL_NONFINITE;
            sum += row[dim] * hidden[dim];
        }
        if (!isfinite(sum)) return DSV4_MODEL_NONFINITE;
        logits[token] = sum;
    }
    return DSV4_MODEL_OK;
}

DSV4ModelStatus dsv4_model_step_f32(DSV4Model *model,
                                    const DSV4ModelWeights *weights,
                                    uint64_t token_id,
                                    float *out_streams,
                                    float *out_logits)
{
    DSV4ModelStatus status;
    DSV4DecoderLayer **work_layers = NULL;
    float *streams_a = NULL;
    float *streams_b = NULL;
    float *collapsed = NULL;
    float *normalized = NULL;
    float *logits = NULL;
    size_t *attention_indices = NULL;
    uint32_t *route_indices = NULL;
    float *route_weights = NULL;
    size_t stream_count;
    size_t max_attention_topk = 0u;
    size_t max_route_topk = 0u;
    size_t layer_index;
    size_t stream;
    const float *embedding;

    if (model == NULL || weights == NULL || out_streams == NULL || out_logits == NULL ||
        weights->embedding_weight == NULL || weights->layer_weights == NULL ||
        weights->output_norm_weight == NULL || weights->lm_head_weight == NULL ||
        token_id >= (uint64_t)model->config.vocab_size ||
        model->next_position >= (uint64_t)model->config.max_seq_len ||
        model->next_position > (uint64_t)INT64_MAX)
        return DSV4_MODEL_INVALID_ARGUMENT;
    if (!checked_mul_size(model->config.hc_mult, model->config.hidden_size, &stream_count))
        return DSV4_MODEL_OVERFLOW;

    for (layer_index = 0u; layer_index < model->config.num_layers; ++layer_index) {
        const DSV4DecoderLayerConfig *config = &model->layer_configs[layer_index];
        if (config->attention.compression_rate == 4u &&
            config->attention.index_top_k > max_attention_topk)
            max_attention_topk = config->attention.index_top_k;
        if (config->moe.top_k > max_route_topk) max_route_topk = config->moe.top_k;
    }

    status = clone_layer_array(model->layers, model->config.num_layers, &work_layers);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_floats(&streams_a, stream_count, 0);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_floats(&streams_b, stream_count, 0);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_floats(&collapsed, model->config.hidden_size, 0);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_floats(&normalized, model->config.hidden_size, 0);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_floats(&logits, model->config.vocab_size, 0);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_size_t(&attention_indices, max_attention_topk);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_u32(&route_indices, max_route_topk);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = alloc_floats(&route_weights, max_route_topk, 0);
    if (status != DSV4_MODEL_OK) goto cleanup;

    embedding = weights->embedding_weight + (size_t)token_id * model->config.hidden_size;
    if (!all_finite(embedding, model->config.hidden_size)) {
        status = DSV4_MODEL_NONFINITE;
        goto cleanup;
    }
    for (stream = 0u; stream < model->config.hc_mult; ++stream)
        memcpy(streams_a + stream * model->config.hidden_size,
               embedding, model->config.hidden_size * sizeof(float));

    for (layer_index = 0u; layer_index < model->config.num_layers; ++layer_index) {
        DSV4DecoderLayerTrace trace;
        size_t attention_count = 0u;
        DSV4DecoderLayerStatus layer_status = dsv4_decoder_layer_step_f32(
            work_layers[layer_index],
            &weights->layer_weights[layer_index],
            (int64_t)model->next_position,
            token_id,
            streams_a,
            streams_b,
            attention_indices,
            max_attention_topk,
            &attention_count,
            route_indices,
            route_weights,
            max_route_topk,
            &trace);
        status = map_decoder(layer_status);
        if (status != DSV4_MODEL_OK) goto cleanup;
        {
            float *tmp = streams_a;
            streams_a = streams_b;
            streams_b = tmp;
        }
    }

    status = hyper_head(model, weights, streams_a, collapsed);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = weighted_rms(collapsed, weights->output_norm_weight,
                          model->config.hidden_size, model->config.norm_eps, normalized);
    if (status != DSV4_MODEL_OK) goto cleanup;
    status = lm_head(model, normalized, weights->lm_head_weight, logits);
    if (status != DSV4_MODEL_OK) goto cleanup;

    for (layer_index = 0u; layer_index < model->config.num_layers; ++layer_index) {
        DSV4DecoderLayer *old = model->layers[layer_index];
        model->layers[layer_index] = work_layers[layer_index];
        work_layers[layer_index] = NULL;
        dsv4_decoder_layer_destroy(old);
    }
    model->next_position += 1u;
    memcpy(out_streams, streams_a, stream_count * sizeof(float));
    memcpy(out_logits, logits, model->config.vocab_size * sizeof(float));
    status = DSV4_MODEL_OK;

cleanup:
    destroy_layer_array(work_layers, model != NULL ? model->config.num_layers : 0u);
    free(streams_a);
    free(streams_b);
    free(collapsed);
    free(normalized);
    free(logits);
    free(attention_indices);
    free(route_indices);
    free(route_weights);
    return status;
}

const char *dsv4_model_status_string(DSV4ModelStatus status)
{
    switch (status) {
        case DSV4_MODEL_OK: return "ok";
        case DSV4_MODEL_INVALID_ARGUMENT: return "invalid argument";
        case DSV4_MODEL_NONFINITE: return "non-finite input, weight or result";
        case DSV4_MODEL_OVERFLOW: return "size or position overflow";
        case DSV4_MODEL_ALLOCATION_FAILED: return "allocation failed";
        case DSV4_MODEL_DEPENDENCY_FAILED: return "dependency failed";
        default: return "unknown status";
    }
}

#include "dsv4_decoder_layer.h"

#include "dsv4_cpu_ops.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct DSV4DecoderLayer {
    DSV4DecoderLayerConfig config;
    DSV4AttentionRuntime *attention;
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

static DSV4DecoderLayerStatus alloc_floats(float **out, size_t count, int zero)
{
    size_t bytes;
    void *memory;
    if (out == NULL) return DSV4_DL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_DL_OK;
    if (!checked_mul_size(count, sizeof(float), &bytes)) return DSV4_DL_OVERFLOW;
    memory = zero != 0 ? calloc(1u, bytes) : malloc(bytes);
    if (memory == NULL) return DSV4_DL_ALLOCATION_FAILED;
    *out = (float *)memory;
    return DSV4_DL_OK;
}

static DSV4DecoderLayerStatus alloc_indices(size_t **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_DL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_DL_OK;
    if (!checked_mul_size(count, sizeof(size_t), &bytes)) return DSV4_DL_OVERFLOW;
    *out = (size_t *)malloc(bytes);
    return *out == NULL ? DSV4_DL_ALLOCATION_FAILED : DSV4_DL_OK;
}

static DSV4DecoderLayerStatus alloc_u32(uint32_t **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_DL_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_DL_OK;
    if (!checked_mul_size(count, sizeof(uint32_t), &bytes)) return DSV4_DL_OVERFLOW;
    *out = (uint32_t *)malloc(bytes);
    return *out == NULL ? DSV4_DL_ALLOCATION_FAILED : DSV4_DL_OK;
}

static DSV4DecoderLayerStatus map_cpu(int status)
{
    if (status == DSV4_CPU_OK) return DSV4_DL_OK;
    if (status == DSV4_CPU_INVALID) return DSV4_DL_INVALID_ARGUMENT;
    if (status == DSV4_CPU_NOMEM) return DSV4_DL_ALLOCATION_FAILED;
    if (status == DSV4_CPU_NUMERIC) return DSV4_DL_NONFINITE;
    return DSV4_DL_DEPENDENCY_FAILED;
}

static DSV4DecoderLayerStatus map_attention(DSV4AttentionRuntimeStatus status)
{
    switch (status) {
        case DSV4_AR_OK: return DSV4_DL_OK;
        case DSV4_AR_INVALID_ARGUMENT: return DSV4_DL_INVALID_ARGUMENT;
        case DSV4_AR_NONFINITE: return DSV4_DL_NONFINITE;
        case DSV4_AR_OVERFLOW: return DSV4_DL_OVERFLOW;
        case DSV4_AR_ALLOCATION_FAILED: return DSV4_DL_ALLOCATION_FAILED;
        case DSV4_AR_DEPENDENCY_FAILED: return DSV4_DL_DEPENDENCY_FAILED;
        default: return DSV4_DL_DEPENDENCY_FAILED;
    }
}

static DSV4DecoderLayerStatus map_moe(DSV4MoEStatus status)
{
    switch (status) {
        case DSV4_MOE_OK: return DSV4_DL_OK;
        case DSV4_MOE_INVALID_ARGUMENT: return DSV4_DL_INVALID_ARGUMENT;
        case DSV4_MOE_NONFINITE: return DSV4_DL_NONFINITE;
        case DSV4_MOE_OVERFLOW: return DSV4_DL_OVERFLOW;
        case DSV4_MOE_ALLOCATION_FAILED: return DSV4_DL_ALLOCATION_FAILED;
        case DSV4_MOE_DEPENDENCY_FAILED: return DSV4_DL_DEPENDENCY_FAILED;
        default: return DSV4_DL_DEPENDENCY_FAILED;
    }
}

static DSV4DecoderLayerStatus validate_config(const DSV4DecoderLayerConfig *config)
{
    size_t hc_rows;
    size_t hc_mix;
    size_t hc_width;
    size_t ignored;

    if (config == NULL || config->hidden_size == 0u || config->hc_mult == 0u ||
        config->sinkhorn_iterations < 1 || !isfinite(config->hc_eps) ||
        config->hc_eps <= 0.0f || !isfinite(config->norm_eps) ||
        config->norm_eps <= 0.0f ||
        config->attention.hidden_size != config->hidden_size ||
        config->moe.hidden_size != config->hidden_size ||
        config->moe.intermediate_size == 0u || config->moe.num_routed_experts == 0u ||
        config->moe.top_k == 0u || config->moe.top_k > config->moe.num_routed_experts ||
        config->moe.vocab_size == 0u || !isfinite(config->moe.route_scale) ||
        config->moe.route_scale < 0.0f || !isfinite(config->moe.swiglu_limit) ||
        config->moe.swiglu_limit <= 0.0f)
        return DSV4_DL_INVALID_ARGUMENT;

    if (!checked_add_size(config->hc_mult, 2u, &hc_rows) ||
        !checked_mul_size(hc_rows, config->hc_mult, &hc_mix) ||
        !checked_mul_size(config->hc_mult, config->hidden_size, &hc_width) ||
        !checked_mul_size(hc_mix, hc_width, &ignored) ||
        !checked_mul_size(config->hc_mult, config->hc_mult, &ignored))
        return DSV4_DL_OVERFLOW;
    return DSV4_DL_OK;
}

DSV4DecoderLayer *dsv4_decoder_layer_create(
    const DSV4DecoderLayerConfig *config,
    DSV4DecoderLayerStatus *out_status)
{
    DSV4DecoderLayerStatus status;
    DSV4AttentionRuntimeStatus attention_status;
    DSV4DecoderLayer *layer = NULL;

    status = validate_config(config);
    if (status != DSV4_DL_OK) goto done;
    layer = (DSV4DecoderLayer *)calloc(1u, sizeof(*layer));
    if (layer == NULL) {
        status = DSV4_DL_ALLOCATION_FAILED;
        goto done;
    }
    layer->config = *config;
    layer->attention = dsv4_attention_runtime_create(&config->attention, &attention_status);
    if (layer->attention == NULL) {
        status = map_attention(attention_status);
        free(layer);
        layer = NULL;
        goto done;
    }
    status = DSV4_DL_OK;

done:
    if (out_status != NULL) *out_status = status;
    return layer;
}

DSV4DecoderLayer *dsv4_decoder_layer_clone(
    const DSV4DecoderLayer *source,
    DSV4DecoderLayerStatus *out_status)
{
    DSV4DecoderLayer *copy = NULL;
    DSV4DecoderLayerStatus status = DSV4_DL_INVALID_ARGUMENT;
    DSV4AttentionRuntimeStatus attention_status;

    if (source == NULL || source->attention == NULL) goto done;
    copy = (DSV4DecoderLayer *)calloc(1u, sizeof(*copy));
    if (copy == NULL) {
        status = DSV4_DL_ALLOCATION_FAILED;
        goto done;
    }
    copy->config = source->config;
    copy->attention = dsv4_attention_runtime_clone(source->attention, &attention_status);
    if (copy->attention == NULL) {
        status = map_attention(attention_status);
        free(copy);
        copy = NULL;
        goto done;
    }
    status = DSV4_DL_OK;

done:
    if (out_status != NULL) *out_status = status;
    return copy;
}

void dsv4_decoder_layer_destroy(DSV4DecoderLayer *layer)
{
    if (layer == NULL) return;
    dsv4_attention_runtime_destroy(layer->attention);
    free(layer);
}

DSV4DecoderLayerStatus dsv4_decoder_layer_reset(DSV4DecoderLayer *layer)
{
    if (layer == NULL || layer->attention == NULL) return DSV4_DL_INVALID_ARGUMENT;
    return map_attention(dsv4_attention_runtime_reset(layer->attention));
}

static DSV4DecoderLayerStatus weighted_rms_norm(const float *input,
                                                const float *weight,
                                                size_t dim,
                                                float eps,
                                                float *output)
{
    size_t i;
    double mean_square = 0.0;
    double inverse;
    if (input == NULL || weight == NULL || output == NULL || dim == 0u ||
        !isfinite(eps) || eps <= 0.0f)
        return DSV4_DL_INVALID_ARGUMENT;
    for (i = 0u; i < dim; ++i) {
        if (!isfinite(input[i]) || !isfinite(weight[i])) return DSV4_DL_NONFINITE;
        mean_square += (double)input[i] * (double)input[i];
    }
    mean_square /= (double)dim;
    if (!isfinite(mean_square) || mean_square + (double)eps <= 0.0)
        return DSV4_DL_NONFINITE;
    inverse = 1.0 / sqrt(mean_square + (double)eps);
    if (!isfinite(inverse)) return DSV4_DL_NONFINITE;
    for (i = 0u; i < dim; ++i) {
        output[i] = (float)((double)input[i] * inverse * (double)weight[i]);
        if (!isfinite(output[i])) return DSV4_DL_NONFINITE;
    }
    return DSV4_DL_OK;
}

static DSV4DecoderLayerStatus hc_mix(const float *streams,
                                     const float *sublayer,
                                     const float *post,
                                     const float *comb,
                                     size_t hc_mult,
                                     size_t hidden_size,
                                     float *output)
{
    size_t out_stream;
    size_t source_stream;
    size_t dim;
    if (streams == NULL || sublayer == NULL || post == NULL || comb == NULL ||
        output == NULL || hc_mult == 0u || hidden_size == 0u)
        return DSV4_DL_INVALID_ARGUMENT;

    for (out_stream = 0u; out_stream < hc_mult; ++out_stream) {
        if (!isfinite(post[out_stream])) return DSV4_DL_NONFINITE;
        for (dim = 0u; dim < hidden_size; ++dim) {
            double residual = 0.0;
            double value;
            if (!isfinite(sublayer[dim])) return DSV4_DL_NONFINITE;
            for (source_stream = 0u; source_stream < hc_mult; ++source_stream) {
                const float coefficient = comb[source_stream * hc_mult + out_stream];
                const float stream_value = streams[source_stream * hidden_size + dim];
                if (!isfinite(coefficient) || !isfinite(stream_value))
                    return DSV4_DL_NONFINITE;
                residual += (double)coefficient * (double)stream_value;
            }
            value = residual + (double)post[out_stream] * (double)sublayer[dim];
            if (!isfinite(value) || value > (double)FLT_MAX || value < -(double)FLT_MAX)
                return DSV4_DL_NONFINITE;
            output[out_stream * hidden_size + dim] = (float)value;
        }
    }
    return DSV4_DL_OK;
}

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
    DSV4DecoderLayerTrace *out_trace)
{
    DSV4DecoderLayerStatus status;
    DSV4AttentionRuntimeStatus attention_status;
    DSV4AttentionRuntime *attention_work = NULL;
    DSV4AttentionRuntimeTrace attention_trace;
    float *post = NULL;
    float *comb = NULL;
    float *collapsed = NULL;
    float *normed = NULL;
    float *attention_output = NULL;
    float *mixed = NULL;
    float *ffn_post = NULL;
    float *ffn_comb = NULL;
    float *ffn_collapsed = NULL;
    float *ffn_normed = NULL;
    float *moe_output = NULL;
    float *final_streams = NULL;
    size_t *attention_indices = NULL;
    uint32_t *route_indices = NULL;
    float *route_weights = NULL;
    size_t attention_count = 0u;
    size_t stream_count;
    size_t comb_count;

    if (layer == NULL || layer->attention == NULL || weights == NULL || streams == NULL ||
        out_streams == NULL || out_attention_count == NULL || out_route_indices == NULL ||
        out_route_weights == NULL || out_trace == NULL || position < 0 ||
        route_capacity < layer->config.moe.top_k)
        return DSV4_DL_INVALID_ARGUMENT;
    if (layer->config.attention.compression_rate == 4u &&
        (out_attention_indices == NULL ||
         attention_capacity < layer->config.attention.index_top_k))
        return DSV4_DL_INVALID_ARGUMENT;

    if (!checked_mul_size(layer->config.hc_mult, layer->config.hidden_size, &stream_count) ||
        !checked_mul_size(layer->config.hc_mult, layer->config.hc_mult, &comb_count))
        return DSV4_DL_OVERFLOW;

    status = alloc_floats(&post, layer->config.hc_mult, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&comb, comb_count, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&collapsed, layer->config.hidden_size, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&normed, layer->config.hidden_size, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&attention_output, layer->config.hidden_size, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&mixed, stream_count, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&ffn_post, layer->config.hc_mult, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&ffn_comb, comb_count, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&ffn_collapsed, layer->config.hidden_size, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&ffn_normed, layer->config.hidden_size, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&moe_output, layer->config.hidden_size, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&final_streams, stream_count, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_u32(&route_indices, layer->config.moe.top_k);
    if (status != DSV4_DL_OK) goto cleanup;
    status = alloc_floats(&route_weights, layer->config.moe.top_k, 0);
    if (status != DSV4_DL_OK) goto cleanup;
    if (layer->config.attention.compression_rate == 4u) {
        status = alloc_indices(&attention_indices, layer->config.attention.index_top_k);
        if (status != DSV4_DL_OK) goto cleanup;
    }

    status = map_cpu(dsv4_cpu_hyperconnection(
        streams,
        weights->attention_hc_fn,
        weights->attention_hc_base,
        weights->attention_hc_scale,
        layer->config.hc_mult,
        layer->config.hidden_size,
        layer->config.sinkhorn_iterations,
        layer->config.hc_eps,
        layer->config.norm_eps,
        post,
        comb,
        collapsed));
    if (status != DSV4_DL_OK) goto cleanup;

    status = weighted_rms_norm(collapsed, weights->input_norm_weight,
                               layer->config.hidden_size, layer->config.norm_eps, normed);
    if (status != DSV4_DL_OK) goto cleanup;

    attention_work = dsv4_attention_runtime_clone(layer->attention, &attention_status);
    if (attention_work == NULL) {
        status = map_attention(attention_status);
        goto cleanup;
    }
    attention_status = dsv4_attention_runtime_step_f32(
        attention_work,
        &weights->attention,
        position,
        normed,
        attention_output,
        attention_indices,
        layer->config.attention.compression_rate == 4u
            ? layer->config.attention.index_top_k : 0u,
        &attention_count,
        &attention_trace);
    status = map_attention(attention_status);
    if (status != DSV4_DL_OK) goto cleanup;

    status = hc_mix(streams, attention_output, post, comb,
                    layer->config.hc_mult, layer->config.hidden_size, mixed);
    if (status != DSV4_DL_OK) goto cleanup;

    status = map_cpu(dsv4_cpu_hyperconnection(
        mixed,
        weights->ffn_hc_fn,
        weights->ffn_hc_base,
        weights->ffn_hc_scale,
        layer->config.hc_mult,
        layer->config.hidden_size,
        layer->config.sinkhorn_iterations,
        layer->config.hc_eps,
        layer->config.norm_eps,
        ffn_post,
        ffn_comb,
        ffn_collapsed));
    if (status != DSV4_DL_OK) goto cleanup;

    status = weighted_rms_norm(ffn_collapsed, weights->post_attention_norm_weight,
                               layer->config.hidden_size, layer->config.norm_eps,
                               ffn_normed);
    if (status != DSV4_DL_OK) goto cleanup;

    status = map_moe(dsv4_moe_forward_f32(
        &layer->config.moe,
        &weights->moe,
        layer->config.layer_idx,
        token_id,
        ffn_normed,
        moe_output,
        route_indices,
        route_weights,
        layer->config.moe.top_k));
    if (status != DSV4_DL_OK) goto cleanup;

    status = hc_mix(mixed, moe_output, ffn_post, ffn_comb,
                    layer->config.hc_mult, layer->config.hidden_size, final_streams);
    if (status != DSV4_DL_OK) goto cleanup;

    {
        DSV4AttentionRuntime *old_attention = layer->attention;
        layer->attention = attention_work;
        attention_work = NULL;
        dsv4_attention_runtime_destroy(old_attention);
    }
    memcpy(out_streams, final_streams, stream_count * sizeof(float));
    if (layer->config.attention.compression_rate == 4u && attention_count != 0u)
        memcpy(out_attention_indices, attention_indices,
               attention_count * sizeof(size_t));
    *out_attention_count = layer->config.attention.compression_rate == 4u
                               ? attention_count : 0u;
    memcpy(out_route_indices, route_indices,
           layer->config.moe.top_k * sizeof(uint32_t));
    memcpy(out_route_weights, route_weights,
           layer->config.moe.top_k * sizeof(float));
    out_trace->attention = attention_trace;
    status = DSV4_DL_OK;

cleanup:
    dsv4_attention_runtime_destroy(attention_work);
    free(post);
    free(comb);
    free(collapsed);
    free(normed);
    free(attention_output);
    free(mixed);
    free(ffn_post);
    free(ffn_comb);
    free(ffn_collapsed);
    free(ffn_normed);
    free(moe_output);
    free(final_streams);
    free(attention_indices);
    free(route_indices);
    free(route_weights);
    return status;
}

const char *dsv4_decoder_layer_status_string(DSV4DecoderLayerStatus status)
{
    switch (status) {
        case DSV4_DL_OK: return "ok";
        case DSV4_DL_INVALID_ARGUMENT: return "invalid argument";
        case DSV4_DL_NONFINITE: return "non-finite input, weight or result";
        case DSV4_DL_OVERFLOW: return "size overflow";
        case DSV4_DL_ALLOCATION_FAILED: return "allocation failed";
        case DSV4_DL_DEPENDENCY_FAILED: return "dependency failed";
        default: return "unknown status";
    }
}

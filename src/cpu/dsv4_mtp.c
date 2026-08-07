#include "dsv4_mtp.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct DSV4MTP {
    DSV4MTPConfig config;
    DSV4DecoderLayer *decoder;
    uint64_t next_position;
};

static int checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL) return 0;
    if (a != 0u && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static DSV4MTPStatus alloc_floats(float **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_MTP_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_MTP_OK;
    if (!checked_mul_size(count, sizeof(float), &bytes)) return DSV4_MTP_OVERFLOW;
    *out = (float *)malloc(bytes);
    return *out == NULL ? DSV4_MTP_ALLOCATION_FAILED : DSV4_MTP_OK;
}

static DSV4MTPStatus alloc_u32(uint32_t **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_MTP_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_MTP_OK;
    if (!checked_mul_size(count, sizeof(uint32_t), &bytes)) return DSV4_MTP_OVERFLOW;
    *out = (uint32_t *)malloc(bytes);
    return *out == NULL ? DSV4_MTP_ALLOCATION_FAILED : DSV4_MTP_OK;
}

static DSV4MTPStatus map_decoder(DSV4DecoderLayerStatus status)
{
    switch (status) {
        case DSV4_DL_OK: return DSV4_MTP_OK;
        case DSV4_DL_INVALID_ARGUMENT: return DSV4_MTP_INVALID_ARGUMENT;
        case DSV4_DL_NONFINITE: return DSV4_MTP_NONFINITE;
        case DSV4_DL_OVERFLOW: return DSV4_MTP_OVERFLOW;
        case DSV4_DL_ALLOCATION_FAILED: return DSV4_MTP_ALLOCATION_FAILED;
        case DSV4_DL_DEPENDENCY_FAILED: return DSV4_MTP_DEPENDENCY_FAILED;
        default: return DSV4_MTP_DEPENDENCY_FAILED;
    }
}

static DSV4MTPStatus validate_config(const DSV4MTPConfig *config)
{
    size_t stream_count;
    size_t ignored;
    if (config == NULL || config->vocab_size == 0u || config->hidden_size == 0u ||
        config->hc_mult == 0u || config->max_seq_len == 0u ||
        config->max_seq_len > (size_t)INT64_MAX ||
        !isfinite(config->hc_eps) || config->hc_eps <= 0.0f ||
        !isfinite(config->norm_eps) || config->norm_eps <= 0.0f)
        return DSV4_MTP_INVALID_ARGUMENT;

    if (config->decoder.hidden_size != config->hidden_size ||
        config->decoder.hc_mult != config->hc_mult ||
        config->decoder.moe.vocab_size != config->vocab_size ||
        config->decoder.layer_idx < config->decoder.moe.num_hash_layers ||
        config->decoder.attention.compression_rate != 0u ||
        config->decoder.hc_eps != config->hc_eps ||
        config->decoder.norm_eps != config->norm_eps)
        return DSV4_MTP_INVALID_ARGUMENT;

    if (!checked_mul_size(config->hc_mult, config->hidden_size, &stream_count) ||
        !checked_mul_size(config->vocab_size, config->hidden_size, &ignored) ||
        !checked_mul_size(config->hidden_size, config->hidden_size, &ignored) ||
        !checked_mul_size(config->hc_mult, stream_count, &ignored))
        return DSV4_MTP_OVERFLOW;
    return DSV4_MTP_OK;
}

DSV4MTP *dsv4_mtp_create(const DSV4MTPConfig *config, DSV4MTPStatus *out_status)
{
    DSV4MTP *mtp = NULL;
    DSV4MTPStatus status = validate_config(config);
    DSV4DecoderLayerStatus decoder_status;

    if (status != DSV4_MTP_OK) goto done;
    mtp = (DSV4MTP *)calloc(1u, sizeof(*mtp));
    if (mtp == NULL) {
        status = DSV4_MTP_ALLOCATION_FAILED;
        goto done;
    }
    mtp->config = *config;
    mtp->decoder = dsv4_decoder_layer_create(&config->decoder, &decoder_status);
    if (mtp->decoder == NULL) {
        status = map_decoder(decoder_status);
        free(mtp);
        mtp = NULL;
        goto done;
    }
    status = DSV4_MTP_OK;

done:
    if (out_status != NULL) *out_status = status;
    return mtp;
}

void dsv4_mtp_destroy(DSV4MTP *mtp)
{
    if (mtp == NULL) return;
    dsv4_decoder_layer_destroy(mtp->decoder);
    free(mtp);
}

DSV4MTPStatus dsv4_mtp_reset(DSV4MTP *mtp)
{
    DSV4DecoderLayer *fresh;
    DSV4DecoderLayerStatus decoder_status;
    if (mtp == NULL) return DSV4_MTP_INVALID_ARGUMENT;
    fresh = dsv4_decoder_layer_create(&mtp->config.decoder, &decoder_status);
    if (fresh == NULL) return map_decoder(decoder_status);
    dsv4_decoder_layer_destroy(mtp->decoder);
    mtp->decoder = fresh;
    mtp->next_position = 0u;
    return DSV4_MTP_OK;
}

static DSV4MTPStatus unweighted_rms(const float *input, size_t count,
                                    float eps, float *output)
{
    double mean_square = 0.0;
    double inverse;
    size_t i;
    if (input == NULL || output == NULL || count == 0u ||
        !isfinite(eps) || eps <= 0.0f)
        return DSV4_MTP_INVALID_ARGUMENT;
    for (i = 0u; i < count; ++i) {
        if (!isfinite(input[i])) return DSV4_MTP_NONFINITE;
        mean_square += (double)input[i] * (double)input[i];
    }
    mean_square /= (double)count;
    if (!isfinite(mean_square) || mean_square + (double)eps <= 0.0)
        return DSV4_MTP_NONFINITE;
    inverse = 1.0 / sqrt(mean_square + (double)eps);
    if (!isfinite(inverse)) return DSV4_MTP_NONFINITE;
    for (i = 0u; i < count; ++i) {
        output[i] = (float)((double)input[i] * inverse);
        if (!isfinite(output[i])) return DSV4_MTP_NONFINITE;
    }
    return DSV4_MTP_OK;
}

static DSV4MTPStatus weighted_rms(const float *input, const float *weight,
                                  size_t count, float eps, float *output)
{
    DSV4MTPStatus status;
    float *normalized = NULL;
    size_t i;
    if (weight == NULL) return DSV4_MTP_INVALID_ARGUMENT;
    status = alloc_floats(&normalized, count);
    if (status != DSV4_MTP_OK) return status;
    status = unweighted_rms(input, count, eps, normalized);
    if (status != DSV4_MTP_OK) goto cleanup;
    for (i = 0u; i < count; ++i) {
        if (!isfinite(weight[i])) {
            status = DSV4_MTP_NONFINITE;
            goto cleanup;
        }
        output[i] = normalized[i] * weight[i];
        if (!isfinite(output[i])) {
            status = DSV4_MTP_NONFINITE;
            goto cleanup;
        }
    }
    status = DSV4_MTP_OK;
cleanup:
    free(normalized);
    return status;
}

static DSV4MTPStatus dense(const float *input, const float *weight,
                           size_t input_dim, size_t output_dim, float *output)
{
    size_t row;
    size_t column;
    if (input == NULL || weight == NULL || output == NULL ||
        input_dim == 0u || output_dim == 0u)
        return DSV4_MTP_INVALID_ARGUMENT;
    for (row = 0u; row < output_dim; ++row) {
        const float *wrow = weight + row * input_dim;
        float sum = 0.0f;
        for (column = 0u; column < input_dim; ++column) {
            if (!isfinite(input[column]) || !isfinite(wrow[column]))
                return DSV4_MTP_NONFINITE;
            sum += input[column] * wrow[column];
        }
        if (!isfinite(sum)) return DSV4_MTP_NONFINITE;
        output[row] = sum;
    }
    return DSV4_MTP_OK;
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

static DSV4MTPStatus hyper_head(const DSV4MTP *mtp,
                                const DSV4MTPWeights *weights,
                                const float *streams,
                                float *collapsed)
{
    DSV4MTPStatus status;
    float *flat_norm = NULL;
    float *pre = NULL;
    size_t stream_count;
    size_t stream;
    size_t input_slot;
    size_t dim;

    if (mtp == NULL || weights == NULL || streams == NULL || collapsed == NULL ||
        weights->hc_head_fn == NULL || weights->hc_head_base == NULL ||
        weights->hc_head_scale == NULL)
        return DSV4_MTP_INVALID_ARGUMENT;
    if (!checked_mul_size(mtp->config.hc_mult, mtp->config.hidden_size, &stream_count))
        return DSV4_MTP_OVERFLOW;

    status = alloc_floats(&flat_norm, stream_count);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&pre, mtp->config.hc_mult);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = unweighted_rms(streams, stream_count, mtp->config.norm_eps, flat_norm);
    if (status != DSV4_MTP_OK) goto cleanup;

    for (stream = 0u; stream < mtp->config.hc_mult; ++stream) {
        const float *row = weights->hc_head_fn + stream * stream_count;
        float mix = 0.0f;
        float arg;
        for (input_slot = 0u; input_slot < stream_count; ++input_slot) {
            if (!isfinite(row[input_slot])) {
                status = DSV4_MTP_NONFINITE;
                goto cleanup;
            }
            mix += row[input_slot] * flat_norm[input_slot];
        }
        if (!isfinite(weights->hc_head_scale[0]) ||
            !isfinite(weights->hc_head_base[stream])) {
            status = DSV4_MTP_NONFINITE;
            goto cleanup;
        }
        arg = mix * weights->hc_head_scale[0] + weights->hc_head_base[stream];
        if (!isfinite(arg)) {
            status = DSV4_MTP_NONFINITE;
            goto cleanup;
        }
        pre[stream] = sigmoid_stable(arg) + mtp->config.hc_eps;
        if (!isfinite(pre[stream])) {
            status = DSV4_MTP_NONFINITE;
            goto cleanup;
        }
    }

    for (dim = 0u; dim < mtp->config.hidden_size; ++dim) {
        float sum = 0.0f;
        for (stream = 0u; stream < mtp->config.hc_mult; ++stream)
            sum += pre[stream] * streams[stream * mtp->config.hidden_size + dim];
        if (!isfinite(sum)) {
            status = DSV4_MTP_NONFINITE;
            goto cleanup;
        }
        collapsed[dim] = sum;
    }
    status = DSV4_MTP_OK;
cleanup:
    free(flat_norm);
    free(pre);
    return status;
}

static DSV4MTPStatus lm_head(const DSV4MTP *mtp, const float *hidden,
                             const float *weight, float *logits)
{
    size_t token;
    size_t dim;
    if (mtp == NULL || hidden == NULL || weight == NULL || logits == NULL)
        return DSV4_MTP_INVALID_ARGUMENT;
    for (token = 0u; token < mtp->config.vocab_size; ++token) {
        const float *row = weight + token * mtp->config.hidden_size;
        float sum = 0.0f;
        for (dim = 0u; dim < mtp->config.hidden_size; ++dim) {
            if (!isfinite(row[dim]) || !isfinite(hidden[dim]))
                return DSV4_MTP_NONFINITE;
            sum += row[dim] * hidden[dim];
        }
        if (!isfinite(sum)) return DSV4_MTP_NONFINITE;
        logits[token] = sum;
    }
    return DSV4_MTP_OK;
}

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
    DSV4MTPTrace *out_trace)
{
    DSV4MTPStatus status = DSV4_MTP_INVALID_ARGUMENT;
    DSV4DecoderLayerStatus decoder_status;
    DSV4DecoderLayer *decoder_work = NULL;
    float *embedding_normed = NULL;
    float *embedding_projected = NULL;
    float *hidden_normed = NULL;
    float *hidden_projected = NULL;
    float *combined = NULL;
    float *decoder_streams = NULL;
    float *collapsed = NULL;
    float *final_hidden = NULL;
    float *logits = NULL;
    uint32_t *route_indices = NULL;
    float *route_weights = NULL;
    DSV4DecoderLayerTrace trace;
    size_t attention_count = 0u;
    size_t stream_count;
    size_t stream;
    size_t dim;

    if (mtp == NULL || mtp->decoder == NULL || weights == NULL ||
        base_streams == NULL || out_streams == NULL || out_logits == NULL ||
        out_route_indices == NULL || out_route_weights == NULL || out_trace == NULL ||
        weights->embedding_weight == NULL || weights->lm_head_weight == NULL ||
        weights->embedding_norm_weight == NULL || weights->hidden_norm_weight == NULL ||
        weights->embedding_proj_weight == NULL || weights->hidden_proj_weight == NULL ||
        weights->output_norm_weight == NULL || token_id >= mtp->config.vocab_size ||
        route_capacity < mtp->config.decoder.moe.top_k ||
        mtp->next_position >= mtp->config.max_seq_len)
        return DSV4_MTP_INVALID_ARGUMENT;

    if (!checked_mul_size(mtp->config.hc_mult, mtp->config.hidden_size, &stream_count))
        return DSV4_MTP_OVERFLOW;

    status = alloc_floats(&embedding_normed, mtp->config.hidden_size);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&embedding_projected, mtp->config.hidden_size);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&hidden_normed, stream_count);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&hidden_projected, stream_count);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&combined, stream_count);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&decoder_streams, stream_count);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&collapsed, mtp->config.hidden_size);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&final_hidden, mtp->config.hidden_size);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&logits, mtp->config.vocab_size);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_u32(&route_indices, mtp->config.decoder.moe.top_k);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = alloc_floats(&route_weights, mtp->config.decoder.moe.top_k);
    if (status != DSV4_MTP_OK) goto cleanup;

    decoder_work = dsv4_decoder_layer_clone(mtp->decoder, &decoder_status);
    if (decoder_work == NULL) {
        status = map_decoder(decoder_status);
        goto cleanup;
    }

    status = weighted_rms(
        weights->embedding_weight + token_id * mtp->config.hidden_size,
        weights->embedding_norm_weight,
        mtp->config.hidden_size,
        mtp->config.norm_eps,
        embedding_normed);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = dense(embedding_normed, weights->embedding_proj_weight,
                   mtp->config.hidden_size, mtp->config.hidden_size,
                   embedding_projected);
    if (status != DSV4_MTP_OK) goto cleanup;

    for (stream = 0u; stream < mtp->config.hc_mult; ++stream) {
        const float *source = base_streams + stream * mtp->config.hidden_size;
        float *normed = hidden_normed + stream * mtp->config.hidden_size;
        float *projected = hidden_projected + stream * mtp->config.hidden_size;
        status = weighted_rms(source, weights->hidden_norm_weight,
                              mtp->config.hidden_size, mtp->config.norm_eps, normed);
        if (status != DSV4_MTP_OK) goto cleanup;
        status = dense(normed, weights->hidden_proj_weight,
                       mtp->config.hidden_size, mtp->config.hidden_size, projected);
        if (status != DSV4_MTP_OK) goto cleanup;
        for (dim = 0u; dim < mtp->config.hidden_size; ++dim) {
            combined[stream * mtp->config.hidden_size + dim] =
                projected[dim] + embedding_projected[dim];
            if (!isfinite(combined[stream * mtp->config.hidden_size + dim])) {
                status = DSV4_MTP_NONFINITE;
                goto cleanup;
            }
        }
    }

    decoder_status = dsv4_decoder_layer_step_f32(
        decoder_work, &weights->decoder, (int64_t)mtp->next_position, token_id,
        combined, decoder_streams, NULL, 0u, &attention_count,
        route_indices, route_weights, mtp->config.decoder.moe.top_k, &trace);
    status = map_decoder(decoder_status);
    if (status != DSV4_MTP_OK) goto cleanup;
    if (attention_count != 0u) {
        status = DSV4_MTP_DEPENDENCY_FAILED;
        goto cleanup;
    }

    status = hyper_head(mtp, weights, decoder_streams, collapsed);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = weighted_rms(collapsed, weights->output_norm_weight,
                          mtp->config.hidden_size, mtp->config.norm_eps, final_hidden);
    if (status != DSV4_MTP_OK) goto cleanup;
    status = lm_head(mtp, final_hidden, weights->lm_head_weight, logits);
    if (status != DSV4_MTP_OK) goto cleanup;

    {
        DSV4DecoderLayer *old = mtp->decoder;
        mtp->decoder = decoder_work;
        decoder_work = NULL;
        dsv4_decoder_layer_destroy(old);
    }
    ++mtp->next_position;
    memcpy(out_streams, decoder_streams, stream_count * sizeof(float));
    memcpy(out_logits, logits, mtp->config.vocab_size * sizeof(float));
    memcpy(out_route_indices, route_indices,
           mtp->config.decoder.moe.top_k * sizeof(uint32_t));
    memcpy(out_route_weights, route_weights,
           mtp->config.decoder.moe.top_k * sizeof(float));
    out_trace->decoder = trace;
    status = DSV4_MTP_OK;

cleanup:
    dsv4_decoder_layer_destroy(decoder_work);
    free(embedding_normed);
    free(embedding_projected);
    free(hidden_normed);
    free(hidden_projected);
    free(combined);
    free(decoder_streams);
    free(collapsed);
    free(final_hidden);
    free(logits);
    free(route_indices);
    free(route_weights);
    return status;
}

const char *dsv4_mtp_status_string(DSV4MTPStatus status)
{
    switch (status) {
        case DSV4_MTP_OK: return "ok";
        case DSV4_MTP_INVALID_ARGUMENT: return "invalid argument";
        case DSV4_MTP_NONFINITE: return "non-finite input, weight or result";
        case DSV4_MTP_OVERFLOW: return "size overflow";
        case DSV4_MTP_ALLOCATION_FAILED: return "allocation failed";
        case DSV4_MTP_DEPENDENCY_FAILED: return "dependency failed";
        default: return "unknown status";
    }
}

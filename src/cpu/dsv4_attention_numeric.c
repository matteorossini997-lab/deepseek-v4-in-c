#include "dsv4_attention_numeric.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int dsv4_checked_mul_size(size_t a, size_t b, size_t *out) {
    if (out == NULL) {
        return 0;
    }
    if (a != 0u && b > SIZE_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int dsv4_all_finite(const float *values, size_t count) {
    size_t i;
    if (values == NULL) {
        return 0;
    }
    for (i = 0u; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static DSV4AttentionNumericStatus dsv4_validate_rope(
    size_t dim,
    size_t rope_dim,
    float theta) {
    if (dim == 0u || rope_dim > dim || (rope_dim % 2u) != 0u) {
        return DSV4_AN_INVALID_ARGUMENT;
    }
    if (!isfinite(theta) || theta <= 0.0f) {
        return DSV4_AN_INVALID_ARGUMENT;
    }
    return DSV4_AN_OK;
}

DSV4AttentionNumericStatus dsv4_partial_rope_f32(
    float *values,
    size_t row_count,
    size_t dim,
    size_t rope_dim,
    int64_t position,
    float theta) {
    size_t total;
    size_t row;
    size_t pair;
    size_t start;
    DSV4AttentionNumericStatus status;

    if (values == NULL || row_count == 0u) {
        return DSV4_AN_INVALID_ARGUMENT;
    }
    status = dsv4_validate_rope(dim, rope_dim, theta);
    if (status != DSV4_AN_OK) {
        return status;
    }
    if (!dsv4_checked_mul_size(row_count, dim, &total)) {
        return DSV4_AN_OVERFLOW;
    }
    if (!dsv4_all_finite(values, total)) {
        return DSV4_AN_NONFINITE;
    }
    if (rope_dim == 0u) {
        return DSV4_AN_OK;
    }

    start = dim - rope_dim;
    for (row = 0u; row < row_count; ++row) {
        float *row_values = values + row * dim;
        for (pair = 0u; pair < rope_dim / 2u; ++pair) {
            const double exponent = -((double)(2u * pair) / (double)rope_dim);
            const double frequency = pow((double)theta, exponent);
            const double angle = (double)position * frequency;
            const float cosine = (float)cos(angle);
            const float sine = (float)sin(angle);
            const size_t first_index = start + pair * 2u;
            const size_t second_index = first_index + 1u;
            const float first = row_values[first_index];
            const float second = row_values[second_index];
            row_values[first_index] = first * cosine - second * sine;
            row_values[second_index] = first * sine + second * cosine;
        }
    }
    return dsv4_all_finite(values, total) ? DSV4_AN_OK : DSV4_AN_NONFINITE;
}

DSV4AttentionNumericStatus dsv4_sink_attention_shared_kv_f32(
    const float *q,
    const float *kv,
    const float *sinks,
    size_t heads,
    size_t tokens,
    size_t head_dim,
    float scale,
    float *out) {
    size_t q_count;
    size_t kv_count;
    size_t out_bytes;
    size_t head;
    size_t token;
    size_t dim;
    float *tmp;

    if (q == NULL || kv == NULL || sinks == NULL || out == NULL || heads == 0u ||
        tokens == 0u || head_dim == 0u || !isfinite(scale) || scale <= 0.0f) {
        return DSV4_AN_INVALID_ARGUMENT;
    }
    if (!dsv4_checked_mul_size(heads, head_dim, &q_count) ||
        !dsv4_checked_mul_size(tokens, head_dim, &kv_count) ||
        !dsv4_checked_mul_size(q_count, sizeof(float), &out_bytes)) {
        return DSV4_AN_OVERFLOW;
    }
    if (!dsv4_all_finite(q, q_count) || !dsv4_all_finite(kv, kv_count) ||
        !dsv4_all_finite(sinks, heads)) {
        return DSV4_AN_NONFINITE;
    }

    tmp = (float *)malloc(out_bytes);
    if (tmp == NULL) {
        return DSV4_AN_ALLOCATION_FAILED;
    }

    for (head = 0u; head < heads; ++head) {
        double maximum = (double)sinks[head];
        double denominator;
        const float *q_head = q + head * head_dim;

        for (token = 0u; token < tokens; ++token) {
            const float *kv_token = kv + token * head_dim;
            double dot = 0.0;
            for (dim = 0u; dim < head_dim; ++dim) {
                dot += (double)q_head[dim] * (double)kv_token[dim];
            }
            dot *= (double)scale;
            if (dot > maximum) {
                maximum = dot;
            }
        }

        denominator = exp((double)sinks[head] - maximum);
        for (token = 0u; token < tokens; ++token) {
            const float *kv_token = kv + token * head_dim;
            double dot = 0.0;
            for (dim = 0u; dim < head_dim; ++dim) {
                dot += (double)q_head[dim] * (double)kv_token[dim];
            }
            denominator += exp(dot * (double)scale - maximum);
        }
        if (!isfinite(denominator) || denominator <= 0.0) {
            free(tmp);
            return DSV4_AN_NONFINITE;
        }

        for (dim = 0u; dim < head_dim; ++dim) {
            double weighted = 0.0;
            for (token = 0u; token < tokens; ++token) {
                const float *kv_token = kv + token * head_dim;
                double dot = 0.0;
                size_t inner;
                for (inner = 0u; inner < head_dim; ++inner) {
                    dot += (double)q_head[inner] * (double)kv_token[inner];
                }
                weighted += exp(dot * (double)scale - maximum) * (double)kv_token[dim];
            }
            tmp[head * head_dim + dim] = (float)(weighted / denominator);
        }
    }

    if (!dsv4_all_finite(tmp, q_count)) {
        free(tmp);
        return DSV4_AN_NONFINITE;
    }
    memcpy(out, tmp, out_bytes);
    free(tmp);
    return DSV4_AN_OK;
}

DSV4AttentionNumericStatus dsv4_compressor_numeric_init(
    DSV4CompressorNumericState *state,
    size_t rate,
    size_t output_dim,
    size_t rope_dim,
    int overlap,
    float rms_eps,
    float theta,
    float *previous_ca_kv,
    float *previous_ca_gate) {
    DSV4AttentionNumericStatus status;
    size_t previous_count;
    size_t previous_bytes;

    if (state == NULL || rate == 0u || output_dim == 0u ||
        (overlap != 0 && overlap != 1) || !isfinite(rms_eps) || rms_eps <= 0.0f) {
        return DSV4_AN_INVALID_ARGUMENT;
    }
    status = dsv4_validate_rope(output_dim, rope_dim, theta);
    if (status != DSV4_AN_OK) {
        return status;
    }
    if (overlap != 0) {
        if (previous_ca_kv == NULL || previous_ca_gate == NULL) {
            return DSV4_AN_INVALID_ARGUMENT;
        }
        if (!dsv4_checked_mul_size(rate, output_dim, &previous_count) ||
            !dsv4_checked_mul_size(previous_count, sizeof(float), &previous_bytes)) {
            return DSV4_AN_OVERFLOW;
        }
        memset(previous_ca_kv, 0, previous_bytes);
        memset(previous_ca_gate, 0, previous_bytes);
    }

    state->rate = rate;
    state->output_dim = output_dim;
    state->rope_dim = rope_dim;
    state->overlap = overlap;
    state->rms_eps = rms_eps;
    state->theta = theta;
    state->emitted = 0u;
    state->has_previous = 0;
    state->previous_ca_kv = previous_ca_kv;
    state->previous_ca_gate = previous_ca_gate;
    return DSV4_AN_OK;
}

static double dsv4_gate_value(
    const float *gate,
    const float *bias,
    size_t index) {
    return (double)gate[index] + (double)bias[index];
}

DSV4AttentionNumericStatus dsv4_compress_projected_window_f32(
    DSV4CompressorNumericState *state,
    const float *projected_kv,
    const float *projected_gate,
    const float *position_bias,
    const float *norm_weight,
    float *out) {
    size_t projection_dim;
    size_t projected_count;
    size_t out_bytes;
    size_t dim;
    size_t position;
    float *tmp;
    double mean_square = 0.0;
    int64_t rope_position;
    DSV4AttentionNumericStatus status;

    if (state == NULL || projected_kv == NULL || projected_gate == NULL ||
        position_bias == NULL || norm_weight == NULL || out == NULL ||
        state->rate == 0u || state->output_dim == 0u) {
        return DSV4_AN_INVALID_ARGUMENT;
    }
    if (state->overlap != 0 &&
        (state->previous_ca_kv == NULL || state->previous_ca_gate == NULL)) {
        return DSV4_AN_INVALID_ARGUMENT;
    }
    if (state->output_dim > SIZE_MAX / (state->overlap != 0 ? 2u : 1u)) {
        return DSV4_AN_OVERFLOW;
    }
    projection_dim = state->output_dim * (state->overlap != 0 ? 2u : 1u);
    if (!dsv4_checked_mul_size(state->rate, projection_dim, &projected_count) ||
        !dsv4_checked_mul_size(state->output_dim, sizeof(float), &out_bytes)) {
        return DSV4_AN_OVERFLOW;
    }
    if (!dsv4_all_finite(projected_kv, projected_count) ||
        !dsv4_all_finite(projected_gate, projected_count) ||
        !dsv4_all_finite(position_bias, projected_count) ||
        !dsv4_all_finite(norm_weight, state->output_dim)) {
        return DSV4_AN_NONFINITE;
    }
    if (state->overlap != 0 && state->has_previous != 0) {
        size_t previous_count;
        if (!dsv4_checked_mul_size(state->rate, state->output_dim, &previous_count)) {
            return DSV4_AN_OVERFLOW;
        }
        if (!dsv4_all_finite(state->previous_ca_kv, previous_count) ||
            !dsv4_all_finite(state->previous_ca_gate, previous_count)) {
            return DSV4_AN_NONFINITE;
        }
    }
    if (state->emitted == UINT64_MAX ||
        (state->rate != 0u && state->emitted > (uint64_t)INT64_MAX / (uint64_t)state->rate)) {
        return DSV4_AN_OVERFLOW;
    }

    tmp = (float *)malloc(out_bytes);
    if (tmp == NULL) {
        return DSV4_AN_ALLOCATION_FAILED;
    }

    for (dim = 0u; dim < state->output_dim; ++dim) {
        double maximum = -DBL_MAX;
        double denominator = 0.0;
        double weighted = 0.0;

        if (state->overlap != 0 && state->has_previous != 0) {
            for (position = 0u; position < state->rate; ++position) {
                const size_t index = position * state->output_dim + dim;
                const double value = (double)state->previous_ca_gate[index];
                if (value > maximum) {
                    maximum = value;
                }
            }
        }
        for (position = 0u; position < state->rate; ++position) {
            const size_t channel = state->overlap != 0 ? state->output_dim + dim : dim;
            const size_t index = position * projection_dim + channel;
            const double value = dsv4_gate_value(projected_gate, position_bias, index);
            if (value > maximum) {
                maximum = value;
            }
        }

        if (state->overlap != 0 && state->has_previous != 0) {
            for (position = 0u; position < state->rate; ++position) {
                const size_t index = position * state->output_dim + dim;
                const double mass = exp((double)state->previous_ca_gate[index] - maximum);
                denominator += mass;
                weighted += mass * (double)state->previous_ca_kv[index];
            }
        }
        for (position = 0u; position < state->rate; ++position) {
            const size_t channel = state->overlap != 0 ? state->output_dim + dim : dim;
            const size_t index = position * projection_dim + channel;
            const double mass = exp(dsv4_gate_value(projected_gate, position_bias, index) - maximum);
            denominator += mass;
            weighted += mass * (double)projected_kv[index];
        }
        if (!isfinite(denominator) || denominator <= 0.0 || !isfinite(weighted)) {
            free(tmp);
            return DSV4_AN_NONFINITE;
        }
        tmp[dim] = (float)(weighted / denominator);
        mean_square += (double)tmp[dim] * (double)tmp[dim];
    }

    mean_square /= (double)state->output_dim;
    if (!isfinite(mean_square) || mean_square + (double)state->rms_eps <= 0.0) {
        free(tmp);
        return DSV4_AN_NONFINITE;
    }
    {
        const double inverse_rms = 1.0 / sqrt(mean_square + (double)state->rms_eps);
        for (dim = 0u; dim < state->output_dim; ++dim) {
            tmp[dim] = (float)((double)tmp[dim] * inverse_rms * (double)norm_weight[dim]);
        }
    }

    rope_position = (int64_t)(state->emitted * (uint64_t)state->rate);
    status = dsv4_partial_rope_f32(
        tmp, 1u, state->output_dim, state->rope_dim, rope_position, state->theta);
    if (status != DSV4_AN_OK) {
        free(tmp);
        return status;
    }

    if (state->overlap != 0) {
        for (position = 0u; position < state->rate; ++position) {
            for (dim = 0u; dim < state->output_dim; ++dim) {
                const size_t source = position * projection_dim + dim;
                const size_t destination = position * state->output_dim + dim;
                state->previous_ca_kv[destination] = projected_kv[source];
                state->previous_ca_gate[destination] =
                    (float)dsv4_gate_value(projected_gate, position_bias, source);
            }
        }
        state->has_previous = 1;
    }
    state->emitted += 1u;
    memcpy(out, tmp, out_bytes);
    free(tmp);
    return DSV4_AN_OK;
}

const char *dsv4_attention_numeric_status_string(DSV4AttentionNumericStatus status) {
    switch (status) {
        case DSV4_AN_OK:
            return "ok";
        case DSV4_AN_INVALID_ARGUMENT:
            return "invalid argument";
        case DSV4_AN_NONFINITE:
            return "non-finite input or result";
        case DSV4_AN_OVERFLOW:
            return "size or position overflow";
        case DSV4_AN_ALLOCATION_FAILED:
            return "allocation failed";
        default:
            return "unknown status";
    }
}

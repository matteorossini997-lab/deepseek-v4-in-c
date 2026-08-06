#include "dsv4_cpu_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int all_finite(const float *values, size_t count)
{
    if (!values) return 0;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

static size_t gcd_size(size_t a, size_t b)
{
    while (b != 0) {
        size_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

static float sigmoid_stable(float x)
{
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

float dsv4_cpu_sqrtsoftplus(float x)
{
    if (isnan(x)) return NAN;
    if (x == INFINITY) return INFINITY;
    if (x == -INFINITY) return 0.0f;
    float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    return sqrtf(softplus);
}

int dsv4_cpu_hash_route(uint64_t token_id, uint32_t layer_idx,
                        size_t num_experts, size_t top_k,
                        uint32_t *indices)
{
    if (!indices || num_experts == 0 || num_experts > UINT32_MAX ||
        top_k == 0 || top_k > num_experts)
        return DSV4_CPU_INVALID;

    size_t cycle = num_experts / gcd_size(num_experts, 7u);
    if (top_k > cycle) return DSV4_CPU_INVALID;

    uint64_t mod = (uint64_t)num_experts;
    uint64_t start = ((token_id % mod) * 5u +
                      ((uint64_t)layer_idx % mod) * 3u) % mod;
    for (size_t slot = 0; slot < top_k; ++slot)
        indices[slot] = (uint32_t)((start + (slot % mod) * 7u) % mod);
    return DSV4_CPU_OK;
}

typedef struct {
    float corrected;
    uint32_t expert;
} RouteCandidate;

static int candidate_better(float corrected, uint32_t expert,
                            const RouteCandidate *other)
{
    return corrected > other->corrected ||
           (corrected == other->corrected && expert < other->expert);
}

int dsv4_cpu_learned_route(const float *hidden,
                           const float *expert_weight,
                           const float *correction_bias,
                           size_t hidden_size,
                           size_t num_experts,
                           size_t top_k,
                           float route_scale,
                           float *logits,
                           uint32_t *indices,
                           float *route_weights)
{
    size_t weight_count = 0;
    if (!hidden || !expert_weight || !correction_bias || !indices ||
        !route_weights || hidden_size == 0 || num_experts == 0 ||
        num_experts > UINT32_MAX || top_k == 0 || top_k > num_experts ||
        !isfinite(route_scale) || route_scale < 0.0f ||
        !mul_size(hidden_size, num_experts, &weight_count) ||
        !all_finite(hidden, hidden_size) ||
        !all_finite(expert_weight, weight_count) ||
        !all_finite(correction_bias, num_experts))
        return DSV4_CPU_INVALID;

    float *scores = (float *)malloc(num_experts * sizeof(*scores));
    RouteCandidate *best = (RouteCandidate *)malloc(top_k * sizeof(*best));
    if (!scores || !best) {
        free(scores);
        free(best);
        return DSV4_CPU_NOMEM;
    }

    size_t nbest = 0;
    for (size_t expert = 0; expert < num_experts; ++expert) {
        const float *row = expert_weight + expert * hidden_size;
        float logit = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i)
            logit += hidden[i] * row[i];
        float score = dsv4_cpu_sqrtsoftplus(logit);
        float corrected = score + correction_bias[expert];
        if (!isfinite(logit) || !isfinite(score) || !isfinite(corrected)) {
            free(scores);
            free(best);
            return DSV4_CPU_NUMERIC;
        }
        if (logits) logits[expert] = logit;
        scores[expert] = score;

        size_t pos = nbest;
        while (pos > 0 && candidate_better(corrected, (uint32_t)expert,
                                            &best[pos - 1]))
            --pos;
        if (pos < top_k) {
            size_t end = nbest < top_k ? nbest : top_k - 1;
            while (end > pos) {
                best[end] = best[end - 1];
                --end;
            }
            best[pos].corrected = corrected;
            best[pos].expert = (uint32_t)expert;
            if (nbest < top_k) ++nbest;
        }
    }

    float sum = 0.0f;
    for (size_t slot = 0; slot < top_k; ++slot) {
        indices[slot] = best[slot].expert;
        route_weights[slot] = scores[best[slot].expert];
        sum += route_weights[slot];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) {
        free(scores);
        free(best);
        return DSV4_CPU_NUMERIC;
    }
    for (size_t slot = 0; slot < top_k; ++slot)
        route_weights[slot] = route_weights[slot] / (sum + 1e-20f) * route_scale;

    free(scores);
    free(best);
    return DSV4_CPU_OK;
}

static int normalize_rows(float *matrix, size_t n, float eps)
{
    for (size_t row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (size_t col = 0; col < n; ++col)
            sum += matrix[row * n + col];
        float denom = sum + eps;
        if (!(denom > 0.0f) || !isfinite(denom)) return 0;
        for (size_t col = 0; col < n; ++col)
            matrix[row * n + col] /= denom;
    }
    return 1;
}

static int normalize_columns(float *matrix, size_t n, float eps)
{
    for (size_t col = 0; col < n; ++col) {
        float sum = 0.0f;
        for (size_t row = 0; row < n; ++row)
            sum += matrix[row * n + col];
        float denom = sum + eps;
        if (!(denom > 0.0f) || !isfinite(denom)) return 0;
        for (size_t row = 0; row < n; ++row)
            matrix[row * n + col] /= denom;
    }
    return 1;
}

int dsv4_cpu_hyperconnection(const float *hidden_streams,
                             const float *fn,
                             const float *base,
                             const float scale[3],
                             size_t hc_mult,
                             size_t hidden_size,
                             int sinkhorn_iterations,
                             float hc_eps,
                             float norm_eps,
                             float *post,
                             float *comb,
                             float *collapsed)
{
    size_t input_size = 0, comb_size = 0, mix = 0, fn_count = 0;
    if (!hidden_streams || !fn || !base || !scale || !post || !comb ||
        !collapsed || hc_mult == 0 || hidden_size == 0 ||
        sinkhorn_iterations < 1 || !isfinite(hc_eps) || hc_eps <= 0.0f ||
        !isfinite(norm_eps) || norm_eps <= 0.0f ||
        !mul_size(hc_mult, hidden_size, &input_size) ||
        !mul_size(hc_mult, hc_mult, &comb_size) ||
        comb_size > SIZE_MAX - 2u * hc_mult)
        return DSV4_CPU_INVALID;
    mix = 2u * hc_mult + comb_size;
    if (!mul_size(mix, input_size, &fn_count) ||
        !all_finite(hidden_streams, input_size) ||
        !all_finite(fn, fn_count) || !all_finite(base, mix) ||
        !all_finite(scale, 3))
        return DSV4_CPU_INVALID;

    float *normed = (float *)malloc(input_size * sizeof(*normed));
    float *linear = (float *)malloc(mix * sizeof(*linear));
    float *pre = (float *)malloc(hc_mult * sizeof(*pre));
    if (!normed || !linear || !pre) {
        free(normed);
        free(linear);
        free(pre);
        return DSV4_CPU_NOMEM;
    }

    float square_sum = 0.0f;
    for (size_t i = 0; i < input_size; ++i)
        square_sum += hidden_streams[i] * hidden_streams[i];
    float inv_rms = 1.0f / sqrtf(square_sum / (float)input_size + norm_eps);
    if (!isfinite(inv_rms)) goto numeric;
    for (size_t i = 0; i < input_size; ++i)
        normed[i] = hidden_streams[i] * inv_rms;

    for (size_t out = 0; out < mix; ++out) {
        float value = 0.0f;
        const float *row = fn + out * input_size;
        for (size_t i = 0; i < input_size; ++i)
            value += normed[i] * row[i];
        linear[out] = value;
    }

    for (size_t i = 0; i < hc_mult; ++i) {
        pre[i] = sigmoid_stable(linear[i] * scale[0] + base[i]) + hc_eps;
        post[i] = 2.0f * sigmoid_stable(linear[hc_mult + i] * scale[1] +
                                        base[hc_mult + i]);
    }

    const size_t comb_offset = 2u * hc_mult;
    for (size_t row = 0; row < hc_mult; ++row) {
        float max_logit = -INFINITY;
        for (size_t col = 0; col < hc_mult; ++col) {
            size_t k = row * hc_mult + col;
            float logit = linear[comb_offset + k] * scale[2] +
                          base[comb_offset + k];
            comb[k] = logit;
            if (logit > max_logit) max_logit = logit;
        }
        float sum = 0.0f;
        for (size_t col = 0; col < hc_mult; ++col) {
            size_t k = row * hc_mult + col;
            comb[k] = expf(comb[k] - max_logit);
            sum += comb[k];
        }
        if (!(sum > 0.0f) || !isfinite(sum)) goto numeric;
        for (size_t col = 0; col < hc_mult; ++col) {
            size_t k = row * hc_mult + col;
            comb[k] = comb[k] / sum + hc_eps;
        }
    }

    if (!normalize_columns(comb, hc_mult, hc_eps)) goto numeric;
    for (int iter = 0; iter < sinkhorn_iterations - 1; ++iter) {
        if (!normalize_rows(comb, hc_mult, hc_eps) ||
            !normalize_columns(comb, hc_mult, hc_eps))
            goto numeric;
    }
    if (!normalize_rows(comb, hc_mult, hc_eps)) goto numeric;

    for (size_t h = 0; h < hidden_size; ++h) {
        float value = 0.0f;
        for (size_t stream = 0; stream < hc_mult; ++stream)
            value += pre[stream] * hidden_streams[stream * hidden_size + h];
        collapsed[h] = value;
    }

    free(normed);
    free(linear);
    free(pre);
    return DSV4_CPU_OK;

numeric:
    free(normed);
    free(linear);
    free(pre);
    return DSV4_CPU_NUMERIC;
}

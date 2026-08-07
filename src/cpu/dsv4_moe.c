#include "dsv4_moe.h"

#include "dsv4_cpu_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static DSV4MoEStatus alloc_floats(float **out, size_t count, int zero)
{
    size_t bytes;
    void *memory;
    if (out == NULL) return DSV4_MOE_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_MOE_OK;
    if (!checked_mul_size(count, sizeof(float), &bytes)) return DSV4_MOE_OVERFLOW;
    memory = zero != 0 ? calloc(1u, bytes) : malloc(bytes);
    if (memory == NULL) return DSV4_MOE_ALLOCATION_FAILED;
    *out = (float *)memory;
    return DSV4_MOE_OK;
}

static DSV4MoEStatus alloc_u32(uint32_t **out, size_t count)
{
    size_t bytes;
    if (out == NULL) return DSV4_MOE_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_MOE_OK;
    if (!checked_mul_size(count, sizeof(uint32_t), &bytes)) return DSV4_MOE_OVERFLOW;
    *out = (uint32_t *)malloc(bytes);
    return *out == NULL ? DSV4_MOE_ALLOCATION_FAILED : DSV4_MOE_OK;
}

static DSV4MoEStatus validate_config(const DSV4MoEConfig *config)
{
    size_t count;
    if (config == NULL || config->hidden_size == 0u ||
        config->intermediate_size == 0u || config->num_routed_experts == 0u ||
        config->num_routed_experts > UINT32_MAX || config->top_k == 0u ||
        config->top_k > config->num_routed_experts || config->vocab_size == 0u ||
        !isfinite(config->route_scale) || config->route_scale < 0.0f ||
        !isfinite(config->swiglu_limit) || config->swiglu_limit <= 0.0f)
        return DSV4_MOE_INVALID_ARGUMENT;
    if (!checked_mul_size(config->num_routed_experts, config->hidden_size, &count) ||
        !checked_mul_size(config->intermediate_size, config->hidden_size, &count) ||
        !checked_mul_size(config->hidden_size, config->intermediate_size, &count) ||
        !checked_mul_size(config->vocab_size, config->top_k, &count))
        return DSV4_MOE_OVERFLOW;
    return DSV4_MOE_OK;
}

static DSV4MoEStatus validate_common_weights(const DSV4MoEConfig *config,
                                             const DSV4MoEWeights *weights,
                                             const float *hidden)
{
    size_t count;
    if (weights == NULL || hidden == NULL || !all_finite(hidden, config->hidden_size))
        return weights == NULL || hidden == NULL ? DSV4_MOE_INVALID_ARGUMENT
                                                  : DSV4_MOE_NONFINITE;
    if (!checked_mul_size(config->num_routed_experts, config->hidden_size, &count))
        return DSV4_MOE_OVERFLOW;
    if (weights->router_weight == NULL) return DSV4_MOE_INVALID_ARGUMENT;
    if (!all_finite(weights->router_weight, count)) return DSV4_MOE_NONFINITE;

    if (!checked_mul_size(config->intermediate_size, config->hidden_size, &count))
        return DSV4_MOE_OVERFLOW;
    if (weights->shared_gate_weight == NULL || weights->shared_up_weight == NULL)
        return DSV4_MOE_INVALID_ARGUMENT;
    if (!all_finite(weights->shared_gate_weight, count) ||
        !all_finite(weights->shared_up_weight, count))
        return DSV4_MOE_NONFINITE;

    if (!checked_mul_size(config->hidden_size, config->intermediate_size, &count))
        return DSV4_MOE_OVERFLOW;
    if (weights->shared_down_weight == NULL) return DSV4_MOE_INVALID_ARGUMENT;
    if (!all_finite(weights->shared_down_weight, count)) return DSV4_MOE_NONFINITE;

    if (weights->expert_gate_weight == NULL || weights->expert_up_weight == NULL ||
        weights->expert_down_weight == NULL)
        return DSV4_MOE_INVALID_ARGUMENT;
    return DSV4_MOE_OK;
}

static float sigmoid_stable(float x)
{
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static DSV4MoEStatus expert_forward(const float *hidden,
                                    const float *gate_weight,
                                    const float *up_weight,
                                    const float *down_weight,
                                    size_t hidden_size,
                                    size_t intermediate_size,
                                    float limit,
                                    float *out)
{
    float *activated = NULL;
    DSV4MoEStatus status;
    size_t row;
    size_t column;
    size_t gate_count;
    size_t down_count;

    if (hidden == NULL || gate_weight == NULL || up_weight == NULL ||
        down_weight == NULL || out == NULL || hidden_size == 0u ||
        intermediate_size == 0u || !isfinite(limit) || limit <= 0.0f)
        return DSV4_MOE_INVALID_ARGUMENT;
    if (!checked_mul_size(intermediate_size, hidden_size, &gate_count) ||
        !checked_mul_size(hidden_size, intermediate_size, &down_count))
        return DSV4_MOE_OVERFLOW;
    if (!all_finite(gate_weight, gate_count) || !all_finite(up_weight, gate_count) ||
        !all_finite(down_weight, down_count))
        return DSV4_MOE_NONFINITE;

    status = alloc_floats(&activated, intermediate_size, 0);
    if (status != DSV4_MOE_OK) return status;

    for (row = 0u; row < intermediate_size; ++row) {
        float gate = 0.0f;
        float up = 0.0f;
        const float *gate_row = gate_weight + row * hidden_size;
        const float *up_row = up_weight + row * hidden_size;
        for (column = 0u; column < hidden_size; ++column) {
            gate += gate_row[column] * hidden[column];
            up += up_row[column] * hidden[column];
        }
        if (!isfinite(gate) || !isfinite(up)) {
            status = DSV4_MOE_NONFINITE;
            goto cleanup;
        }
        if (gate > limit) gate = limit;
        if (up > limit) up = limit;
        if (up < -limit) up = -limit;
        activated[row] = gate * sigmoid_stable(gate) * up;
        if (!isfinite(activated[row])) {
            status = DSV4_MOE_NONFINITE;
            goto cleanup;
        }
    }

    for (row = 0u; row < hidden_size; ++row) {
        float sum = 0.0f;
        const float *down_row = down_weight + row * intermediate_size;
        for (column = 0u; column < intermediate_size; ++column)
            sum += down_row[column] * activated[column];
        if (!isfinite(sum)) {
            status = DSV4_MOE_NONFINITE;
            goto cleanup;
        }
        out[row] = sum;
    }
    status = DSV4_MOE_OK;

cleanup:
    free(activated);
    return status;
}

static DSV4MoEStatus hash_route(const DSV4MoEConfig *config,
                                const DSV4MoEWeights *weights,
                                uint64_t token_id,
                                const float *hidden,
                                uint32_t *indices,
                                float *route_weights)
{
    float *scores = NULL;
    DSV4MoEStatus status;
    size_t expert;
    size_t dim;
    size_t slot;
    float sum = 0.0f;
    const uint32_t *row;

    if (weights->hash_table == NULL || token_id >= (uint64_t)config->vocab_size)
        return DSV4_MOE_INVALID_ARGUMENT;
    status = alloc_floats(&scores, config->num_routed_experts, 0);
    if (status != DSV4_MOE_OK) return status;

    for (expert = 0u; expert < config->num_routed_experts; ++expert) {
        const float *router_row = weights->router_weight + expert * config->hidden_size;
        float logit = 0.0f;
        for (dim = 0u; dim < config->hidden_size; ++dim)
            logit += hidden[dim] * router_row[dim];
        scores[expert] = dsv4_cpu_sqrtsoftplus(logit);
        if (!isfinite(logit) || !isfinite(scores[expert])) {
            status = DSV4_MOE_NONFINITE;
            goto cleanup;
        }
    }

    row = weights->hash_table + (size_t)token_id * config->top_k;
    for (slot = 0u; slot < config->top_k; ++slot) {
        const uint32_t expert_index = row[slot];
        if ((size_t)expert_index >= config->num_routed_experts) {
            status = DSV4_MOE_INVALID_ARGUMENT;
            goto cleanup;
        }
        indices[slot] = expert_index;
        route_weights[slot] = scores[expert_index];
        sum += route_weights[slot];
    }
    if (!(sum > 0.0f) || !isfinite(sum)) {
        status = DSV4_MOE_NONFINITE;
        goto cleanup;
    }
    for (slot = 0u; slot < config->top_k; ++slot) {
        route_weights[slot] = route_weights[slot] / (sum + 1e-20f) * config->route_scale;
        if (!isfinite(route_weights[slot])) {
            status = DSV4_MOE_NONFINITE;
            goto cleanup;
        }
    }
    status = DSV4_MOE_OK;

cleanup:
    free(scores);
    return status;
}

static DSV4MoEStatus learned_route(const DSV4MoEConfig *config,
                                   const DSV4MoEWeights *weights,
                                   const float *hidden,
                                   uint32_t *indices,
                                   float *route_weights)
{
    int rc;
    if (weights->correction_bias == NULL) return DSV4_MOE_INVALID_ARGUMENT;
    if (!all_finite(weights->correction_bias, config->num_routed_experts))
        return DSV4_MOE_NONFINITE;
    rc = dsv4_cpu_learned_route(hidden,
                                weights->router_weight,
                                weights->correction_bias,
                                config->hidden_size,
                                config->num_routed_experts,
                                config->top_k,
                                config->route_scale,
                                NULL,
                                indices,
                                route_weights);
    if (rc == DSV4_CPU_OK) return DSV4_MOE_OK;
    if (rc == DSV4_CPU_NOMEM) return DSV4_MOE_ALLOCATION_FAILED;
    if (rc == DSV4_CPU_NUMERIC) return DSV4_MOE_NONFINITE;
    return DSV4_MOE_DEPENDENCY_FAILED;
}

DSV4MoEStatus dsv4_moe_forward_f32(const DSV4MoEConfig *config,
                                   const DSV4MoEWeights *weights,
                                   uint32_t layer_idx,
                                   uint64_t token_id,
                                   const float *hidden,
                                   float *out_hidden,
                                   uint32_t *out_route_indices,
                                   float *out_route_weights,
                                   size_t route_capacity)
{
    DSV4MoEStatus status;
    uint32_t *indices = NULL;
    float *route_weights = NULL;
    float *output = NULL;
    float *expert_output = NULL;
    float *shared_output = NULL;
    size_t expert_gate_stride;
    size_t expert_down_stride;
    size_t slot;
    size_t dim;

    status = validate_config(config);
    if (status != DSV4_MOE_OK) return status;
    if (out_hidden == NULL || out_route_indices == NULL || out_route_weights == NULL ||
        route_capacity < config->top_k)
        return DSV4_MOE_INVALID_ARGUMENT;
    status = validate_common_weights(config, weights, hidden);
    if (status != DSV4_MOE_OK) return status;

    status = alloc_u32(&indices, config->top_k);
    if (status != DSV4_MOE_OK) goto cleanup;
    status = alloc_floats(&route_weights, config->top_k, 0);
    if (status != DSV4_MOE_OK) goto cleanup;
    status = alloc_floats(&output, config->hidden_size, 1);
    if (status != DSV4_MOE_OK) goto cleanup;
    status = alloc_floats(&expert_output, config->hidden_size, 0);
    if (status != DSV4_MOE_OK) goto cleanup;
    status = alloc_floats(&shared_output, config->hidden_size, 0);
    if (status != DSV4_MOE_OK) goto cleanup;

    if ((size_t)layer_idx < config->num_hash_layers)
        status = hash_route(config, weights, token_id, hidden, indices, route_weights);
    else
        status = learned_route(config, weights, hidden, indices, route_weights);
    if (status != DSV4_MOE_OK) goto cleanup;

    if (!checked_mul_size(config->intermediate_size, config->hidden_size,
                          &expert_gate_stride) ||
        !checked_mul_size(config->hidden_size, config->intermediate_size,
                          &expert_down_stride)) {
        status = DSV4_MOE_OVERFLOW;
        goto cleanup;
    }

    for (slot = 0u; slot < config->top_k; ++slot) {
        const size_t expert = (size_t)indices[slot];
        const float *gate = weights->expert_gate_weight + expert * expert_gate_stride;
        const float *up = weights->expert_up_weight + expert * expert_gate_stride;
        const float *down = weights->expert_down_weight + expert * expert_down_stride;
        status = expert_forward(hidden, gate, up, down,
                                config->hidden_size,
                                config->intermediate_size,
                                config->swiglu_limit,
                                expert_output);
        if (status != DSV4_MOE_OK) goto cleanup;
        for (dim = 0u; dim < config->hidden_size; ++dim) {
            output[dim] += expert_output[dim] * route_weights[slot];
            if (!isfinite(output[dim])) {
                status = DSV4_MOE_NONFINITE;
                goto cleanup;
            }
        }
    }

    status = expert_forward(hidden,
                            weights->shared_gate_weight,
                            weights->shared_up_weight,
                            weights->shared_down_weight,
                            config->hidden_size,
                            config->intermediate_size,
                            config->swiglu_limit,
                            shared_output);
    if (status != DSV4_MOE_OK) goto cleanup;
    for (dim = 0u; dim < config->hidden_size; ++dim) {
        output[dim] += shared_output[dim];
        if (!isfinite(output[dim])) {
            status = DSV4_MOE_NONFINITE;
            goto cleanup;
        }
    }

    memcpy(out_hidden, output, config->hidden_size * sizeof(float));
    memcpy(out_route_indices, indices, config->top_k * sizeof(uint32_t));
    memcpy(out_route_weights, route_weights, config->top_k * sizeof(float));
    status = DSV4_MOE_OK;

cleanup:
    free(indices);
    free(route_weights);
    free(output);
    free(expert_output);
    free(shared_output);
    return status;
}

const char *dsv4_moe_status_string(DSV4MoEStatus status)
{
    switch (status) {
        case DSV4_MOE_OK: return "ok";
        case DSV4_MOE_INVALID_ARGUMENT: return "invalid argument";
        case DSV4_MOE_NONFINITE: return "non-finite input, weight or result";
        case DSV4_MOE_OVERFLOW: return "size overflow";
        case DSV4_MOE_ALLOCATION_FAILED: return "allocation failed";
        case DSV4_MOE_DEPENDENCY_FAILED: return "dependency failed";
        default: return "unknown status";
    }
}

#include "dsv4_attention_runtime.h"

#include "dsv4_attention_numeric.h"
#include "dsv4_attention_step.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct DSV4AttentionRuntime {
    DSV4AttentionRuntimeConfig config;
    uint64_t next_position;

    size_t raw_count;
    float *raw_kv;

    size_t comp_projection_dim;
    size_t comp_buffer_count;
    float *comp_buffer_kv;
    float *comp_buffer_gate;
    float *comp_previous_ca_kv;
    float *comp_previous_ca_gate;
    DSV4CompressorNumericState comp_numeric;
    size_t compressed_count;
    size_t compressed_capacity;
    float *compressed_kv;

    size_t index_projection_dim;
    size_t index_buffer_count;
    float *index_buffer_kv;
    float *index_buffer_gate;
    float *index_previous_ca_kv;
    float *index_previous_ca_gate;
    DSV4CompressorNumericState index_numeric;
    size_t index_compressed_count;
    size_t index_compressed_capacity;
    float *index_compressed;
};

static int checked_mul_size(size_t a, size_t b, size_t *out) {
    if (out == NULL) return 0;
    if (a != 0u && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int checked_add_size(size_t a, size_t b, size_t *out) {
    if (out == NULL || b > SIZE_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static int all_finite(const float *values, size_t count) {
    size_t i;
    if (values == NULL) return 0;
    for (i = 0u; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

static DSV4AttentionRuntimeStatus alloc_floats(float **out, size_t count, int zero) {
    size_t bytes;
    void *memory;
    if (out == NULL) return DSV4_AR_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_AR_OK;
    if (!checked_mul_size(count, sizeof(float), &bytes)) return DSV4_AR_OVERFLOW;
    memory = zero != 0 ? calloc(1u, bytes) : malloc(bytes);
    if (memory == NULL) return DSV4_AR_ALLOCATION_FAILED;
    *out = (float *)memory;
    return DSV4_AR_OK;
}

static DSV4AttentionRuntimeStatus alloc_indices(size_t **out, size_t count) {
    size_t bytes;
    if (out == NULL) return DSV4_AR_INVALID_ARGUMENT;
    *out = NULL;
    if (count == 0u) return DSV4_AR_OK;
    if (!checked_mul_size(count, sizeof(size_t), &bytes)) return DSV4_AR_OVERFLOW;
    *out = (size_t *)malloc(bytes);
    return *out == NULL ? DSV4_AR_ALLOCATION_FAILED : DSV4_AR_OK;
}

static DSV4AttentionRuntimeStatus map_numeric(DSV4AttentionNumericStatus status) {
    switch (status) {
        case DSV4_AN_OK: return DSV4_AR_OK;
        case DSV4_AN_INVALID_ARGUMENT: return DSV4_AR_INVALID_ARGUMENT;
        case DSV4_AN_NONFINITE: return DSV4_AR_NONFINITE;
        case DSV4_AN_OVERFLOW: return DSV4_AR_OVERFLOW;
        case DSV4_AN_ALLOCATION_FAILED: return DSV4_AR_ALLOCATION_FAILED;
        default: return DSV4_AR_DEPENDENCY_FAILED;
    }
}

static DSV4AttentionRuntimeStatus map_step(DSV4AttentionStepStatus status) {
    switch (status) {
        case DSV4_AS_OK: return DSV4_AR_OK;
        case DSV4_AS_INVALID_ARGUMENT: return DSV4_AR_INVALID_ARGUMENT;
        case DSV4_AS_NONFINITE: return DSV4_AR_NONFINITE;
        case DSV4_AS_OVERFLOW: return DSV4_AR_OVERFLOW;
        case DSV4_AS_ALLOCATION_FAILED: return DSV4_AR_ALLOCATION_FAILED;
        case DSV4_AS_DEPENDENCY_FAILED: return DSV4_AR_DEPENDENCY_FAILED;
        default: return DSV4_AR_DEPENDENCY_FAILED;
    }
}

static DSV4AttentionRuntimeStatus validate_config(
    const DSV4AttentionRuntimeConfig *config) {
    size_t total_width;
    size_t ignored;
    if (config == NULL || config->hidden_size == 0u || config->q_lora_rank == 0u ||
        config->num_heads == 0u || config->head_dim == 0u ||
        config->output_groups == 0u || config->output_rank == 0u ||
        config->sliding_window == 0u ||
        (config->compression_rate != 0u && config->compression_rate != 4u &&
         config->compression_rate != 128u) ||
        config->rope_dim > config->head_dim || (config->rope_dim % 2u) != 0u ||
        !isfinite(config->rms_eps) || config->rms_eps <= 0.0f ||
        !isfinite(config->rope_theta) || config->rope_theta <= 0.0f ||
        !isfinite(config->compress_rope_theta) || config->compress_rope_theta <= 0.0f) {
        return DSV4_AR_INVALID_ARGUMENT;
    }
    if (!checked_mul_size(config->num_heads, config->head_dim, &total_width))
        return DSV4_AR_OVERFLOW;
    if (total_width % config->output_groups != 0u)
        return DSV4_AR_INVALID_ARGUMENT;
    if (!checked_mul_size(config->sliding_window, config->head_dim, &ignored) ||
        !checked_mul_size(config->q_lora_rank, config->hidden_size, &ignored) ||
        !checked_mul_size(total_width, config->q_lora_rank, &ignored) ||
        !checked_mul_size(config->head_dim, config->hidden_size, &ignored))
        return DSV4_AR_OVERFLOW;

    if (config->compression_rate == 4u) {
        if (config->index_num_heads == 0u || config->index_head_dim == 0u ||
            config->index_top_k == 0u || config->index_rope_dim > config->index_head_dim ||
            (config->index_rope_dim % 2u) != 0u) {
            return DSV4_AR_INVALID_ARGUMENT;
        }
        if (!checked_mul_size(config->index_num_heads, config->index_head_dim, &ignored) ||
            !checked_mul_size(ignored, config->q_lora_rank, &ignored) ||
            !checked_mul_size(config->index_num_heads, config->hidden_size, &ignored))
            return DSV4_AR_OVERFLOW;
    }
    return DSV4_AR_OK;
}

static void free_members(DSV4AttentionRuntime *runtime) {
    if (runtime == NULL) return;
    free(runtime->raw_kv);
    free(runtime->comp_buffer_kv);
    free(runtime->comp_buffer_gate);
    free(runtime->comp_previous_ca_kv);
    free(runtime->comp_previous_ca_gate);
    free(runtime->compressed_kv);
    free(runtime->index_buffer_kv);
    free(runtime->index_buffer_gate);
    free(runtime->index_previous_ca_kv);
    free(runtime->index_previous_ca_gate);
    free(runtime->index_compressed);
    runtime->raw_kv = NULL;
    runtime->comp_buffer_kv = NULL;
    runtime->comp_buffer_gate = NULL;
    runtime->comp_previous_ca_kv = NULL;
    runtime->comp_previous_ca_gate = NULL;
    runtime->compressed_kv = NULL;
    runtime->index_buffer_kv = NULL;
    runtime->index_buffer_gate = NULL;
    runtime->index_previous_ca_kv = NULL;
    runtime->index_previous_ca_gate = NULL;
    runtime->index_compressed = NULL;
}

DSV4AttentionRuntime *dsv4_attention_runtime_create(
    const DSV4AttentionRuntimeConfig *config,
    DSV4AttentionRuntimeStatus *out_status) {
    DSV4AttentionRuntime *runtime = NULL;
    DSV4AttentionRuntimeStatus status;
    size_t count;
    size_t previous_count;
    int overlap;

    status = validate_config(config);
    if (status != DSV4_AR_OK) goto done;

    runtime = (DSV4AttentionRuntime *)calloc(1u, sizeof(*runtime));
    if (runtime == NULL) {
        status = DSV4_AR_ALLOCATION_FAILED;
        goto done;
    }
    runtime->config = *config;

    if (!checked_mul_size(config->sliding_window, config->head_dim, &count)) {
        status = DSV4_AR_OVERFLOW;
        goto fail;
    }
    status = alloc_floats(&runtime->raw_kv, count, 1);
    if (status != DSV4_AR_OK) goto fail;

    if (config->compression_rate != 0u) {
        overlap = config->compression_rate == 4u ? 1 : 0;
        if (config->head_dim > SIZE_MAX / (overlap != 0 ? 2u : 1u)) {
            status = DSV4_AR_OVERFLOW;
            goto fail;
        }
        runtime->comp_projection_dim = config->head_dim * (overlap != 0 ? 2u : 1u);
        if (!checked_mul_size(config->compression_rate, runtime->comp_projection_dim, &count)) {
            status = DSV4_AR_OVERFLOW;
            goto fail;
        }
        status = alloc_floats(&runtime->comp_buffer_kv, count, 1);
        if (status != DSV4_AR_OK) goto fail;
        status = alloc_floats(&runtime->comp_buffer_gate, count, 1);
        if (status != DSV4_AR_OK) goto fail;

        if (overlap != 0) {
            if (!checked_mul_size(config->compression_rate, config->head_dim, &previous_count)) {
                status = DSV4_AR_OVERFLOW;
                goto fail;
            }
            status = alloc_floats(&runtime->comp_previous_ca_kv, previous_count, 1);
            if (status != DSV4_AR_OK) goto fail;
            status = alloc_floats(&runtime->comp_previous_ca_gate, previous_count, 1);
            if (status != DSV4_AR_OK) goto fail;
        }
        status = map_numeric(dsv4_compressor_numeric_init(
            &runtime->comp_numeric,
            config->compression_rate,
            config->head_dim,
            config->rope_dim,
            overlap,
            config->rms_eps,
            config->compress_rope_theta,
            runtime->comp_previous_ca_kv,
            runtime->comp_previous_ca_gate));
        if (status != DSV4_AR_OK) goto fail;
    }

    if (config->compression_rate == 4u) {
        if (config->index_head_dim > SIZE_MAX / 2u) {
            status = DSV4_AR_OVERFLOW;
            goto fail;
        }
        runtime->index_projection_dim = 2u * config->index_head_dim;
        if (!checked_mul_size(4u, runtime->index_projection_dim, &count)) {
            status = DSV4_AR_OVERFLOW;
            goto fail;
        }
        status = alloc_floats(&runtime->index_buffer_kv, count, 1);
        if (status != DSV4_AR_OK) goto fail;
        status = alloc_floats(&runtime->index_buffer_gate, count, 1);
        if (status != DSV4_AR_OK) goto fail;
        if (!checked_mul_size(4u, config->index_head_dim, &previous_count)) {
            status = DSV4_AR_OVERFLOW;
            goto fail;
        }
        status = alloc_floats(&runtime->index_previous_ca_kv, previous_count, 1);
        if (status != DSV4_AR_OK) goto fail;
        status = alloc_floats(&runtime->index_previous_ca_gate, previous_count, 1);
        if (status != DSV4_AR_OK) goto fail;
        status = map_numeric(dsv4_compressor_numeric_init(
            &runtime->index_numeric,
            4u,
            config->index_head_dim,
            config->index_rope_dim,
            1,
            config->rms_eps,
            config->compress_rope_theta,
            runtime->index_previous_ca_kv,
            runtime->index_previous_ca_gate));
        if (status != DSV4_AR_OK) goto fail;
    }

    status = DSV4_AR_OK;
    goto done;

fail:
    free_members(runtime);
    free(runtime);
    runtime = NULL;
done:
    if (out_status != NULL) *out_status = status;
    return runtime;
}

void dsv4_attention_runtime_destroy(DSV4AttentionRuntime *runtime) {
    if (runtime == NULL) return;
    free_members(runtime);
    free(runtime);
}

static DSV4AttentionRuntimeStatus copy_vectors(
    const float *source,
    size_t count,
    size_t dim,
    float **out,
    size_t *out_capacity) {
    DSV4AttentionRuntimeStatus status;
    size_t elements;
    size_t bytes;
    if (out == NULL || out_capacity == NULL) return DSV4_AR_INVALID_ARGUMENT;
    *out = NULL;
    *out_capacity = 0u;
    if (count == 0u) return DSV4_AR_OK;
    if (source == NULL || !checked_mul_size(count, dim, &elements) ||
        !checked_mul_size(elements, sizeof(float), &bytes))
        return source == NULL ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_OVERFLOW;
    status = alloc_floats(out, elements, 0);
    if (status != DSV4_AR_OK) return status;
    memcpy(*out, source, bytes);
    *out_capacity = count;
    return DSV4_AR_OK;
}

static DSV4AttentionRuntime *clone_runtime(
    const DSV4AttentionRuntime *source,
    DSV4AttentionRuntimeStatus *out_status) {
    DSV4AttentionRuntime *copy;
    DSV4AttentionRuntimeStatus status;
    size_t count;
    size_t bytes;

    if (source == NULL) {
        if (out_status != NULL) *out_status = DSV4_AR_INVALID_ARGUMENT;
        return NULL;
    }
    copy = dsv4_attention_runtime_create(&source->config, &status);
    if (copy == NULL) {
        if (out_status != NULL) *out_status = status;
        return NULL;
    }
    copy->next_position = source->next_position;
    copy->raw_count = source->raw_count;
    if (!checked_mul_size(source->config.sliding_window, source->config.head_dim, &count) ||
        !checked_mul_size(count, sizeof(float), &bytes)) {
        status = DSV4_AR_OVERFLOW;
        goto fail;
    }
    memcpy(copy->raw_kv, source->raw_kv, bytes);

    if (source->config.compression_rate != 0u) {
        copy->comp_buffer_count = source->comp_buffer_count;
        if (!checked_mul_size(source->config.compression_rate,
                              source->comp_projection_dim, &count) ||
            !checked_mul_size(count, sizeof(float), &bytes)) {
            status = DSV4_AR_OVERFLOW;
            goto fail;
        }
        memcpy(copy->comp_buffer_kv, source->comp_buffer_kv, bytes);
        memcpy(copy->comp_buffer_gate, source->comp_buffer_gate, bytes);
        copy->comp_numeric.emitted = source->comp_numeric.emitted;
        copy->comp_numeric.has_previous = source->comp_numeric.has_previous;
        if (source->config.compression_rate == 4u) {
            if (!checked_mul_size(4u, source->config.head_dim, &count) ||
                !checked_mul_size(count, sizeof(float), &bytes)) {
                status = DSV4_AR_OVERFLOW;
                goto fail;
            }
            memcpy(copy->comp_previous_ca_kv, source->comp_previous_ca_kv, bytes);
            memcpy(copy->comp_previous_ca_gate, source->comp_previous_ca_gate, bytes);
        }
        status = copy_vectors(source->compressed_kv, source->compressed_count,
                              source->config.head_dim, &copy->compressed_kv,
                              &copy->compressed_capacity);
        if (status != DSV4_AR_OK) goto fail;
        copy->compressed_count = source->compressed_count;
    }

    if (source->config.compression_rate == 4u) {
        copy->index_buffer_count = source->index_buffer_count;
        if (!checked_mul_size(4u, source->index_projection_dim, &count) ||
            !checked_mul_size(count, sizeof(float), &bytes)) {
            status = DSV4_AR_OVERFLOW;
            goto fail;
        }
        memcpy(copy->index_buffer_kv, source->index_buffer_kv, bytes);
        memcpy(copy->index_buffer_gate, source->index_buffer_gate, bytes);
        copy->index_numeric.emitted = source->index_numeric.emitted;
        copy->index_numeric.has_previous = source->index_numeric.has_previous;
        if (!checked_mul_size(4u, source->config.index_head_dim, &count) ||
            !checked_mul_size(count, sizeof(float), &bytes)) {
            status = DSV4_AR_OVERFLOW;
            goto fail;
        }
        memcpy(copy->index_previous_ca_kv, source->index_previous_ca_kv, bytes);
        memcpy(copy->index_previous_ca_gate, source->index_previous_ca_gate, bytes);
        status = copy_vectors(source->index_compressed, source->index_compressed_count,
                              source->config.index_head_dim, &copy->index_compressed,
                              &copy->index_compressed_capacity);
        if (status != DSV4_AR_OK) goto fail;
        copy->index_compressed_count = source->index_compressed_count;
    }

    if (out_status != NULL) *out_status = DSV4_AR_OK;
    return copy;

fail:
    dsv4_attention_runtime_destroy(copy);
    if (out_status != NULL) *out_status = status;
    return NULL;
}

static void commit_runtime(
    DSV4AttentionRuntime *target,
    DSV4AttentionRuntime *replacement) {
    DSV4AttentionRuntime old = *target;
    *target = *replacement;
    *replacement = old;
    dsv4_attention_runtime_destroy(replacement);
}

DSV4AttentionRuntimeStatus dsv4_attention_runtime_reset(
    DSV4AttentionRuntime *runtime) {
    DSV4AttentionRuntimeStatus status;
    DSV4AttentionRuntime *fresh;
    if (runtime == NULL) return DSV4_AR_INVALID_ARGUMENT;
    fresh = dsv4_attention_runtime_create(&runtime->config, &status);
    if (fresh == NULL) return status;
    commit_runtime(runtime, fresh);
    return DSV4_AR_OK;
}

static DSV4AttentionRuntimeStatus dense_linear(
    const float *input,
    const float *weight,
    size_t input_dim,
    size_t output_dim,
    float *output) {
    size_t row;
    size_t column;
    if (input == NULL || weight == NULL || output == NULL ||
        input_dim == 0u || output_dim == 0u)
        return DSV4_AR_INVALID_ARGUMENT;
    for (row = 0u; row < output_dim; ++row) {
        float sum = 0.0f;
        const float *weight_row = weight + row * input_dim;
        for (column = 0u; column < input_dim; ++column)
            sum += weight_row[column] * input[column];
        if (!isfinite(sum)) return DSV4_AR_NONFINITE;
        output[row] = sum;
    }
    return DSV4_AR_OK;
}

static DSV4AttentionRuntimeStatus weighted_rms_inplace(
    float *values,
    const float *weight,
    size_t dim,
    float eps) {
    size_t i;
    double mean_square = 0.0;
    double inverse;
    if (values == NULL || weight == NULL || dim == 0u ||
        !isfinite(eps) || eps <= 0.0f)
        return DSV4_AR_INVALID_ARGUMENT;
    for (i = 0u; i < dim; ++i)
        mean_square += (double)values[i] * (double)values[i];
    mean_square /= (double)dim;
    if (!isfinite(mean_square) || mean_square + (double)eps <= 0.0)
        return DSV4_AR_NONFINITE;
    inverse = 1.0 / sqrt(mean_square + (double)eps);
    if (!isfinite(inverse)) return DSV4_AR_NONFINITE;
    for (i = 0u; i < dim; ++i) {
        values[i] = (float)((double)values[i] * inverse * (double)weight[i]);
        if (!isfinite(values[i])) return DSV4_AR_NONFINITE;
    }
    return DSV4_AR_OK;
}

static DSV4AttentionRuntimeStatus unweighted_head_rms_inplace(
    float *values,
    size_t heads,
    size_t head_dim,
    float eps) {
    size_t head;
    size_t dim;
    if (values == NULL || heads == 0u || head_dim == 0u ||
        !isfinite(eps) || eps <= 0.0f)
        return DSV4_AR_INVALID_ARGUMENT;
    for (head = 0u; head < heads; ++head) {
        double mean_square = 0.0;
        double inverse;
        float *row = values + head * head_dim;
        for (dim = 0u; dim < head_dim; ++dim)
            mean_square += (double)row[dim] * (double)row[dim];
        mean_square /= (double)head_dim;
        if (!isfinite(mean_square) || mean_square + (double)eps <= 0.0)
            return DSV4_AR_NONFINITE;
        inverse = 1.0 / sqrt(mean_square + (double)eps);
        if (!isfinite(inverse)) return DSV4_AR_NONFINITE;
        for (dim = 0u; dim < head_dim; ++dim) {
            row[dim] = (float)((double)row[dim] * inverse);
            if (!isfinite(row[dim])) return DSV4_AR_NONFINITE;
        }
    }
    return DSV4_AR_OK;
}

static DSV4AttentionRuntimeStatus append_vector(
    float **data,
    size_t *count,
    size_t *capacity,
    size_t dim,
    const float *vector) {
    size_t needed;
    size_t new_capacity;
    size_t elements;
    size_t old_elements;
    size_t old_bytes;
    float *replacement;
    DSV4AttentionRuntimeStatus status;

    if (data == NULL || count == NULL || capacity == NULL || dim == 0u || vector == NULL)
        return DSV4_AR_INVALID_ARGUMENT;
    if (!checked_add_size(*count, 1u, &needed)) return DSV4_AR_OVERFLOW;
    if (needed > *capacity) {
        new_capacity = *capacity == 0u ? 4u : *capacity;
        while (new_capacity < needed) {
            if (new_capacity > SIZE_MAX / 2u) {
                new_capacity = needed;
                break;
            }
            new_capacity *= 2u;
        }
        if (!checked_mul_size(new_capacity, dim, &elements)) return DSV4_AR_OVERFLOW;
        status = alloc_floats(&replacement, elements, 0);
        if (status != DSV4_AR_OK) return status;
        if (*count != 0u) {
            if (!checked_mul_size(*count, dim, &old_elements) ||
                !checked_mul_size(old_elements, sizeof(float), &old_bytes)) {
                free(replacement);
                return DSV4_AR_OVERFLOW;
            }
            memcpy(replacement, *data, old_bytes);
        }
        free(*data);
        *data = replacement;
        *capacity = new_capacity;
    }
    memcpy(*data + (*count * dim), vector, dim * sizeof(float));
    *count = needed;
    return DSV4_AR_OK;
}

static DSV4AttentionRuntimeStatus append_raw_kv(
    DSV4AttentionRuntime *runtime,
    const float *kv) {
    const size_t dim = runtime->config.head_dim;
    const size_t window = runtime->config.sliding_window;
    size_t move_count;
    size_t move_bytes;
    if (runtime == NULL || kv == NULL) return DSV4_AR_INVALID_ARGUMENT;
    if (runtime->raw_count < window) {
        memcpy(runtime->raw_kv + runtime->raw_count * dim, kv, dim * sizeof(float));
        runtime->raw_count += 1u;
        return DSV4_AR_OK;
    }
    if (window > 1u) {
        if (!checked_mul_size(window - 1u, dim, &move_count) ||
            !checked_mul_size(move_count, sizeof(float), &move_bytes))
            return DSV4_AR_OVERFLOW;
        memmove(runtime->raw_kv, runtime->raw_kv + dim, move_bytes);
    }
    memcpy(runtime->raw_kv + (window - 1u) * dim, kv, dim * sizeof(float));
    return DSV4_AR_OK;
}

static DSV4AttentionRuntimeStatus push_attention_compressor(
    DSV4AttentionRuntime *runtime,
    const DSV4AttentionRuntimeWeights *weights,
    const float *hidden) {
    DSV4AttentionRuntimeStatus status;
    float *kv_row;
    float *gate_row;
    float *compressed = NULL;
    size_t row_offset;

    if (runtime->config.compression_rate == 0u) return DSV4_AR_OK;
    row_offset = runtime->comp_buffer_count * runtime->comp_projection_dim;
    kv_row = runtime->comp_buffer_kv + row_offset;
    gate_row = runtime->comp_buffer_gate + row_offset;
    status = dense_linear(hidden, weights->compress_kv_weight,
                          runtime->config.hidden_size,
                          runtime->comp_projection_dim, kv_row);
    if (status != DSV4_AR_OK) return status;
    status = dense_linear(hidden, weights->compress_gate_weight,
                          runtime->config.hidden_size,
                          runtime->comp_projection_dim, gate_row);
    if (status != DSV4_AR_OK) return status;
    runtime->comp_buffer_count += 1u;
    if (runtime->comp_buffer_count < runtime->config.compression_rate)
        return DSV4_AR_OK;
    if (runtime->comp_buffer_count != runtime->config.compression_rate)
        return DSV4_AR_DEPENDENCY_FAILED;

    status = alloc_floats(&compressed, runtime->config.head_dim, 0);
    if (status != DSV4_AR_OK) return status;
    status = map_numeric(dsv4_compress_projected_window_f32(
        &runtime->comp_numeric,
        runtime->comp_buffer_kv,
        runtime->comp_buffer_gate,
        weights->compress_position_bias,
        weights->compress_norm_weight,
        compressed));
    if (status == DSV4_AR_OK) {
        status = append_vector(&runtime->compressed_kv,
                               &runtime->compressed_count,
                               &runtime->compressed_capacity,
                               runtime->config.head_dim,
                               compressed);
    }
    if (status == DSV4_AR_OK) runtime->comp_buffer_count = 0u;
    free(compressed);
    return status;
}

static DSV4AttentionRuntimeStatus push_index_compressor(
    DSV4AttentionRuntime *runtime,
    const DSV4AttentionRuntimeWeights *weights,
    const float *hidden) {
    DSV4AttentionRuntimeStatus status;
    float *kv_row;
    float *gate_row;
    float *compressed = NULL;
    size_t row_offset;

    if (runtime->config.compression_rate != 4u) return DSV4_AR_OK;
    row_offset = runtime->index_buffer_count * runtime->index_projection_dim;
    kv_row = runtime->index_buffer_kv + row_offset;
    gate_row = runtime->index_buffer_gate + row_offset;
    status = dense_linear(hidden, weights->index_compress_kv_weight,
                          runtime->config.hidden_size,
                          runtime->index_projection_dim, kv_row);
    if (status != DSV4_AR_OK) return status;
    status = dense_linear(hidden, weights->index_compress_gate_weight,
                          runtime->config.hidden_size,
                          runtime->index_projection_dim, gate_row);
    if (status != DSV4_AR_OK) return status;
    runtime->index_buffer_count += 1u;
    if (runtime->index_buffer_count < 4u) return DSV4_AR_OK;
    if (runtime->index_buffer_count != 4u) return DSV4_AR_DEPENDENCY_FAILED;

    status = alloc_floats(&compressed, runtime->config.index_head_dim, 0);
    if (status != DSV4_AR_OK) return status;
    status = map_numeric(dsv4_compress_projected_window_f32(
        &runtime->index_numeric,
        runtime->index_buffer_kv,
        runtime->index_buffer_gate,
        weights->index_compress_position_bias,
        weights->index_compress_norm_weight,
        compressed));
    if (status == DSV4_AR_OK) {
        status = append_vector(&runtime->index_compressed,
                               &runtime->index_compressed_count,
                               &runtime->index_compressed_capacity,
                               runtime->config.index_head_dim,
                               compressed);
    }
    if (status == DSV4_AR_OK) runtime->index_buffer_count = 0u;
    free(compressed);
    return status;
}

static DSV4AttentionRuntimeStatus validate_weights(
    const DSV4AttentionRuntimeConfig *config,
    const DSV4AttentionRuntimeWeights *weights) {
    size_t total_width;
    size_t in_per_group;
    size_t grouped;
    size_t count;
    size_t projection_dim;

    if (config == NULL || weights == NULL) return DSV4_AR_INVALID_ARGUMENT;
    if (!checked_mul_size(config->num_heads, config->head_dim, &total_width) ||
        !checked_mul_size(config->q_lora_rank, config->hidden_size, &count))
        return DSV4_AR_OVERFLOW;
    if (!all_finite(weights->q_a_weight, count) ||
        !all_finite(weights->q_a_norm_weight, config->q_lora_rank))
        return weights->q_a_weight == NULL || weights->q_a_norm_weight == NULL
                   ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
    if (!checked_mul_size(total_width, config->q_lora_rank, &count))
        return DSV4_AR_OVERFLOW;
    if (!all_finite(weights->q_b_weight, count))
        return weights->q_b_weight == NULL ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
    if (!checked_mul_size(config->head_dim, config->hidden_size, &count))
        return DSV4_AR_OVERFLOW;
    if (!all_finite(weights->kv_weight, count) ||
        !all_finite(weights->kv_norm_weight, config->head_dim) ||
        !all_finite(weights->sinks, config->num_heads))
        return weights->kv_weight == NULL || weights->kv_norm_weight == NULL || weights->sinks == NULL
                   ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;

    in_per_group = total_width / config->output_groups;
    if (!checked_mul_size(config->output_groups, config->output_rank, &grouped) ||
        !checked_mul_size(grouped, in_per_group, &count))
        return DSV4_AR_OVERFLOW;
    if (!all_finite(weights->o_a_weight, count))
        return weights->o_a_weight == NULL ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
    if (!checked_mul_size(config->hidden_size, grouped, &count))
        return DSV4_AR_OVERFLOW;
    if (!all_finite(weights->o_b_weight, count))
        return weights->o_b_weight == NULL ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;

    if (config->compression_rate != 0u) {
        if (config->head_dim > SIZE_MAX / (config->compression_rate == 4u ? 2u : 1u))
            return DSV4_AR_OVERFLOW;
        projection_dim = config->head_dim * (config->compression_rate == 4u ? 2u : 1u);
        if (!checked_mul_size(projection_dim, config->hidden_size, &count))
            return DSV4_AR_OVERFLOW;
        if (!all_finite(weights->compress_kv_weight, count) ||
            !all_finite(weights->compress_gate_weight, count))
            return weights->compress_kv_weight == NULL || weights->compress_gate_weight == NULL
                       ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
        if (!checked_mul_size(config->compression_rate, projection_dim, &count))
            return DSV4_AR_OVERFLOW;
        if (!all_finite(weights->compress_position_bias, count) ||
            !all_finite(weights->compress_norm_weight, config->head_dim))
            return weights->compress_position_bias == NULL || weights->compress_norm_weight == NULL
                       ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
    }

    if (config->compression_rate == 4u) {
        if (!checked_mul_size(config->index_num_heads, config->index_head_dim, &count) ||
            !checked_mul_size(count, config->q_lora_rank, &count))
            return DSV4_AR_OVERFLOW;
        if (!all_finite(weights->index_q_weight, count))
            return weights->index_q_weight == NULL ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
        if (!checked_mul_size(config->index_num_heads, config->hidden_size, &count))
            return DSV4_AR_OVERFLOW;
        if (!all_finite(weights->index_head_weight, count))
            return weights->index_head_weight == NULL ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
        if (config->index_head_dim > SIZE_MAX / 2u) return DSV4_AR_OVERFLOW;
        projection_dim = 2u * config->index_head_dim;
        if (!checked_mul_size(projection_dim, config->hidden_size, &count))
            return DSV4_AR_OVERFLOW;
        if (!all_finite(weights->index_compress_kv_weight, count) ||
            !all_finite(weights->index_compress_gate_weight, count))
            return weights->index_compress_kv_weight == NULL || weights->index_compress_gate_weight == NULL
                       ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
        if (!checked_mul_size(4u, projection_dim, &count)) return DSV4_AR_OVERFLOW;
        if (!all_finite(weights->index_compress_position_bias, count) ||
            !all_finite(weights->index_compress_norm_weight, config->index_head_dim))
            return weights->index_compress_position_bias == NULL ||
                           weights->index_compress_norm_weight == NULL
                       ? DSV4_AR_INVALID_ARGUMENT : DSV4_AR_NONFINITE;
    }
    return DSV4_AR_OK;
}

DSV4AttentionRuntimeStatus dsv4_attention_runtime_step_f32(
    DSV4AttentionRuntime *runtime,
    const DSV4AttentionRuntimeWeights *weights,
    int64_t position,
    const float *hidden,
    float *out_hidden,
    size_t *out_selected_indices,
    size_t out_selected_capacity,
    size_t *out_selected_count,
    DSV4AttentionRuntimeTrace *out_trace) {
    DSV4AttentionRuntimeStatus status;
    DSV4AttentionRuntime *work = NULL;
    DSV4AttentionStepConfig step_config;
    DSV4AttentionRuntimeTrace trace;
    float *q_residual = NULL;
    float *q = NULL;
    float *kv = NULL;
    float *index_q = NULL;
    float *index_head_weights = NULL;
    float *hidden_result = NULL;
    size_t *selected_result = NULL;
    size_t selected_count = 0u;
    size_t total_width;
    size_t index_width = 0u;
    float theta;

    if (runtime == NULL || weights == NULL || hidden == NULL || out_hidden == NULL ||
        out_selected_count == NULL || out_trace == NULL || position < 0)
        return DSV4_AR_INVALID_ARGUMENT;
    if (runtime->next_position > (uint64_t)INT64_MAX ||
        (uint64_t)position != runtime->next_position)
        return DSV4_AR_INVALID_ARGUMENT;
    if (position == INT64_MAX) return DSV4_AR_OVERFLOW;
    if (runtime->config.compression_rate == 4u &&
        (out_selected_indices == NULL || out_selected_capacity < runtime->config.index_top_k))
        return DSV4_AR_INVALID_ARGUMENT;
    if (!all_finite(hidden, runtime->config.hidden_size)) return DSV4_AR_NONFINITE;

    status = validate_weights(&runtime->config, weights);
    if (status != DSV4_AR_OK) return status;
    work = clone_runtime(runtime, &status);
    if (work == NULL) return status;

    if (!checked_mul_size(work->config.num_heads, work->config.head_dim, &total_width)) {
        status = DSV4_AR_OVERFLOW;
        goto cleanup;
    }
    status = alloc_floats(&q_residual, work->config.q_lora_rank, 0);
    if (status != DSV4_AR_OK) goto cleanup;
    status = alloc_floats(&q, total_width, 0);
    if (status != DSV4_AR_OK) goto cleanup;
    status = alloc_floats(&kv, work->config.head_dim, 0);
    if (status != DSV4_AR_OK) goto cleanup;
    status = alloc_floats(&hidden_result, work->config.hidden_size, 0);
    if (status != DSV4_AR_OK) goto cleanup;
    if (work->config.compression_rate == 4u) {
        if (!checked_mul_size(work->config.index_num_heads,
                              work->config.index_head_dim, &index_width)) {
            status = DSV4_AR_OVERFLOW;
            goto cleanup;
        }
        status = alloc_floats(&index_q, index_width, 0);
        if (status != DSV4_AR_OK) goto cleanup;
        status = alloc_floats(&index_head_weights, work->config.index_num_heads, 0);
        if (status != DSV4_AR_OK) goto cleanup;
        status = alloc_indices(&selected_result, work->config.index_top_k);
        if (status != DSV4_AR_OK) goto cleanup;
    }

    status = dense_linear(hidden, weights->q_a_weight,
                          work->config.hidden_size,
                          work->config.q_lora_rank, q_residual);
    if (status != DSV4_AR_OK) goto cleanup;
    status = weighted_rms_inplace(q_residual, weights->q_a_norm_weight,
                                  work->config.q_lora_rank,
                                  work->config.rms_eps);
    if (status != DSV4_AR_OK) goto cleanup;
    status = dense_linear(q_residual, weights->q_b_weight,
                          work->config.q_lora_rank,
                          total_width, q);
    if (status != DSV4_AR_OK) goto cleanup;
    status = unweighted_head_rms_inplace(q, work->config.num_heads,
                                         work->config.head_dim,
                                         work->config.rms_eps);
    if (status != DSV4_AR_OK) goto cleanup;

    status = dense_linear(hidden, weights->kv_weight,
                          work->config.hidden_size,
                          work->config.head_dim, kv);
    if (status != DSV4_AR_OK) goto cleanup;
    status = weighted_rms_inplace(kv, weights->kv_norm_weight,
                                  work->config.head_dim,
                                  work->config.rms_eps);
    if (status != DSV4_AR_OK) goto cleanup;
    theta = work->config.compression_rate == 0u
                ? work->config.rope_theta
                : work->config.compress_rope_theta;
    status = map_numeric(dsv4_partial_rope_f32(
        kv, 1u, work->config.head_dim, work->config.rope_dim,
        position, theta));
    if (status != DSV4_AR_OK) goto cleanup;
    status = append_raw_kv(work, kv);
    if (status != DSV4_AR_OK) goto cleanup;

    status = push_attention_compressor(work, weights, hidden);
    if (status != DSV4_AR_OK) goto cleanup;
    status = push_index_compressor(work, weights, hidden);
    if (status != DSV4_AR_OK) goto cleanup;

    if (work->config.compression_rate == 4u) {
        if (work->compressed_count != work->index_compressed_count) {
            status = DSV4_AR_DEPENDENCY_FAILED;
            goto cleanup;
        }
        if (work->index_compressed_count != 0u) {
            status = dense_linear(q_residual, weights->index_q_weight,
                                  work->config.q_lora_rank,
                                  index_width, index_q);
            if (status != DSV4_AR_OK) goto cleanup;
            status = dense_linear(hidden, weights->index_head_weight,
                                  work->config.hidden_size,
                                  work->config.index_num_heads,
                                  index_head_weights);
            if (status != DSV4_AR_OK) goto cleanup;
        }
    }

    memset(&step_config, 0, sizeof(step_config));
    step_config.num_heads = work->config.num_heads;
    step_config.head_dim = work->config.head_dim;
    step_config.rope_dim = work->config.rope_dim;
    step_config.raw_tokens = work->raw_count;
    step_config.compressed_tokens = work->compressed_count;
    step_config.index_num_heads = work->config.index_num_heads;
    step_config.index_head_dim = work->config.index_head_dim;
    step_config.index_rope_dim = work->config.index_rope_dim;
    step_config.index_top_k = work->config.index_top_k;
    step_config.output_groups = work->config.output_groups;
    step_config.output_rank = work->config.output_rank;
    step_config.hidden_size = work->config.hidden_size;
    step_config.rope_theta = theta;
    step_config.use_sparse_index = work->config.compression_rate == 4u ? 1 : 0;

    status = map_step(dsv4_attention_postprojected_step_f32(
        &step_config,
        position,
        q,
        work->raw_kv,
        work->compressed_kv,
        work->index_compressed_count != 0u ? index_q : NULL,
        work->index_compressed,
        work->index_compressed_count != 0u ? index_head_weights : NULL,
        weights->sinks,
        weights->o_a_weight,
        weights->o_b_weight,
        hidden_result,
        selected_result,
        &selected_count));
    if (status != DSV4_AR_OK) goto cleanup;

    work->next_position += 1u;
    trace.raw_cache_length = work->raw_count;
    trace.compressed_count = work->compressed_count;
    trace.indexer_compressed_count = work->index_compressed_count;

    memcpy(out_hidden, hidden_result, work->config.hidden_size * sizeof(float));
    if (work->config.compression_rate == 4u && selected_count != 0u)
        memcpy(out_selected_indices, selected_result, selected_count * sizeof(size_t));
    *out_selected_count = work->config.compression_rate == 4u ? selected_count : 0u;
    *out_trace = trace;
    commit_runtime(runtime, work);
    work = NULL;
    status = DSV4_AR_OK;

cleanup:
    free(q_residual);
    free(q);
    free(kv);
    free(index_q);
    free(index_head_weights);
    free(hidden_result);
    free(selected_result);
    dsv4_attention_runtime_destroy(work);
    return status;
}

const char *dsv4_attention_runtime_status_string(
    DSV4AttentionRuntimeStatus status) {
    switch (status) {
        case DSV4_AR_OK: return "ok";
        case DSV4_AR_INVALID_ARGUMENT: return "invalid argument";
        case DSV4_AR_NONFINITE: return "non-finite input, weight or result";
        case DSV4_AR_OVERFLOW: return "size or position overflow";
        case DSV4_AR_ALLOCATION_FAILED: return "allocation failed";
        case DSV4_AR_DEPENDENCY_FAILED: return "dependency failed";
        default: return "unknown status";
    }
}

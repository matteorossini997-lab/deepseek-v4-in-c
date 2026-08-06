#include "dsv4_attention_step.h"

#include "dsv4_attention_numeric.h"
#include "dsv4_attention_state.h"
#include "dsv4_sparse_index.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static DSV4AttentionStepStatus map_numeric(DSV4AttentionNumericStatus status) {
    switch (status) {
        case DSV4_AN_OK: return DSV4_AS_OK;
        case DSV4_AN_INVALID_ARGUMENT: return DSV4_AS_INVALID_ARGUMENT;
        case DSV4_AN_NONFINITE: return DSV4_AS_NONFINITE;
        case DSV4_AN_OVERFLOW: return DSV4_AS_OVERFLOW;
        case DSV4_AN_ALLOCATION_FAILED: return DSV4_AS_ALLOCATION_FAILED;
        default: return DSV4_AS_DEPENDENCY_FAILED;
    }
}

static DSV4AttentionStepStatus map_sparse(DSV4SparseIndexStatus status) {
    switch (status) {
        case DSV4_SI_OK: return DSV4_AS_OK;
        case DSV4_SI_INVALID_ARGUMENT: return DSV4_AS_INVALID_ARGUMENT;
        case DSV4_SI_NONFINITE: return DSV4_AS_NONFINITE;
        case DSV4_SI_OVERFLOW: return DSV4_AS_OVERFLOW;
        case DSV4_SI_ALLOCATION_FAILED: return DSV4_AS_ALLOCATION_FAILED;
        default: return DSV4_AS_DEPENDENCY_FAILED;
    }
}

static void free_all(float *q_rot,
                     float *index_q_rot,
                     float *index_scores,
                     size_t *selected,
                     float *kv_all,
                     float *attention,
                     float *grouped,
                     float *hidden) {
    free(q_rot);
    free(index_q_rot);
    free(index_scores);
    free(selected);
    free(kv_all);
    free(attention);
    free(grouped);
    free(hidden);
}

DSV4AttentionStepStatus dsv4_attention_postprojected_step_f32(
    const DSV4AttentionStepConfig *config,
    int64_t position,
    const float *q,
    const float *raw_kv,
    const float *compressed_kv,
    const float *index_q,
    const float *index_compressed,
    const float *index_head_weights,
    const float *sinks,
    const float *o_a_weight,
    const float *o_b_weight,
    float *out_hidden,
    size_t *out_selected_indices,
    size_t *out_selected_count) {
    size_t q_count, q_bytes, raw_count, compressed_count, total_width;
    size_t in_per_group, oa_count, ob_in, ob_count, grouped_count;
    size_t selected_capacity = 0u, selected_count = 0u, kv_tokens, kv_count, kv_bytes;
    size_t index_q_count = 0u, index_q_bytes = 0u, index_compressed_count = 0u;
    size_t selected_bytes = 0u, scores_bytes = 0u, hidden_bytes, grouped_bytes;
    size_t offset, i, row;
    float scale;
    float *q_rot = NULL, *index_q_rot = NULL, *index_scores = NULL;
    size_t *selected = NULL;
    float *kv_all = NULL, *attention = NULL, *grouped = NULL, *hidden = NULL;
    DSV4AttentionStepStatus status = DSV4_AS_OK;

    if (config == NULL || q == NULL || sinks == NULL || o_a_weight == NULL ||
        o_b_weight == NULL || out_hidden == NULL || out_selected_count == NULL ||
        config->num_heads == 0u || config->head_dim == 0u ||
        config->output_groups == 0u || config->output_rank == 0u ||
        config->hidden_size == 0u ||
        (config->use_sparse_index != 0 && config->use_sparse_index != 1) ||
        !isfinite(config->rope_theta) || config->rope_theta <= 0.0f ||
        config->raw_tokens == 0u || position < 0) {
        return DSV4_AS_INVALID_ARGUMENT;
    }
    if (config->rope_dim > config->head_dim || (config->rope_dim % 2u) != 0u)
        return DSV4_AS_INVALID_ARGUMENT;
    if (!checked_mul_size(config->num_heads, config->head_dim, &q_count) ||
        !checked_mul_size(q_count, sizeof(float), &q_bytes) ||
        !checked_mul_size(config->raw_tokens, config->head_dim, &raw_count) ||
        !checked_mul_size(config->compressed_tokens, config->head_dim, &compressed_count) ||
        !checked_mul_size(config->num_heads, config->head_dim, &total_width) ||
        total_width % config->output_groups != 0u) {
        return DSV4_AS_OVERFLOW;
    }
    in_per_group = total_width / config->output_groups;
    if (!checked_mul_size(config->output_groups, config->output_rank, &grouped_count) ||
        !checked_mul_size(grouped_count, in_per_group, &oa_count) ||
        !checked_mul_size(config->output_groups, config->output_rank, &ob_in) ||
        !checked_mul_size(config->hidden_size, ob_in, &ob_count) ||
        !checked_mul_size(config->hidden_size, sizeof(float), &hidden_bytes) ||
        !checked_mul_size(grouped_count, sizeof(float), &grouped_bytes)) {
        return DSV4_AS_OVERFLOW;
    }
    if (!all_finite(q, q_count) || !all_finite(raw_kv, raw_count) ||
        (config->compressed_tokens != 0u && !all_finite(compressed_kv, compressed_count)) ||
        !all_finite(sinks, config->num_heads) || !all_finite(o_a_weight, oa_count) ||
        !all_finite(o_b_weight, ob_count)) {
        return DSV4_AS_NONFINITE;
    }

    if (config->use_sparse_index != 0 && config->compressed_tokens != 0u) {
        if (index_q == NULL || index_compressed == NULL || index_head_weights == NULL ||
            out_selected_indices == NULL || config->index_num_heads == 0u ||
            config->index_head_dim == 0u || config->index_top_k == 0u ||
            config->index_rope_dim > config->index_head_dim ||
            (config->index_rope_dim % 2u) != 0u) {
            return DSV4_AS_INVALID_ARGUMENT;
        }
        if (!checked_mul_size(config->index_num_heads, config->index_head_dim, &index_q_count) ||
            !checked_mul_size(index_q_count, sizeof(float), &index_q_bytes) ||
            !checked_mul_size(config->compressed_tokens, config->index_head_dim,
                              &index_compressed_count)) {
            return DSV4_AS_OVERFLOW;
        }
        if (!all_finite(index_q, index_q_count) ||
            !all_finite(index_compressed, index_compressed_count) ||
            !all_finite(index_head_weights, config->index_num_heads)) {
            return DSV4_AS_NONFINITE;
        }
        selected_capacity = config->index_top_k < config->compressed_tokens
                                ? config->index_top_k : config->compressed_tokens;
        if (!checked_mul_size(selected_capacity, sizeof(size_t), &selected_bytes) ||
            !checked_mul_size(selected_capacity, sizeof(float), &scores_bytes))
            return DSV4_AS_OVERFLOW;
    } else if (config->compressed_tokens != 0u) {
        selected_count = config->compressed_tokens;
    }

    q_rot = (float *)malloc(q_bytes);
    attention = (float *)malloc(q_bytes);
    grouped = (float *)malloc(grouped_bytes);
    hidden = (float *)malloc(hidden_bytes);
    if (q_rot == NULL || attention == NULL || grouped == NULL || hidden == NULL) {
        status = DSV4_AS_ALLOCATION_FAILED;
        goto cleanup;
    }
    memcpy(q_rot, q, q_bytes);
    status = map_numeric(dsv4_partial_rope_f32(
        q_rot, config->num_heads, config->head_dim, config->rope_dim,
        position, config->rope_theta));
    if (status != DSV4_AS_OK) goto cleanup;

    if (selected_capacity != 0u) {
        index_q_rot = (float *)malloc(index_q_bytes);
        index_scores = (float *)malloc(scores_bytes);
        selected = (size_t *)malloc(selected_bytes);
        if (index_q_rot == NULL || index_scores == NULL || selected == NULL) {
            status = DSV4_AS_ALLOCATION_FAILED;
            goto cleanup;
        }
        memcpy(index_q_rot, index_q, index_q_bytes);
        status = map_numeric(dsv4_partial_rope_f32(
            index_q_rot, config->index_num_heads, config->index_head_dim,
            config->index_rope_dim, position, config->rope_theta));
        if (status != DSV4_AS_OK) goto cleanup;
        status = map_sparse(dsv4_sparse_index_topk_f32(
            index_q_rot, index_compressed, index_head_weights,
            config->index_num_heads, config->compressed_tokens,
            config->index_head_dim, config->index_top_k,
            index_scores, selected, &selected_count));
        if (status != DSV4_AS_OK) goto cleanup;
    }

    if (!checked_add_size(config->raw_tokens, selected_count, &kv_tokens) ||
        !checked_mul_size(kv_tokens, config->head_dim, &kv_count) ||
        !checked_mul_size(kv_count, sizeof(float), &kv_bytes)) {
        status = DSV4_AS_OVERFLOW;
        goto cleanup;
    }
    kv_all = (float *)malloc(kv_bytes);
    if (kv_all == NULL) {
        status = DSV4_AS_ALLOCATION_FAILED;
        goto cleanup;
    }
    memcpy(kv_all, raw_kv, raw_count * sizeof(float));
    offset = raw_count;
    if (config->compressed_tokens != 0u) {
        if (config->use_sparse_index != 0) {
            for (i = 0u; i < selected_count; ++i) {
                memcpy(kv_all + offset, compressed_kv + selected[i] * config->head_dim,
                       config->head_dim * sizeof(float));
                offset += config->head_dim;
            }
        } else {
            memcpy(kv_all + offset, compressed_kv, compressed_count * sizeof(float));
        }
    }

    scale = 1.0f / sqrtf((float)config->head_dim);
    if (!isfinite(scale)) {
        status = DSV4_AS_NONFINITE;
        goto cleanup;
    }
    status = map_numeric(dsv4_sink_attention_shared_kv_f32(
        q_rot, kv_all, sinks, config->num_heads, kv_tokens,
        config->head_dim, scale, attention));
    if (status != DSV4_AS_OK) goto cleanup;
    status = map_numeric(dsv4_partial_rope_f32(
        attention, config->num_heads, config->head_dim, config->rope_dim,
        -position, config->rope_theta));
    if (status != DSV4_AS_OK) goto cleanup;

    if (dsv4_cpu_grouped_linear(
            attention, o_a_weight, 1u, config->output_groups,
            in_per_group, config->output_rank, grouped) != 0) {
        status = DSV4_AS_DEPENDENCY_FAILED;
        goto cleanup;
    }
    for (row = 0u; row < config->hidden_size; ++row) {
        float sum = 0.0f;
        const float *weight_row = o_b_weight + row * ob_in;
        for (i = 0u; i < ob_in; ++i) sum += weight_row[i] * grouped[i];
        if (!isfinite(sum)) {
            status = DSV4_AS_NONFINITE;
            goto cleanup;
        }
        hidden[row] = sum;
    }

    memcpy(out_hidden, hidden, hidden_bytes);
    if (selected_capacity != 0u)
        memcpy(out_selected_indices, selected, selected_count * sizeof(size_t));
    *out_selected_count = selected_capacity != 0u ? selected_count : 0u;

cleanup:
    free_all(q_rot, index_q_rot, index_scores, selected, kv_all,
             attention, grouped, hidden);
    return status;
}

const char *dsv4_attention_step_status_string(DSV4AttentionStepStatus status) {
    switch (status) {
        case DSV4_AS_OK: return "ok";
        case DSV4_AS_INVALID_ARGUMENT: return "invalid argument";
        case DSV4_AS_NONFINITE: return "non-finite input or result";
        case DSV4_AS_OVERFLOW: return "size overflow";
        case DSV4_AS_ALLOCATION_FAILED: return "allocation failed";
        case DSV4_AS_DEPENDENCY_FAILED: return "dependency failed";
        default: return "unknown status";
    }
}

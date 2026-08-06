#include "dsv4_sparse_index.h"

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
    size_t index;

    if (values == NULL) {
        return 0;
    }
    for (index = 0u; index < count; ++index) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int dsv4_candidate_is_better(
    float candidate_score,
    size_t candidate_index,
    float current_score,
    size_t current_index) {
    if (candidate_score > current_score) {
        return 1;
    }
    if (candidate_score < current_score) {
        return 0;
    }
    return candidate_index < current_index;
}

DSV4SparseIndexStatus dsv4_sparse_index_topk_f32(
    const float *q,
    const float *compressed,
    const float *head_weights,
    size_t heads,
    size_t tokens,
    size_t head_dim,
    size_t top_k,
    float *out_scores,
    size_t *out_indices,
    size_t *out_count) {
    size_t q_count;
    size_t compressed_count;
    size_t selected_count;
    size_t scores_bytes;
    size_t selected_scores_bytes;
    size_t selected_indices_bytes;
    float *scores = NULL;
    float *selected_scores = NULL;
    size_t *selected_indices = NULL;
    size_t token;
    size_t head;
    size_t dimension;
    float head_scale;
    float dimension_scale;

    if (q == NULL || compressed == NULL || head_weights == NULL ||
        out_scores == NULL || out_indices == NULL || out_count == NULL ||
        heads == 0u || tokens == 0u || head_dim == 0u || top_k == 0u) {
        return DSV4_SI_INVALID_ARGUMENT;
    }
    if (!dsv4_checked_mul_size(heads, head_dim, &q_count) ||
        !dsv4_checked_mul_size(tokens, head_dim, &compressed_count)) {
        return DSV4_SI_OVERFLOW;
    }

    selected_count = top_k < tokens ? top_k : tokens;
    if (!dsv4_checked_mul_size(tokens, sizeof(float), &scores_bytes) ||
        !dsv4_checked_mul_size(
            selected_count, sizeof(float), &selected_scores_bytes) ||
        !dsv4_checked_mul_size(
            selected_count, sizeof(size_t), &selected_indices_bytes)) {
        return DSV4_SI_OVERFLOW;
    }
    if (!dsv4_all_finite(q, q_count) ||
        !dsv4_all_finite(compressed, compressed_count) ||
        !dsv4_all_finite(head_weights, heads)) {
        return DSV4_SI_NONFINITE;
    }

    scores = (float *)malloc(scores_bytes);
    selected_scores = (float *)malloc(selected_scores_bytes);
    selected_indices = (size_t *)malloc(selected_indices_bytes);
    if (scores == NULL || selected_scores == NULL || selected_indices == NULL) {
        free(scores);
        free(selected_scores);
        free(selected_indices);
        return DSV4_SI_ALLOCATION_FAILED;
    }

    dimension_scale = 1.0f / sqrtf((float)head_dim);
    head_scale = 1.0f / sqrtf((float)heads);
    if (!isfinite(dimension_scale) || !isfinite(head_scale)) {
        free(scores);
        free(selected_scores);
        free(selected_indices);
        return DSV4_SI_NONFINITE;
    }

    for (token = 0u; token < tokens; ++token) {
        const float *token_values = compressed + token * head_dim;
        float total = 0.0f;

        for (head = 0u; head < heads; ++head) {
            const float *head_query = q + head * head_dim;
            float dot = 0.0f;

            for (dimension = 0u; dimension < head_dim; ++dimension) {
                dot += head_query[dimension] * token_values[dimension];
            }
            if (dot < 0.0f) {
                dot = 0.0f;
            }
            total += dot * dimension_scale * head_weights[head] * head_scale;
        }
        if (!isfinite(total)) {
            free(scores);
            free(selected_scores);
            free(selected_indices);
            return DSV4_SI_NONFINITE;
        }
        scores[token] = total;
    }

    for (token = 0u; token < selected_count; ++token) {
        size_t candidate;
        size_t best_index = SIZE_MAX;
        float best_score = 0.0f;

        for (candidate = 0u; candidate < tokens; ++candidate) {
            size_t prior;
            int already_selected = 0;

            for (prior = 0u; prior < token; ++prior) {
                if (selected_indices[prior] == candidate) {
                    already_selected = 1;
                    break;
                }
            }
            if (already_selected != 0) {
                continue;
            }
            if (best_index == SIZE_MAX ||
                dsv4_candidate_is_better(
                    scores[candidate], candidate, best_score, best_index)) {
                best_score = scores[candidate];
                best_index = candidate;
            }
        }
        selected_scores[token] = best_score;
        selected_indices[token] = best_index;
    }

    memcpy(out_scores, selected_scores, selected_scores_bytes);
    memcpy(out_indices, selected_indices, selected_indices_bytes);
    *out_count = selected_count;

    free(scores);
    free(selected_scores);
    free(selected_indices);
    return DSV4_SI_OK;
}

const char *dsv4_sparse_index_status_string(DSV4SparseIndexStatus status) {
    switch (status) {
        case DSV4_SI_OK:
            return "ok";
        case DSV4_SI_INVALID_ARGUMENT:
            return "invalid argument";
        case DSV4_SI_NONFINITE:
            return "non-finite input or result";
        case DSV4_SI_OVERFLOW:
            return "size overflow";
        case DSV4_SI_ALLOCATION_FAILED:
            return "allocation failed";
        default:
            return "unknown status";
    }
}

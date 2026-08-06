#ifndef DSV4_SPARSE_INDEX_H
#define DSV4_SPARSE_INDEX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4SparseIndexStatus {
    DSV4_SI_OK = 0,
    DSV4_SI_INVALID_ARGUMENT = 1,
    DSV4_SI_NONFINITE = 2,
    DSV4_SI_OVERFLOW = 3,
    DSV4_SI_ALLOCATION_FAILED = 4
} DSV4SparseIndexStatus;

/*
 * Implements the post-projection MiniIndexer score:
 *
 *   score[t] = sum_h relu(dot(q[h], compressed[t]))
 *                    * head_dim^-0.5 * head_weight[h] * heads^-0.5
 *
 * and returns min(top_k, tokens) entries sorted by score descending. Exact
 * ties are resolved by lower token index. Outputs are published atomically.
 */
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
    size_t *out_count);

const char *dsv4_sparse_index_status_string(DSV4SparseIndexStatus status);

#ifdef __cplusplus
}
#endif

#endif

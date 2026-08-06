#ifndef DSV4_CPU_OPS_H
#define DSV4_CPU_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DSV4_CPU_OK = 0,
    DSV4_CPU_INVALID = -1,
    DSV4_CPU_NOMEM = -2,
    DSV4_CPU_NUMERIC = -3
} DSV4CpuStatus;

/* Stable FP32 implementation of sqrt(softplus(x)). */
float dsv4_cpu_sqrtsoftplus(float x);

/* Canonical mini-oracle hash route:
 *   start = token_id * 5 + layer_idx * 3
 *   expert(slot) = (start + slot * 7) mod num_experts
 * Returns a unique route or DSV4_CPU_INVALID when the requested top-k exceeds
 * the permutation cycle induced by step 7.
 */
int dsv4_cpu_hash_route(uint64_t token_id, uint32_t layer_idx,
                        size_t num_experts, size_t top_k,
                        uint32_t *indices);

/* Learned routing for one hidden row. Selection uses
 * sqrtsoftplus(logit) + correction_bias, while route weights are gathered from
 * the unbiased sqrtsoftplus scores, normalized, then multiplied by route_scale.
 *
 * For portability the returned route is canonicalized by descending corrected
 * score, with lower expert id breaking exact ties. This preserves the selected
 * set and expert-weight pairs while avoiding backend-dependent sorted=False
 * ordering.
 */
int dsv4_cpu_learned_route(const float *hidden,
                           const float *expert_weight,
                           const float *correction_bias,
                           size_t hidden_size,
                           size_t num_experts,
                           size_t top_k,
                           float route_scale,
                           float *logits,
                           uint32_t *indices,
                           float *route_weights);

/* Hyper-Connection for one [hc_mult, hidden_size] stream group.
 * fn is [(2 + hc_mult) * hc_mult, hc_mult * hidden_size], row-major.
 * base has (2 + hc_mult) * hc_mult entries and scale has 3 entries.
 * Outputs match the mini-oracle order: post, Sinkhorn combination matrix,
 * collapsed stream. comb is row-major [hc_mult, hc_mult].
 */
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
                             float *collapsed);

#ifdef __cplusplus
}
#endif

#endif

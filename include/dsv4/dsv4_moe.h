#ifndef DSV4_MOE_H
#define DSV4_MOE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DSV4MoEStatus {
    DSV4_MOE_OK = 0,
    DSV4_MOE_INVALID_ARGUMENT = 1,
    DSV4_MOE_NONFINITE = 2,
    DSV4_MOE_OVERFLOW = 3,
    DSV4_MOE_ALLOCATION_FAILED = 4,
    DSV4_MOE_DEPENDENCY_FAILED = 5
} DSV4MoEStatus;

typedef struct DSV4MoEConfig {
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_routed_experts;
    size_t top_k;
    size_t num_hash_layers;
    size_t vocab_size;
    float route_scale;
    float swiglu_limit;
} DSV4MoEConfig;

typedef struct DSV4MoEWeights {
    /* [num_routed_experts, hidden_size] */
    const float *router_weight;

    /* [num_routed_experts], required for layer_idx >= num_hash_layers. */
    const float *correction_bias;

    /* [vocab_size, top_k], required for layer_idx < num_hash_layers. */
    const uint32_t *hash_table;

    /* Routed experts: gate/up [experts, intermediate, hidden],
     * down [experts, hidden, intermediate]. */
    const float *expert_gate_weight;
    const float *expert_up_weight;
    const float *expert_down_weight;

    /* One unweighted shared expert: gate/up [intermediate, hidden],
     * down [hidden, intermediate]. */
    const float *shared_gate_weight;
    const float *shared_up_weight;
    const float *shared_down_weight;
} DSV4MoEWeights;

/*
 * FP32 reference execution for one token.
 *
 * Hash layers consume the explicit caller-owned tid2eid table; learned layers
 * use correction-bias selection through the P1-A canonical learned router.
 * Route diagnostics are returned in the router's native/canonical slot order.
 *
 * Publication is transactional: on failure out_hidden, route indices and route
 * weights are unchanged.
 */
DSV4MoEStatus dsv4_moe_forward_f32(
    const DSV4MoEConfig *config,
    const DSV4MoEWeights *weights,
    uint32_t layer_idx,
    uint64_t token_id,
    const float *hidden,
    float *out_hidden,
    uint32_t *out_route_indices,
    float *out_route_weights,
    size_t route_capacity);

const char *dsv4_moe_status_string(DSV4MoEStatus status);

#ifdef __cplusplus
}
#endif

#endif

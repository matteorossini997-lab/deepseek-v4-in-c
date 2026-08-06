#include "dsv4_cpu_ops.h"
#include "dsv4_cpu_ops_vectors.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check_int(const char *name, unsigned got, unsigned want)
{
    if (got != want) {
        fprintf(stderr, "FAIL %-30s got %u want %u\n", name, got, want);
        failures++;
    }
}

static void check_close(const char *name, float got, float want, float tol)
{
    float limit = tol + tol * fabsf(want);
    if (!isfinite(got) || fabsf(got - want) > limit) {
        fprintf(stderr, "FAIL %-30s got %.9g want %.9g delta %.3g limit %.3g\n",
                name, got, want, fabsf(got - want), limit);
        failures++;
    }
}

static void test_sqrtsoftplus(void)
{
    for (size_t i = 0; i < DSV4_VEC_SQRT_COUNT; ++i) {
        char name[64];
        snprintf(name, sizeof name, "sqrtsoftplus[%zu]", i);
        check_close(name, dsv4_cpu_sqrtsoftplus(dsv4_vec_sqrt_input[i]),
                    dsv4_vec_sqrt_expected[i], 2e-6f);
    }
    check_int("sqrtsoftplus -inf", dsv4_cpu_sqrtsoftplus(-INFINITY) == 0.0f, 1);
    check_int("sqrtsoftplus nan", isnan(dsv4_cpu_sqrtsoftplus(NAN)), 1);
}

static void test_hash_route(void)
{
    static const uint64_t token_ids[4] = {0u, 1u, 17u, 255u};
    static const unsigned expected[2][4][6] = {
        {{0,7,14,5,12,3}, {5,12,3,10,1,8}, {5,12,3,10,1,8}, {11,2,9,0,7,14}},
        {{3,10,1,8,15,6}, {8,15,6,13,4,11}, {8,15,6,13,4,11}, {14,5,12,3,10,1}}
    };
    for (unsigned layer = 0; layer < 2; ++layer) {
        for (size_t row = 0; row < 4; ++row) {
            uint32_t route[6] = {0};
            check_int("hash status",
                      dsv4_cpu_hash_route(token_ids[row], layer, 16, 6, route),
                      DSV4_CPU_OK);
            for (size_t slot = 0; slot < 6; ++slot)
                check_int("hash route", route[slot], expected[layer][row][slot]);
        }
    }
    uint32_t route[8];
    check_int("hash duplicate-cycle reject",
              dsv4_cpu_hash_route(0, 0, 14, 3, route), DSV4_CPU_INVALID);
}

static void test_learned_route(void)
{
    for (size_t row = 0; row < DSV4_VEC_ROUTE_ROWS; ++row) {
        float logits[DSV4_VEC_ROUTE_EXPERTS];
        float weights[DSV4_VEC_ROUTE_TOPK];
        uint32_t indices[DSV4_VEC_ROUTE_TOPK];
        int status = dsv4_cpu_learned_route(
            dsv4_vec_route_hidden + row * DSV4_VEC_ROUTE_HIDDEN,
            dsv4_vec_route_weight, dsv4_vec_route_bias,
            DSV4_VEC_ROUTE_HIDDEN, DSV4_VEC_ROUTE_EXPERTS,
            DSV4_VEC_ROUTE_TOPK, 1.5f, logits, indices, weights);
        check_int("learned status", status, DSV4_CPU_OK);
        for (size_t expert = 0; expert < DSV4_VEC_ROUTE_EXPERTS; ++expert)
            check_close("learned logit", logits[expert],
                        dsv4_vec_route_logits[row * DSV4_VEC_ROUTE_EXPERTS + expert],
                        2e-6f);
        float sum = 0.0f;
        for (size_t slot = 0; slot < DSV4_VEC_ROUTE_TOPK; ++slot) {
            check_int("learned index", indices[slot],
                      dsv4_vec_route_indices[row * DSV4_VEC_ROUTE_TOPK + slot]);
            check_close("learned weight", weights[slot],
                        dsv4_vec_route_weights[row * DSV4_VEC_ROUTE_TOPK + slot],
                        4e-6f);
            sum += weights[slot];
        }
        check_close("learned route scale", sum, 1.5f, 3e-6f);
    }

    float zero_bias[DSV4_VEC_ROUTE_EXPERTS] = {0};
    float weights[DSV4_VEC_ROUTE_TOPK];
    uint32_t biased[DSV4_VEC_ROUTE_TOPK], unbiased[DSV4_VEC_ROUTE_TOPK];
    dsv4_cpu_learned_route(dsv4_vec_route_hidden, dsv4_vec_route_weight,
                           dsv4_vec_route_bias, DSV4_VEC_ROUTE_HIDDEN,
                           DSV4_VEC_ROUTE_EXPERTS, DSV4_VEC_ROUTE_TOPK, 1.5f,
                           NULL, biased, weights);
    dsv4_cpu_learned_route(dsv4_vec_route_hidden, dsv4_vec_route_weight,
                           zero_bias, DSV4_VEC_ROUTE_HIDDEN,
                           DSV4_VEC_ROUTE_EXPERTS, DSV4_VEC_ROUTE_TOPK, 1.5f,
                           NULL, unbiased, weights);
    check_int("correction bias changes route", memcmp(biased, unbiased, sizeof biased) != 0, 1);
}

static void test_hyperconnection(void)
{
    float fn[DSV4_VEC_HC_MIX * DSV4_VEC_HC * DSV4_VEC_HIDDEN];
    for (size_t out = 0; out < DSV4_VEC_HC_MIX; ++out) {
        for (size_t in = 0; in < DSV4_VEC_HC * DSV4_VEC_HIDDEN; ++in) {
            int numerator = (int)((out * 13u + in * 7u) % 23u) - 11;
            fn[out * DSV4_VEC_HC * DSV4_VEC_HIDDEN + in] =
                (float)numerator / 100.0f;
        }
    }
    float post[DSV4_VEC_HC];
    float comb[DSV4_VEC_HC * DSV4_VEC_HC];
    float collapsed[DSV4_VEC_HIDDEN];
    int status = dsv4_cpu_hyperconnection(
        dsv4_vec_hc_streams, fn, dsv4_vec_hc_base,
        dsv4_vec_hc_scale, DSV4_VEC_HC, DSV4_VEC_HIDDEN, 20,
        1e-6f, 1e-6f, post, comb, collapsed);
    check_int("hyperconnection status", status, DSV4_CPU_OK);
    for (size_t i = 0; i < DSV4_VEC_HC; ++i)
        check_close("hyperconnection post", post[i], dsv4_vec_hc_post[i], 8e-6f);
    for (size_t i = 0; i < DSV4_VEC_HC * DSV4_VEC_HC; ++i)
        check_close("hyperconnection comb", comb[i], dsv4_vec_hc_comb[i], 1.5e-5f);
    for (size_t i = 0; i < DSV4_VEC_HIDDEN; ++i)
        check_close("hyperconnection collapsed", collapsed[i],
                    dsv4_vec_hc_collapsed[i], 1.5e-5f);

    for (size_t row = 0; row < DSV4_VEC_HC; ++row) {
        float row_sum = 0.0f, col_sum = 0.0f;
        for (size_t col = 0; col < DSV4_VEC_HC; ++col) {
            row_sum += comb[row * DSV4_VEC_HC + col];
            col_sum += comb[col * DSV4_VEC_HC + row];
        }
        check_close("Sinkhorn row sum", row_sum, 1.0f, 2e-4f);
        check_close("Sinkhorn column sum", col_sum, 1.0f, 2e-4f);
    }

    check_int("hyperconnection iteration reject",
              dsv4_cpu_hyperconnection(dsv4_vec_hc_streams, fn,
                  dsv4_vec_hc_base, dsv4_vec_hc_scale, DSV4_VEC_HC,
                  DSV4_VEC_HIDDEN, 0, 1e-6f, 1e-6f, post, comb, collapsed),
              DSV4_CPU_INVALID);
}

int main(void)
{
    test_sqrtsoftplus();
    test_hash_route();
    test_learned_route();
    test_hyperconnection();
    if (failures) {
        fprintf(stderr, "DSV4 CPU OPS: FAIL (%d)\n", failures);
        return 1;
    }
    puts("DSV4 CPU OPS: PASS");
    return 0;
}

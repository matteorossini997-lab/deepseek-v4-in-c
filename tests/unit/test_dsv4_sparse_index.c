#include "dsv4_sparse_index.h"
#include "dsv4_sparse_index_vectors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void expect_close(
    const float *actual,
    const float *expected,
    size_t count,
    float tolerance,
    const char *label) {
    size_t index;

    for (index = 0u; index < count; ++index) {
        const float error = fabsf(actual[index] - expected[index]);
        if (!(error <= tolerance)) {
            fprintf(
                stderr,
                "FAIL: %s[%zu]: got %.9g expected %.9g error %.9g\n",
                label,
                index,
                actual[index],
                expected[index],
                error);
            exit(1);
        }
    }
}

static void expect_indices(
    const size_t *actual,
    const size_t *expected,
    size_t count,
    const char *label) {
    size_t index;

    for (index = 0u; index < count; ++index) {
        if (actual[index] != expected[index]) {
            fprintf(
                stderr,
                "FAIL: %s[%zu]: got %zu expected %zu\n",
                label,
                index,
                actual[index],
                expected[index]);
            exit(1);
        }
    }
}

static void test_canonical_topk(void) {
    float scores[DSV4_SI_TOKENS];
    size_t indices[DSV4_SI_TOKENS];
    size_t count = 0u;

    if (dsv4_sparse_index_topk_f32(
            dsv4_si_q,
            dsv4_si_compressed,
            dsv4_si_head_weights,
            DSV4_SI_HEADS,
            DSV4_SI_TOKENS,
            DSV4_SI_HEAD_DIM,
            DSV4_SI_TOPK,
            scores,
            indices,
            &count) != DSV4_SI_OK) {
        fail("canonical top-k status");
    }
    if (count != DSV4_SI_TOPK) {
        fail("canonical top-k count");
    }
    expect_close(scores, dsv4_si_top_scores, count, 2.0e-6f, "top scores");
    expect_indices(indices, dsv4_si_top_indices, count, "top indices");

    if (dsv4_sparse_index_topk_f32(
            dsv4_si_q,
            dsv4_si_compressed,
            dsv4_si_head_weights,
            DSV4_SI_HEADS,
            DSV4_SI_TOKENS,
            DSV4_SI_HEAD_DIM,
            99u,
            scores,
            indices,
            &count) != DSV4_SI_OK) {
        fail("all-score status");
    }
    if (count != DSV4_SI_TOKENS) {
        fail("all-score count");
    }
    expect_close(scores, dsv4_si_all_scores, count, 2.0e-6f, "all scores");
    expect_indices(indices, dsv4_si_all_indices, count, "all indices");
}

static void test_tie_break(void) {
    const float query[2] = {1.0f, 0.0f};
    const float compressed[6] = {
        1.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float head_weights[1] = {1.0f};
    float scores[2];
    size_t indices[2];
    size_t count = 0u;

    if (dsv4_sparse_index_topk_f32(
            query,
            compressed,
            head_weights,
            1u,
            3u,
            2u,
            2u,
            scores,
            indices,
            &count) != DSV4_SI_OK) {
        fail("tie status");
    }
    if (count != 2u || indices[0] != 0u || indices[1] != 1u) {
        fail("tie order");
    }
}

static void test_rejection_is_transactional(void) {
    float bad_query[DSV4_SI_HEADS * DSV4_SI_HEAD_DIM];
    float scores[DSV4_SI_TOKENS];
    float before_scores[DSV4_SI_TOKENS];
    size_t indices[DSV4_SI_TOKENS];
    size_t before_indices[DSV4_SI_TOKENS];
    size_t count = DSV4_SI_TOKENS;

    memcpy(bad_query, dsv4_si_q, sizeof(bad_query));
    bad_query[5] = NAN;
    memcpy(scores, dsv4_si_all_scores, sizeof(scores));
    memcpy(indices, dsv4_si_all_indices, sizeof(indices));
    memcpy(before_scores, scores, sizeof(scores));
    memcpy(before_indices, indices, sizeof(indices));

    if (dsv4_sparse_index_topk_f32(
            bad_query,
            dsv4_si_compressed,
            dsv4_si_head_weights,
            DSV4_SI_HEADS,
            DSV4_SI_TOKENS,
            DSV4_SI_HEAD_DIM,
            DSV4_SI_TOPK,
            scores,
            indices,
            &count) != DSV4_SI_NONFINITE) {
        fail("non-finite rejection status");
    }
    if (count != DSV4_SI_TOKENS ||
        memcmp(scores, before_scores, sizeof(scores)) != 0 ||
        memcmp(indices, before_indices, sizeof(indices)) != 0) {
        fail("rejected call changed outputs");
    }
}

int main(void) {
    test_canonical_topk();
    test_tie_break();
    test_rejection_is_transactional();
    puts("DeepSeek sparse index tests passed");
    return 0;
}

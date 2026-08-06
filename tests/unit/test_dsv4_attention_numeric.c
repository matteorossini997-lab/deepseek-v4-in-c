#include "dsv4_attention_numeric.h"
#include "dsv4_attention_numeric_vectors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void expect_status(DSV4AttentionNumericStatus actual, DSV4AttentionNumericStatus expected, const char *label) {
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s: got %s, expected %s\n", label,
                dsv4_attention_numeric_status_string(actual),
                dsv4_attention_numeric_status_string(expected));
        exit(1);
    }
}

static void expect_close(const float *actual, const float *expected, size_t count, float atol, const char *label) {
    size_t i;
    for (i = 0u; i < count; ++i) {
        const float error = fabsf(actual[i] - expected[i]);
        if (!(error <= atol)) {
            fprintf(stderr, "FAIL: %s[%zu]: got %.9g expected %.9g error %.9g\n",
                    label, i, actual[i], expected[i], error);
            exit(1);
        }
    }
}

static void test_rope(void) {
    float values[16];
    float unchanged[16];
    memcpy(values, dsv4_an_rope_input, sizeof(values));
    expect_status(dsv4_partial_rope_f32(values, 1u, 16u, 8u, 7u, 10000.0f), DSV4_AN_OK, "rope");
    expect_close(values, dsv4_an_rope_expected, 16u, 2.0e-6f, "rope parity");
    expect_status(dsv4_partial_rope_f32(values, 1u, 16u, 8u, -7, 10000.0f), DSV4_AN_OK, "inverse rope");
    expect_close(values, dsv4_an_rope_input, 16u, 2.0e-6f, "rope inverse parity");

    memcpy(unchanged, dsv4_an_rope_input, sizeof(unchanged));
    expect_status(dsv4_partial_rope_f32(unchanged, 1u, 16u, 3u, 7u, 10000.0f),
                  DSV4_AN_INVALID_ARGUMENT, "odd rope dim");
    expect_close(unchanged, dsv4_an_rope_input, 16u, 0.0f, "rope rejection transactional");
}

static void test_sink_attention(void) {
    float out[64];
    float high_sink[4] = {20.0f, 20.0f, 20.0f, 20.0f};
    float attenuated[64];
    size_t i;
    float base_norm = 0.0f;
    float attenuated_norm = 0.0f;

    expect_status(dsv4_sink_attention_shared_kv_f32(
                      dsv4_an_q, dsv4_an_kv, dsv4_an_sinks, 4u, 3u, 16u, 0.25f, out),
                  DSV4_AN_OK, "sink attention");
    expect_close(out, dsv4_an_attention_expected, 64u, 3.0e-6f, "sink attention parity");

    expect_status(dsv4_sink_attention_shared_kv_f32(
                      dsv4_an_q, dsv4_an_kv, high_sink, 4u, 3u, 16u, 0.25f, attenuated),
                  DSV4_AN_OK, "high sink attention");
    for (i = 0u; i < 64u; ++i) {
        base_norm += out[i] * out[i];
        attenuated_norm += attenuated[i] * attenuated[i];
    }
    if (!(attenuated_norm < base_norm * 1.0e-6f)) {
        fail("sink mass did not suppress value output");
    }
}

static void test_compressor_case(
    int overlap,
    const float *kv1,
    const float *gate1,
    const float *kv2,
    const float *gate2,
    const float *bias,
    const float *norm_weight,
    const float *expected1,
    const float *expected2,
    const char *label) {
    DSV4CompressorNumericState state;
    float previous_kv[16];
    float previous_gate[16];
    float out[4];
    float previous_snapshot[16];
    float bad_gate[32];
    size_t projected_count = overlap != 0 ? 32u : 16u;

    expect_status(dsv4_compressor_numeric_init(
                      &state, 4u, 4u, 4u, overlap, 1.0e-6f, 160000.0f,
                      overlap != 0 ? previous_kv : NULL,
                      overlap != 0 ? previous_gate : NULL),
                  DSV4_AN_OK, "compressor init");
    expect_status(dsv4_compress_projected_window_f32(
                      &state, kv1, gate1, bias, norm_weight, out),
                  DSV4_AN_OK, label);
    expect_close(out, expected1, 4u, 3.0e-6f, label);
    if (state.emitted != 1u || state.has_previous != overlap) {
        fail("compressor state after first emission");
    }

    expect_status(dsv4_compress_projected_window_f32(
                      &state, kv2, gate2, bias, norm_weight, out),
                  DSV4_AN_OK, label);
    expect_close(out, expected2, 4u, 3.0e-6f, label);
    if (state.emitted != 2u) {
        fail("compressor state after second emission");
    }

    memcpy(bad_gate, gate2, projected_count * sizeof(float));
    bad_gate[projected_count - 1u] = NAN;
    if (overlap != 0) {
        memcpy(previous_snapshot, previous_kv, sizeof(previous_snapshot));
    }
    expect_status(dsv4_compress_projected_window_f32(
                      &state, kv2, bad_gate, bias, norm_weight, out),
                  DSV4_AN_NONFINITE, "nonfinite compressor rejection");
    if (state.emitted != 2u) {
        fail("rejected compressor step advanced emitted count");
    }
    if (overlap != 0) {
        expect_close(previous_kv, previous_snapshot, 16u, 0.0f, "rejected compressor state");
    }
}

int main(void) {
    DSV4CompressorNumericState invalid;
    float only_one_buffer[16];

    test_rope();
    test_sink_attention();
    test_compressor_case(1, dsv4_an_csa_kv1, dsv4_an_csa_gate1,
                         dsv4_an_csa_kv2, dsv4_an_csa_gate2,
                         dsv4_an_csa_bias, dsv4_an_csa_norm_weight,
                         dsv4_an_csa_expected1, dsv4_an_csa_expected2, "CSA compressor");
    test_compressor_case(0, dsv4_an_hca_kv1, dsv4_an_hca_gate1,
                         dsv4_an_hca_kv2, dsv4_an_hca_gate2,
                         dsv4_an_hca_bias, dsv4_an_hca_norm_weight,
                         dsv4_an_hca_expected1, dsv4_an_hca_expected2, "HCA compressor");

    expect_status(dsv4_compressor_numeric_init(&invalid, 4u, 4u, 4u, 1, 1.0e-6f,
                                               160000.0f, only_one_buffer, NULL),
                  DSV4_AN_INVALID_ARGUMENT, "overlap buffers");
    puts("DeepSeek attention numeric tests passed");
    return 0;
}

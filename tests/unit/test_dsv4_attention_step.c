#include "dsv4_attention_step.h"
#include "dsv4_attention_step_vectors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void closev(const float *actual,
                   const float *expected,
                   size_t count,
                   float tolerance,
                   const char *label) {
    size_t index;
    for (index = 0u; index < count; ++index) {
        const float error = fabsf(actual[index] - expected[index]);
        if (!(error <= tolerance)) {
            fprintf(stderr,
                    "FAIL %s[%zu] %.9g %.9g e=%.9g\n",
                    label,
                    index,
                    actual[index],
                    expected[index],
                    error);
            exit(1);
        }
    }
}

static DSV4AttentionStepConfig config_for(int sparse, size_t compressed_tokens) {
    DSV4AttentionStepConfig config = {
        DSV4_STEP_HEADS,
        DSV4_STEP_HEAD_DIM,
        DSV4_STEP_ROPE_DIM,
        DSV4_STEP_RAW_TOKENS,
        compressed_tokens,
        DSV4_STEP_INDEX_HEADS,
        DSV4_STEP_INDEX_DIM,
        DSV4_STEP_INDEX_ROPE_DIM,
        DSV4_STEP_TOPK,
        DSV4_STEP_GROUPS,
        DSV4_STEP_RANK,
        DSV4_STEP_HIDDEN,
        160000.0f,
        sparse,
    };
    return config;
}

int main(void) {
    float output[DSV4_STEP_HIDDEN];
    float before[DSV4_STEP_HIDDEN];
    size_t selected[DSV4_STEP_TOPK];
    size_t count = 99u;
    DSV4AttentionStepConfig config =
        config_for(1, DSV4_STEP_COMPRESSED_TOKENS);

    if (dsv4_attention_postprojected_step_f32(
            &config,
            DSV4_STEP_POSITION,
            dsv4_step_q,
            dsv4_step_raw,
            dsv4_step_compressed,
            dsv4_step_index_q,
            dsv4_step_index_compressed,
            dsv4_step_index_head_weights,
            dsv4_step_sinks,
            dsv4_step_oa,
            dsv4_step_ob,
            output,
            selected,
            &count) != DSV4_AS_OK) {
        die("CSA status");
    }
    if (count != DSV4_STEP_TOPK ||
        memcmp(selected, dsv4_step_selected, sizeof(selected)) != 0) {
        die("CSA selected");
    }
    closev(output, dsv4_step_expected_csa, DSV4_STEP_HIDDEN, 4.0e-6f, "CSA");

    config = config_for(0, DSV4_STEP_COMPRESSED_TOKENS);
    count = 99u;
    if (dsv4_attention_postprojected_step_f32(
            &config,
            DSV4_STEP_POSITION,
            dsv4_step_q,
            dsv4_step_raw,
            dsv4_step_compressed,
            NULL,
            NULL,
            NULL,
            dsv4_step_sinks,
            dsv4_step_oa,
            dsv4_step_ob,
            output,
            NULL,
            &count) != DSV4_AS_OK) {
        die("HCA status");
    }
    if (count != 0u) {
        die("HCA index count");
    }
    closev(output, dsv4_step_expected_hca, DSV4_STEP_HIDDEN, 4.0e-6f, "HCA");

    config = config_for(0, 0u);
    count = 99u;
    if (dsv4_attention_postprojected_step_f32(
            &config,
            DSV4_STEP_POSITION,
            dsv4_step_q,
            dsv4_step_raw,
            NULL,
            NULL,
            NULL,
            NULL,
            dsv4_step_sinks,
            dsv4_step_oa,
            dsv4_step_ob,
            output,
            NULL,
            &count) != DSV4_AS_OK) {
        die("sliding status");
    }
    if (count != 0u) {
        die("sliding index count");
    }
    closev(output,
           dsv4_step_expected_sliding,
           DSV4_STEP_HIDDEN,
           4.0e-6f,
           "sliding");

    config = config_for(1, DSV4_STEP_COMPRESSED_TOKENS);
    for (size_t index = 0u; index < DSV4_STEP_HIDDEN; ++index) {
        output[index] = (float)index;
        before[index] = (float)index;
    }
    {
        float bad_query[DSV4_STEP_HEADS * DSV4_STEP_HEAD_DIM];
        memcpy(bad_query, dsv4_step_q, sizeof(bad_query));
        bad_query[2] = NAN;
        count = 77u;
        if (dsv4_attention_postprojected_step_f32(
                &config,
                DSV4_STEP_POSITION,
                bad_query,
                dsv4_step_raw,
                dsv4_step_compressed,
                dsv4_step_index_q,
                dsv4_step_index_compressed,
                dsv4_step_index_head_weights,
                dsv4_step_sinks,
                dsv4_step_oa,
                dsv4_step_ob,
                output,
                selected,
                &count) != DSV4_AS_NONFINITE) {
            die("reject status");
        }
    }
    if (count != 77u || memcmp(output, before, sizeof(output)) != 0) {
        die("transaction");
    }

    puts("DeepSeek attention step tests passed");
    return 0;
}

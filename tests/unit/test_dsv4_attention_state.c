#include "dsv4_attention_state.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check_u64(const char *name, uint64_t got, uint64_t want)
{
    if (got != want) {
        fprintf(stderr, "FAIL %-28s got %llu want %llu\n", name,
                (unsigned long long)got, (unsigned long long)want);
        ++failures;
    }
}

static void check_float(const char *name, float got, float want)
{
    if (!isfinite(got) || fabsf(got - want) > 1e-6f) {
        fprintf(stderr, "FAIL %-28s got %.8g want %.8g\n", name, got, want);
        ++failures;
    }
}

static void test_sliding(void)
{
    DSV4AttentionState state;
    check_u64("sliding init", dsv4_attention_state_init(&state, 16, 0), 0);
    for (uint64_t position = 0; position < 21; ++position) {
        DSV4AttentionTrace trace;
        check_u64("sliding step",
                  dsv4_attention_state_step(&state, position, &trace), 0);
        size_t expected_length = position + 1u < 16u
            ? (size_t)(position + 1u) : 16u;
        uint64_t expected_start = position + 1u - expected_length;
        check_u64("sliding length", trace.raw_cache_length, expected_length);
        check_u64("sliding start", trace.raw_start_position, expected_start);
        check_u64("sliding slot", trace.raw_write_slot, position % 16u);
        check_u64("sliding compressed", trace.compressed_count, 0);
    }
}

static void test_csa(void)
{
    static const uint64_t expected[9] = {0,0,0,1,1,1,1,2,2};
    DSV4AttentionState state;
    check_u64("CSA init", dsv4_attention_state_init(&state, 16, 4), 0);
    for (uint64_t position = 0; position < 9; ++position) {
        DSV4AttentionTrace trace;
        check_u64("CSA step",
                  dsv4_attention_state_step(&state, position, &trace), 0);
        check_u64("CSA count", trace.compressed_count, expected[position]);
        check_u64("CSA index count", trace.indexer_compressed_count,
                  expected[position]);
        check_u64("CSA emitted", trace.compressed_emitted,
                  position == 3 || position == 7);
        check_u64("CSA index emitted", trace.indexer_emitted,
                  position == 3 || position == 7);
    }
}

static void test_hca(void)
{
    DSV4AttentionState state;
    check_u64("HCA init", dsv4_attention_state_init(&state, 16, 128), 0);
    for (uint64_t position = 0; position < 130; ++position) {
        DSV4AttentionTrace trace;
        check_u64("HCA step",
                  dsv4_attention_state_step(&state, position, &trace), 0);
        check_u64("HCA count", trace.compressed_count,
                  position < 127 ? 0 : 1);
        check_u64("HCA no index", trace.indexer_compressed_count, 0);
        check_u64("HCA emitted", trace.compressed_emitted,
                  position == 127);
    }
}

static void test_state_rejections(void)
{
    DSV4AttentionState state;
    DSV4AttentionTrace trace;
    check_u64("bad rate",
              dsv4_attention_state_init(&state, 16, 8) != 0, 1);
    dsv4_attention_state_init(&state, 16, 4);
    check_u64("noncontiguous reject",
              dsv4_attention_state_step(&state, 1, &trace) != 0, 1);
    check_u64("state unchanged", state.next_position, 0);
}

static void test_grouped_linear(void)
{
    static const float x[12] = {
        1,2,3, -1,0.5f,2,
        0,-1,1, 2,1,-0.5f
    };
    static const float weight[12] = {
        0.1f,0.2f,0.3f, -0.2f,0.5f,0.1f,
        0.4f,-0.1f,0.2f, 0.3f,0.25f,-0.2f
    };
    static const float expected[8] = {
        1.4f,1.1f,-0.05f,-0.575f,0.1f,-0.4f,0.6f,0.95f
    };
    float output[8] = {0};
    check_u64("grouped status",
              dsv4_cpu_grouped_linear(x, weight, 2, 2, 3, 2, output), 0);
    for (size_t i = 0; i < 8; ++i)
        check_float("grouped value", output[i], expected[i]);
}

static void test_shared_kv(void)
{
    static const float kv[6] = {1,2,3,-1,0.5f,4};
    float output[24] = {0};
    check_u64("shared KV status",
              dsv4_cpu_repeat_shared_kv(kv, 2, 3, 4, output), 0);
    for (size_t token = 0; token < 2; ++token)
        for (size_t head = 0; head < 4; ++head)
            for (size_t dim = 0; dim < 3; ++dim)
                check_float("shared KV value",
                            output[(token * 4 + head) * 3 + dim],
                            kv[token * 3 + dim]);
}

int main(void)
{
    test_sliding();
    test_csa();
    test_hca();
    test_state_rejections();
    test_grouped_linear();
    test_shared_kv();
    if (failures) {
        fprintf(stderr, "DSV4 ATTENTION STATE: FAIL (%d)\n", failures);
        return 1;
    }
    puts("DSV4 ATTENTION STATE: PASS");
    return 0;
}

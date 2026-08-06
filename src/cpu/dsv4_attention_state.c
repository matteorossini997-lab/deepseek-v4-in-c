#include "dsv4_attention_state.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int mul_size(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int all_finite(const float *values, size_t count)
{
    if (!values) return 0;
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(values[i])) return 0;
    return 1;
}

int dsv4_attention_state_init(DSV4AttentionState *state,
                              size_t sliding_window,
                              size_t compression_rate)
{
    if (!state || sliding_window == 0 ||
        (compression_rate != DSV4_ATTENTION_SLIDING &&
         compression_rate != DSV4_ATTENTION_CSA &&
         compression_rate != DSV4_ATTENTION_HCA))
        return -1;
    memset(state, 0, sizeof(*state));
    state->sliding_window = sliding_window;
    state->compression_rate = compression_rate;
    state->has_indexer = compression_rate == DSV4_ATTENTION_CSA;
    return 0;
}

int dsv4_attention_state_step(DSV4AttentionState *state,
                              uint64_t position,
                              DSV4AttentionTrace *trace)
{
    if (!state || !trace || state->sliding_window == 0 ||
        position != state->next_position || position == UINT64_MAX)
        return -1;

    DSV4AttentionState next = *state;
    DSV4AttentionTrace out;
    memset(&out, 0, sizeof(out));

    out.raw_write_slot = (size_t)(position % next.sliding_window);
    if (next.raw_length < next.sliding_window)
        ++next.raw_length;
    next.raw_start_position = position + 1u - (uint64_t)next.raw_length;

    if (next.compression_rate != DSV4_ATTENTION_SLIDING) {
        ++next.pending_compressor_tokens;
        if (next.pending_compressor_tokens == next.compression_rate) {
            next.pending_compressor_tokens = 0;
            if (next.compressed_count == UINT64_MAX) return -1;
            ++next.compressed_count;
            out.compressed_emitted = 1;
            if (next.has_indexer) {
                if (next.indexer_compressed_count == UINT64_MAX) return -1;
                ++next.indexer_compressed_count;
                out.indexer_emitted = 1;
            }
        }
    }

    next.next_position = position + 1u;
    out.raw_cache_length = next.raw_length;
    out.raw_start_position = next.raw_start_position;
    out.compressed_count = next.compressed_count;
    out.indexer_compressed_count = next.indexer_compressed_count;
    *state = next;
    *trace = out;
    return 0;
}

int dsv4_cpu_grouped_linear(const float *x,
                            const float *weight,
                            size_t tokens,
                            size_t groups,
                            size_t in_per_group,
                            size_t out_per_group,
                            float *output)
{
    size_t x_count = 0, w_count = 0, y_count = 0, tmp = 0;
    if (!x || !weight || !output || tokens == 0 || groups == 0 ||
        in_per_group == 0 || out_per_group == 0 ||
        !mul_size(tokens, groups, &tmp) ||
        !mul_size(tmp, in_per_group, &x_count) ||
        !mul_size(groups, out_per_group, &tmp) ||
        !mul_size(tmp, in_per_group, &w_count) ||
        !mul_size(tokens, groups, &tmp) ||
        !mul_size(tmp, out_per_group, &y_count) ||
        !all_finite(x, x_count) || !all_finite(weight, w_count))
        return -1;

    for (size_t token = 0; token < tokens; ++token) {
        for (size_t group = 0; group < groups; ++group) {
            const float *input = x + (token * groups + group) * in_per_group;
            for (size_t out = 0; out < out_per_group; ++out) {
                const float *row = weight +
                    (group * out_per_group + out) * in_per_group;
                float sum = 0.0f;
                for (size_t in = 0; in < in_per_group; ++in)
                    sum += input[in] * row[in];
                if (!isfinite(sum)) return -1;
                output[(token * groups + group) * out_per_group + out] = sum;
            }
        }
    }
    (void)y_count;
    return 0;
}

int dsv4_cpu_repeat_shared_kv(const float *kv,
                              size_t tokens,
                              size_t head_dim,
                              size_t num_heads,
                              float *output)
{
    size_t kv_count = 0, token_stride = 0, output_count = 0;
    if (!kv || !output || tokens == 0 || head_dim == 0 || num_heads == 0 ||
        head_dim > SIZE_MAX / sizeof(*kv) ||
        !mul_size(tokens, head_dim, &kv_count) ||
        !mul_size(num_heads, head_dim, &token_stride) ||
        !mul_size(tokens, token_stride, &output_count) ||
        !all_finite(kv, kv_count))
        return -1;

    for (size_t token = 0; token < tokens; ++token) {
        const float *source = kv + token * head_dim;
        for (size_t head = 0; head < num_heads; ++head)
            memcpy(output + token * token_stride + head * head_dim,
                   source, head_dim * sizeof(*source));
    }
    (void)output_count;
    return 0;
}

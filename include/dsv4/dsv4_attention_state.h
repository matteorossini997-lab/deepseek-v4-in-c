#ifndef DSV4_ATTENTION_STATE_H
#define DSV4_ATTENTION_STATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DSV4_ATTENTION_SLIDING = 0,
    DSV4_ATTENTION_CSA = 4,
    DSV4_ATTENTION_HCA = 128
} DSV4AttentionRate;

typedef struct {
    size_t sliding_window;
    size_t compression_rate;
    uint64_t next_position;
    uint64_t raw_start_position;
    size_t raw_length;
    size_t pending_compressor_tokens;
    uint64_t compressed_count;
    uint64_t indexer_compressed_count;
    int has_indexer;
} DSV4AttentionState;

typedef struct {
    size_t raw_cache_length;
    uint64_t raw_start_position;
    size_t raw_write_slot;
    uint64_t compressed_count;
    uint64_t indexer_compressed_count;
    int compressed_emitted;
    int indexer_emitted;
} DSV4AttentionTrace;

int dsv4_attention_state_init(DSV4AttentionState *state,
                              size_t sliding_window,
                              size_t compression_rate);

int dsv4_attention_state_step(DSV4AttentionState *state,
                              uint64_t position,
                              DSV4AttentionTrace *trace);

/* x: [tokens, groups, in_per_group]
 * weight: [groups, out_per_group, in_per_group]
 * output: [tokens, groups, out_per_group]
 */
int dsv4_cpu_grouped_linear(const float *x,
                            const float *weight,
                            size_t tokens,
                            size_t groups,
                            size_t in_per_group,
                            size_t out_per_group,
                            float *output);

/* Repeat one shared KV head across query heads.
 * kv: [tokens, head_dim]
 * output: [tokens, num_heads, head_dim]
 */
int dsv4_cpu_repeat_shared_kv(const float *kv,
                              size_t tokens,
                              size_t head_dim,
                              size_t num_heads,
                              float *output);

#ifdef __cplusplus
}
#endif

#endif

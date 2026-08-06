/* SPDX-License-Identifier: Apache-2.0 */
#ifndef DSV4_CONFIG_H
#define DSV4_CONFIG_H

#include <stdint.h>

#define DSV4_MAX_COMPRESS_RATIOS 128
#define DSV4_CONFIG_ERROR_CAP 4096

typedef enum {
    DSV4_CONFIG_UNKNOWN = 0,
    DSV4_CONFIG_HF = 1,
    DSV4_CONFIG_INFERENCE = 2,
    DSV4_CONFIG_MIXED = 3
} DSV4ConfigLayout;

enum {
    DSV4_CONTRACT_NORM_EPS      = 1u << 0,
    DSV4_CONTRACT_HC_EPS        = 1u << 1,
    DSV4_CONTRACT_MTP_LAYERS    = 1u << 2,
    DSV4_CONTRACT_FP8_BLOCK     = 1u << 3,
    DSV4_CONTRACT_FP4_BLOCK     = 1u << 4,
    DSV4_CONTRACT_MAX_SEQ_LEN   = 1u << 5,
    DSV4_CONTRACT_TOPK_METHOD   = 1u << 6,
    DSV4_CONTRACT_NORM_TOPK     = 1u << 7,
    DSV4_CONTRACT_KV_HEADS      = 1u << 8
};

typedef struct {
    DSV4ConfigLayout layout;
    unsigned contract_mask;

    int vocab_size;
    int dim;
    int moe_inter_dim;
    int n_layers;
    int n_hash_layers;
    int n_mtp_layers;
    int n_heads;
    int n_kv_heads;

    int n_routed_experts;
    int n_shared_experts;
    int n_activated_experts;
    char score_func[24];
    char topk_method[24];
    int norm_topk_prob;
    double route_scale;
    double swiglu_limit;

    int q_lora_rank;
    int head_dim;
    int rope_head_dim;
    int o_groups;
    int o_lora_rank;
    int window_size;

    int64_t original_seq_len;
    int64_t max_seq_len;
    double rope_theta;
    double rope_factor;
    double beta_fast;
    double beta_slow;
    double compress_rope_theta;

    int index_n_heads;
    int index_head_dim;
    int index_topk;
    int hc_mult;
    int hc_sinkhorn_iters;
    double hc_eps;
    double norm_eps;

    char dtype[16];
    char expert_dtype[16];
    char scale_fmt[16];
    int fp8_block_rows;
    int fp8_block_cols;
    int fp4_block_size;

    int n_compress_ratios;
    int compress_ratios[DSV4_MAX_COMPRESS_RATIOS];
} DSV4Config;

typedef struct {
    int count;
    char message[DSV4_CONFIG_ERROR_CAP];
} DSV4ConfigError;

int dsv4_config_load_file(DSV4Config *cfg, const char *path, DSV4ConfigError *err);
int dsv4_config_is_flash_profile(const DSV4Config *cfg, DSV4ConfigError *err);
const char *dsv4_config_layout_name(DSV4ConfigLayout layout);

#endif

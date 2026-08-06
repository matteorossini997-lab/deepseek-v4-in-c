/* SPDX-License-Identifier: Apache-2.0 */
/* P0-B RED gate: production header/source intentionally absent in the first commit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_config.h"

static int failures;

static void check_int(int condition, const char *name, long got, long want)
{
    if (condition) {
        printf("  ok    %-28s %ld\n", name, got);
    } else {
        printf("  FAIL  %-28s got %ld, want %ld\n", name, got, want);
        failures++;
    }
}

static void check_text(int condition, const char *name, const char *got, const char *want)
{
    if (condition) {
        printf("  ok    %-28s %s\n", name, got);
    } else {
        printf("  FAIL  %-28s got %s, want %s\n", name, got, want);
        failures++;
    }
}

static int check_common(const DSV4Config *cfg)
{
    check_int(cfg->vocab_size == 129280, "vocab", cfg->vocab_size, 129280);
    check_int(cfg->dim == 4096, "hidden", cfg->dim, 4096);
    check_int(cfg->moe_inter_dim == 2048, "moe intermediate", cfg->moe_inter_dim, 2048);
    check_int(cfg->n_layers == 43, "base layers", cfg->n_layers, 43);
    check_int(cfg->n_hash_layers == 3, "hash layers", cfg->n_hash_layers, 3);
    check_int(cfg->n_mtp_layers == 1, "MTP layers", cfg->n_mtp_layers, 1);
    check_int(cfg->n_heads == 64, "attention heads", cfg->n_heads, 64);
    check_int(cfg->n_routed_experts == 256, "routed experts", cfg->n_routed_experts, 256);
    check_int(cfg->n_shared_experts == 1, "shared experts", cfg->n_shared_experts, 1);
    check_int(cfg->n_activated_experts == 6, "active experts", cfg->n_activated_experts, 6);
    check_text(!strcmp(cfg->score_func, "sqrtsoftplus"), "score function",
               cfg->score_func, "sqrtsoftplus");
    check_int(cfg->q_lora_rank == 1024, "q lora rank", cfg->q_lora_rank, 1024);
    check_int(cfg->head_dim == 512, "head dim", cfg->head_dim, 512);
    check_int(cfg->rope_head_dim == 64, "rope head dim", cfg->rope_head_dim, 64);
    check_int(cfg->window_size == 128, "window", cfg->window_size, 128);
    check_int(cfg->original_seq_len == 65536, "original sequence", cfg->original_seq_len, 65536);
    check_int(cfg->max_seq_len == 1048576, "maximum sequence", cfg->max_seq_len, 1048576);
    check_int(cfg->index_n_heads == 64, "index heads", cfg->index_n_heads, 64);
    check_int(cfg->index_head_dim == 128, "index head dim", cfg->index_head_dim, 128);
    check_int(cfg->index_topk == 512, "index topk", cfg->index_topk, 512);
    check_int(cfg->hc_mult == 4, "HC multiplier", cfg->hc_mult, 4);
    check_int(cfg->hc_sinkhorn_iters == 20, "Sinkhorn iterations",
              cfg->hc_sinkhorn_iters, 20);
    check_text(!strcmp(cfg->dtype, "fp8"), "trunk dtype", cfg->dtype, "fp8");
    check_text(!strcmp(cfg->expert_dtype, "fp4"), "expert dtype",
               cfg->expert_dtype, "fp4");
    check_text(!strcmp(cfg->scale_fmt, "ue8m0"), "scale format",
               cfg->scale_fmt, "ue8m0");
    check_int(cfg->fp8_block_rows == 128 && cfg->fp8_block_cols == 128,
              "FP8 block 128x128", cfg->fp8_block_rows, 128);
    check_int(cfg->fp4_block_size == 32, "FP4 block", cfg->fp4_block_size, 32);
    check_int(cfg->n_compress_ratios == 44, "compression entries",
              cfg->n_compress_ratios, 44);
    check_int(cfg->compress_ratios[0] == 0 && cfg->compress_ratios[2] == 4 &&
              cfg->compress_ratios[3] == 128 && cfg->compress_ratios[43] == 0,
              "compression schedule", cfg->compress_ratios[3], 128);
    return failures == 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: test_dsv4_config <hf|inference|reject> <path> [needle]\n");
        return 2;
    }

    DSV4Config cfg;
    DSV4ConfigError err;
    int ok = dsv4_config_load_file(&cfg, argv[2], &err);

    if (!strcmp(argv[1], "reject")) {
        if (ok) {
            fprintf(stderr, "expected rejection but config loaded\n");
            return 1;
        }
        if (argc >= 4 && !strstr(err.message, argv[3])) {
            fprintf(stderr, "error did not contain '%s': %s\n", argv[3], err.message);
            return 1;
        }
        printf("  ok    rejected: %s\n", err.message);
        return 0;
    }

    if (!ok) {
        fprintf(stderr, "load failed: %s\n", err.message);
        return 1;
    }

    DSV4ConfigLayout expected = !strcmp(argv[1], "hf")
                                    ? DSV4_CONFIG_HF
                                    : DSV4_CONFIG_INFERENCE;
    check_int(cfg.layout == expected, "layout", cfg.layout, expected);
    check_text(!strcmp(dsv4_config_layout_name(cfg.layout), argv[1]), "layout name",
               dsv4_config_layout_name(cfg.layout), argv[1]);
    check_common(&cfg);

    DSV4ConfigError profile_err;
    check_int(dsv4_config_is_flash_profile(&cfg, &profile_err), "Flash profile", 1, 1);

    if (cfg.layout == DSV4_CONFIG_INFERENCE) {
        unsigned required = DSV4_CONTRACT_NORM_EPS | DSV4_CONTRACT_HC_EPS |
                            DSV4_CONTRACT_MTP_LAYERS | DSV4_CONTRACT_FP8_BLOCK |
                            DSV4_CONTRACT_FP4_BLOCK | DSV4_CONTRACT_MAX_SEQ_LEN;
        check_int((cfg.contract_mask & required) == required, "contract-derived mask",
                  (long)(cfg.contract_mask & required), (long)required);
    } else {
        check_int((cfg.contract_mask & DSV4_CONTRACT_FP4_BLOCK) != 0,
                  "HF FP4 block contract", cfg.contract_mask, DSV4_CONTRACT_FP4_BLOCK);
    }

    printf("\n%s\n", failures ? "DSV4 CONFIG: FAIL" : "DSV4 CONFIG: PASS");
    return failures ? 1 : 0;
}

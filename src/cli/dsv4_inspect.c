/* SPDX-License-Identifier: Apache-2.0 */
#include <stdio.h>
#include <string.h>

#include "dsv4_config.h"

static void usage(FILE *out)
{
    fprintf(out, "usage: dsv4-inspect [--json] <config.json>\n");
}

int main(int argc, char **argv)
{
    int json = 0;
    const char *path = NULL;
    if (argc == 2) {
        path = argv[1];
    } else if (argc == 3 && !strcmp(argv[1], "--json")) {
        json = 1;
        path = argv[2];
    } else {
        usage(stderr);
        return 2;
    }

    DSV4Config cfg;
    DSV4ConfigError err;
    if (!dsv4_config_load_file(&cfg, path, &err)) {
        fprintf(stderr, "dsv4-inspect: %s\n", err.message);
        return 3;
    }

    DSV4ConfigError profile_err;
    if (!dsv4_config_is_flash_profile(&cfg, &profile_err)) {
        fprintf(stderr, "dsv4-inspect: not the supported DeepSeek V4 Flash profile: %s\n",
                profile_err.message);
        return 3;
    }

    if (json) {
        printf("{\"profile\":\"deepseek-v4-flash\",\"layout\":\"%s\","
               "\"layers\":%d,\"mtp_layers\":%d,\"hidden\":%d,"
               "\"routed_experts\":%d,\"active_experts\":%d,"
               "\"trunk_dtype\":\"%s\",\"expert_dtype\":\"%s\","
               "\"fp8_block\":[%d,%d],\"fp4_block\":%d,"
               "\"max_seq_len\":%lld,\"contract_mask\":%u}\n",
               dsv4_config_layout_name(cfg.layout), cfg.n_layers, cfg.n_mtp_layers,
               cfg.dim, cfg.n_routed_experts, cfg.n_activated_experts,
               cfg.dtype, cfg.expert_dtype, cfg.fp8_block_rows, cfg.fp8_block_cols,
               cfg.fp4_block_size, (long long)cfg.max_seq_len, cfg.contract_mask);
    } else {
        printf("profile          : deepseek-v4-flash\n");
        printf("layout           : %s\n", dsv4_config_layout_name(cfg.layout));
        printf("layers           : %d + %d MTP\n", cfg.n_layers, cfg.n_mtp_layers);
        printf("hidden           : %d\n", cfg.dim);
        printf("experts          : %d routed, top-%d, %d shared\n",
               cfg.n_routed_experts, cfg.n_activated_experts, cfg.n_shared_experts);
        printf("quantization     : %s trunk, %s experts, %dx%d/%d blocks\n",
               cfg.dtype, cfg.expert_dtype, cfg.fp8_block_rows,
               cfg.fp8_block_cols, cfg.fp4_block_size);
        printf("maximum context  : %lld\n", (long long)cfg.max_seq_len);
        printf("contract-derived : 0x%08x\n", cfg.contract_mask);
    }
    return 0;
}

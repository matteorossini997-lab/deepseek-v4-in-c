/* SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_config.h"
#include "dsv4_inventory.h"

static void usage(FILE *stream)
{
    fprintf(stream,
            "usage: dsv4-inventory [--json] [--tensors] [--config FILE] "
            "[--tsv FILE] MODEL_DIR\n");
}

int main(int argc, char **argv)
{
    int json = 0;
    int tensors = 0;
    const char *config_path = NULL;
    const char *tsv_path = NULL;
    const char *model_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--json")) {
            json = 1;
        } else if (!strcmp(argv[i], "--tensors")) {
            tensors = 1;
        } else if (!strcmp(argv[i], "--config")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            config_path = argv[i];
        } else if (!strcmp(argv[i], "--tsv")) {
            if (++i >= argc) {
                usage(stderr);
                return 2;
            }
            tsv_path = argv[i];
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(stdout);
            return 0;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "dsv4-inventory: unknown option %s\n", argv[i]);
            usage(stderr);
            return 2;
        } else if (model_dir) {
            fprintf(stderr, "dsv4-inventory: only one model directory is allowed\n");
            return 2;
        } else {
            model_dir = argv[i];
        }
    }
    if (!model_dir) {
        usage(stderr);
        return 2;
    }

    DSV4Inventory inventory;
    DSV4InventoryError error;
    if (dsv4_inventory_open(&inventory, model_dir, &error) != 0) {
        fprintf(stderr, "dsv4-inventory: %s\n", error.message);
        return 1;
    }

    if (config_path) {
        DSV4Config config;
        DSV4ConfigError config_error;
        if (!dsv4_config_load_file(&config, config_path, &config_error)) {
            fprintf(stderr, "dsv4-inventory: %s\n", config_error.message);
            dsv4_inventory_close(&inventory);
            return 1;
        }
        if (!dsv4_config_is_flash_profile(&config, &config_error)) {
            fprintf(stderr, "dsv4-inventory: %s\n", config_error.message);
            dsv4_inventory_close(&inventory);
            return 1;
        }
        if (dsv4_inventory_validate_flash(&inventory, &config, &error) != 0) {
            fprintf(stderr, "dsv4-inventory: %s\n", error.message);
            dsv4_inventory_close(&inventory);
            return 1;
        }
    }

    if (json) {
        if (dsv4_inventory_write_json(&inventory, stdout, tensors) != 0) {
            fprintf(stderr, "dsv4-inventory: failed to write JSON\n");
            dsv4_inventory_close(&inventory);
            return 1;
        }
    } else {
        printf("shards              : %zu\n", inventory.nshards);
        printf("tensors             : %zu\n", inventory.ntensors);
        printf("declared data bytes : %llu\n",
               (unsigned long long)inventory.declared_total_size);
        printf("indexed data bytes  : %llu\n",
               (unsigned long long)inventory.total_data_bytes);
        printf("profile validated   : %s\n",
               inventory.profile_validated ? "yes" : "no");
        for (int i = 0; i < DSV4_CLASS_COUNT; i++) {
            if (!inventory.count_by_class[i]) continue;
            printf("class %-16s : %zu tensors, %llu bytes\n",
                   dsv4_tensor_class_name((DSV4TensorClass)i),
                   inventory.count_by_class[i],
                   (unsigned long long)inventory.bytes_by_class[i]);
        }
    }

    if (tsv_path) {
        FILE *tsv = fopen(tsv_path, "wb");
        if (!tsv) {
            perror(tsv_path);
            dsv4_inventory_close(&inventory);
            return 1;
        }
        int rc = dsv4_inventory_write_tsv(&inventory, tsv);
        if (fclose(tsv) != 0) rc = -1;
        if (rc != 0) {
            fprintf(stderr, "dsv4-inventory: failed to write %s\n", tsv_path);
            dsv4_inventory_close(&inventory);
            return 1;
        }
    }

    dsv4_inventory_close(&inventory);
    return 0;
}

/* SPDX-License-Identifier: Apache-2.0 */
#ifndef DSV4_INVENTORY_H
#define DSV4_INVENTORY_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "dsv4_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DSV4_INV_MAX_DIMS 8
#define DSV4_INV_MAX_NAME 4096
#define DSV4_INV_ERROR_CAP 4096

typedef enum {
    DSV4_DT_UNKNOWN = 0,
    DSV4_DT_BOOL,
    DSV4_DT_F4,
    DSV4_DT_F6_E2M3,
    DSV4_DT_F6_E3M2,
    DSV4_DT_U8,
    DSV4_DT_I8,
    DSV4_DT_F8_E5M2,
    DSV4_DT_F8_E4M3,
    DSV4_DT_F8_E8M0,
    DSV4_DT_F8_E4M3FNUZ,
    DSV4_DT_F8_E5M2FNUZ,
    DSV4_DT_I16,
    DSV4_DT_U16,
    DSV4_DT_F16,
    DSV4_DT_BF16,
    DSV4_DT_I32,
    DSV4_DT_U32,
    DSV4_DT_F32,
    DSV4_DT_C64,
    DSV4_DT_F64,
    DSV4_DT_I64,
    DSV4_DT_U64,
    DSV4_DT_COUNT
} DSV4Dtype;

typedef enum {
    DSV4_CLASS_OTHER = 0,
    DSV4_CLASS_EMBEDDING,
    DSV4_CLASS_HEAD,
    DSV4_CLASS_NORM,
    DSV4_CLASS_HYPER_CONNECTION,
    DSV4_CLASS_ATTENTION,
    DSV4_CLASS_COMPRESSOR_INDEX,
    DSV4_CLASS_ROUTER_HASH,
    DSV4_CLASS_ROUTED_EXPERT,
    DSV4_CLASS_SHARED_EXPERT,
    DSV4_CLASS_MTP,
    DSV4_CLASS_COUNT
} DSV4TensorClass;

typedef struct {
    char message[DSV4_INV_ERROR_CAP];
} DSV4InventoryError;

typedef struct {
    char *name;
    char *path;
    uint64_t file_size;
    uint64_t header_size;
    uint64_t data_size;
    size_t first_tensor;
    size_t tensor_count;
} DSV4ShardInfo;

typedef struct {
    char *name;
    size_t shard;
    DSV4Dtype dtype;
    unsigned bits_per_element;
    int ndim;
    uint64_t shape[DSV4_INV_MAX_DIMS];
    uint64_t data_start;
    uint64_t data_end;
    uint64_t file_offset;
    uint64_t nbytes;
    DSV4TensorClass class_id;
    int layer;
    int expert;
} DSV4TensorInfo;

typedef struct {
    DSV4ShardInfo *shards;
    size_t nshards;
    DSV4TensorInfo *tensors;
    size_t ntensors;
    uint64_t declared_total_size;
    uint64_t total_data_bytes;
    uint64_t bytes_by_dtype[DSV4_DT_COUNT];
    uint64_t bytes_by_class[DSV4_CLASS_COUNT];
    size_t count_by_dtype[DSV4_DT_COUNT];
    size_t count_by_class[DSV4_CLASS_COUNT];
    int profile_validated;
    int32_t *lookup;
    size_t lookup_capacity;
} DSV4Inventory;

int dsv4_inventory_open(DSV4Inventory *inventory, const char *model_dir,
                        DSV4InventoryError *error);
void dsv4_inventory_close(DSV4Inventory *inventory);

const DSV4TensorInfo *dsv4_inventory_find(const DSV4Inventory *inventory,
                                          const char *name);

int dsv4_inventory_validate_flash(DSV4Inventory *inventory,
                                  const DSV4Config *config,
                                  DSV4InventoryError *error);

const char *dsv4_dtype_name(DSV4Dtype dtype);
unsigned dsv4_dtype_bits(DSV4Dtype dtype);
const char *dsv4_tensor_class_name(DSV4TensorClass class_id);

int dsv4_inventory_write_json(const DSV4Inventory *inventory, FILE *stream,
                              int include_tensors);
int dsv4_inventory_write_tsv(const DSV4Inventory *inventory, FILE *stream);

#ifdef __cplusplus
}
#endif

#endif

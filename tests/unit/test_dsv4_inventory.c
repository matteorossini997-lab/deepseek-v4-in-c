/* SPDX-License-Identifier: Apache-2.0 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "dsv4_inventory.h"

typedef struct {
    const char *name;
    const char *dtype;
    int ndim;
    uint64_t shape[4];
} FixtureTensor;

typedef struct {
    char *p;
    size_t n;
    size_t cap;
} Buffer;

static void die(const char *message)
{
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static void check(int condition, const char *message)
{
    if (!condition) die(message);
}

static void buf_init(Buffer *b)
{
    b->cap = 4096;
    b->n = 0;
    b->p = (char *)malloc(b->cap);
    if (!b->p) die("out of memory");
    b->p[0] = '\0';
}

static void buf_add(Buffer *b, const char *text)
{
    size_t len = strlen(text);
    if (b->n + len + 1 > b->cap) {
        while (b->n + len + 1 > b->cap) b->cap *= 2;
        b->p = (char *)realloc(b->p, b->cap);
        if (!b->p) die("out of memory");
    }
    memcpy(b->p + b->n, text, len + 1);
    b->n += len;
}

static void buf_addf(Buffer *b, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    va_list copy;
    va_copy(copy, ap);
    int needed = vsnprintf(NULL, 0, fmt, copy);
    va_end(copy);
    if (needed < 0) die("format failure");
    if (b->n + (size_t)needed + 1 > b->cap) {
        while (b->n + (size_t)needed + 1 > b->cap) b->cap *= 2;
        b->p = (char *)realloc(b->p, b->cap);
        if (!b->p) die("out of memory");
    }
    (void)vsnprintf(b->p + b->n, b->cap - b->n, fmt, ap);
    va_end(ap);
    b->n += (size_t)needed;
}

static unsigned fixture_bits(const char *dtype)
{
    if (!strcmp(dtype, "F4")) return 4;
    if (!strcmp(dtype, "BF16")) return 16;
    if (!strcmp(dtype, "F32") || !strcmp(dtype, "I32")) return 32;
    if (!strcmp(dtype, "F8_E4M3") || !strcmp(dtype, "F8_E8M0") ||
        !strcmp(dtype, "I8") || !strcmp(dtype, "U8")) return 8;
    return 8;
}

static uint64_t fixture_nbytes(const FixtureTensor *t)
{
    uint64_t numel = 1;
    for (int i = 0; i < t->ndim; i++) numel *= t->shape[i];
    uint64_t bits = numel * fixture_bits(t->dtype);
    return (bits + 7) / 8;
}

static void write_u64_le(FILE *f, uint64_t value)
{
    unsigned char bytes[8];
    for (int i = 0; i < 8; i++) bytes[i] = (unsigned char)(value >> (8 * i));
    check(fwrite(bytes, 1, sizeof bytes, f) == sizeof bytes, "write length");
}

static uint64_t write_shard(const char *dir, const char *filename,
                            const FixtureTensor *tensors, size_t count,
                            int offset_mode)
{
    Buffer h;
    buf_init(&h);
    buf_add(&h, "{");
    uint64_t cursor = 0;
    uint64_t logical_total = 0;
    for (size_t i = 0; i < count; i++) {
        uint64_t bytes = fixture_nbytes(&tensors[i]);
        uint64_t start = cursor;
        if (offset_mode == 1 && i == 1) start = cursor - 1; /* overlap */
        if (offset_mode == 2 && i == 1) start = cursor + 1; /* hole */
        uint64_t end = start + bytes;
        if (offset_mode == 3 && i == 0) end++; /* dtype/shape mismatch */
        if (i) buf_add(&h, ",");
        buf_addf(&h, "\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                 tensors[i].name, tensors[i].dtype);
        for (int d = 0; d < tensors[i].ndim; d++)
            buf_addf(&h, "%s%" PRIu64, d ? "," : "", tensors[i].shape[d]);
        buf_addf(&h, "],\"data_offsets\":[%" PRIu64 ",%" PRIu64 "]}",
                 start, end);
        cursor = end;
        logical_total += bytes;
    }
    buf_add(&h, "}");

    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, filename);
    FILE *f = fopen(path, "wb");
    if (!f) die("open shard");
    write_u64_le(f, h.n);
    check(fwrite(h.p, 1, h.n, f) == h.n, "write header");
    for (uint64_t i = 0; i < cursor; i++) fputc(0, f);
    check(fclose(f) == 0, "close shard");
    free(h.p);
    return logical_total;
}

static void write_index(const char *dir, const FixtureTensor *a, size_t na,
                        const FixtureTensor *b, size_t nb, uint64_t total,
                        int mode)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    FILE *f = fopen(path, "wb");
    if (!f) die("open index");
    fprintf(f, "{\"metadata\":{\"total_size\":%" PRIu64 "},\"weight_map\":{", total);
    int first = 1;
    for (size_t i = 0; i < na; i++) {
        const char *shard = "model-00001-of-00002.safetensors";
        if (mode == 2 && i == 0) shard = "model-00002-of-00002.safetensors";
        if (mode == 3 && i == 0) shard = "../escape.safetensors";
        fprintf(f, "%s\"%s\":\"%s\"", first ? "" : ",", a[i].name, shard);
        first = 0;
        if (mode == 4 && i == 0)
            fprintf(f, ",\"%s\":\"%s\"", a[i].name, shard);
    }
    for (size_t i = 0; i < nb; i++) {
        const char *shard = "model-00002-of-00002.safetensors";
        fprintf(f, "%s\"%s\":\"%s\"", first ? "" : ",", b[i].name, shard);
        first = 0;
    }
    if (mode == 1)
        fprintf(f, ",\"ghost.weight\":\"model-00003-of-00003.safetensors\"");
    fprintf(f, "}}\n");
    fclose(f);
}

static char *new_dir(void)
{
    char *path = strdup("/tmp/dsv4-inventory-XXXXXX");
    if (!path || !mkdtemp(path)) die("mkdtemp");
    return path;
}

static const FixtureTensor shard1[] = {
    {"model.embed_tokens.weight", "BF16", 2, {8, 8}},
    {"model.layers.0.mlp.gate.tid2eid", "I32", 2, {4, 6}},
    {"model.layers.3.self_attn.q_a_proj.weight", "F8_E4M3", 2, {128, 128}},
    {"model.layers.3.self_attn.q_a_proj.weight_scale_inv", "F8_E8M0", 2, {1, 1}},
};

static const FixtureTensor shard2[] = {
    {"model.layers.3.mlp.experts.7.gate_proj.weight", "I8", 2, {8, 16}},
    {"model.layers.3.mlp.experts.7.gate_proj.weight_scale_inv", "F8_E8M0", 2, {8, 1}},
    {"model.layers.43.mlp.experts.2.down_proj.weight", "I8", 2, {8, 16}},
    {"model.layers.43.mlp.experts.2.down_proj.weight_scale_inv", "F8_E8M0", 2, {8, 1}},
    {"lm_head.weight", "BF16", 2, {8, 8}},
};

static DSV4Config flash_config(void)
{
    DSV4Config c;
    memset(&c, 0, sizeof c);
    c.n_layers = 43;
    c.n_mtp_layers = 1;
    c.n_routed_experts = 256;
    c.n_activated_experts = 6;
    c.n_shared_experts = 1;
    c.fp8_block_rows = 128;
    c.fp8_block_cols = 128;
    c.fp4_block_size = 32;
    strcpy(c.dtype, "fp8");
    strcpy(c.expert_dtype, "fp4");
    strcpy(c.scale_fmt, "ue8m0");
    return c;
}

static char *make_fixture(int index_mode, int shard_offset_mode,
                          int bad_fp4_scale, int bad_fp8_scale,
                          int unsupported_dtype)
{
    char *dir = new_dir();
    FixtureTensor a[sizeof shard1 / sizeof shard1[0]];
    FixtureTensor b[sizeof shard2 / sizeof shard2[0]];
    memcpy(a, shard1, sizeof a);
    memcpy(b, shard2, sizeof b);
    if (bad_fp8_scale) a[3].shape[1] = 2;
    if (bad_fp4_scale) b[1].shape[1] = 2;
    if (unsupported_dtype) a[0].dtype = "Q42";

    uint64_t total = 0;
    total += write_shard(dir, "model-00001-of-00002.safetensors",
                         a, sizeof a / sizeof a[0], shard_offset_mode);
    total += write_shard(dir, "model-00002-of-00002.safetensors",
                         b, sizeof b / sizeof b[0], 0);
    write_index(dir, a, sizeof a / sizeof a[0],
                b, sizeof b / sizeof b[0], total, index_mode);
    return dir;
}

static void expect_open_failure(char *dir, const char *needle)
{
    DSV4Inventory inv;
    DSV4InventoryError err;
    memset(&inv, 0, sizeof inv);
    memset(&err, 0, sizeof err);
    check(dsv4_inventory_open(&inv, dir, &err) != 0, "expected open failure");
    check(strstr(err.message, needle) != NULL, err.message);
    dsv4_inventory_close(&inv);
    free(dir);
}

static char *make_single_fixture(const char *name)
{
    char *dir = new_dir();
    FixtureTensor tensor = {name, "BF16", 1, {8}};
    uint64_t total = write_shard(dir, "model-00001-of-00002.safetensors",
                                 &tensor, 1, 0);
    write_index(dir, &tensor, 1, NULL, 0, total, 0);
    return dir;
}

static void test_name_guards(void)
{
    char *long_name = (char *)malloc(DSV4_INV_MAX_NAME + sizeof ".weight");
    check(long_name != NULL, "allocate long name");
    memset(long_name, 'a', DSV4_INV_MAX_NAME);
    memcpy(long_name + DSV4_INV_MAX_NAME, ".weight", sizeof ".weight");
    char *dir = make_single_fixture(long_name);
    expect_open_failure(dir, "tensor name too long");
    free(long_name);

    expect_open_failure(
        make_single_fixture("model.layers.999999999999999999999.norm.weight"),
        "malformed layer index");
    expect_open_failure(
        make_single_fixture("model.layers.0.mlp.experts.invalid.gate_proj.weight"),
        "malformed expert index");
}

static void test_valid(void)
{
    char *dir = make_fixture(0, 0, 0, 0, 0);
    DSV4Inventory inv;
    DSV4InventoryError err;
    memset(&inv, 0, sizeof inv);
    memset(&err, 0, sizeof err);
    check(dsv4_inventory_open(&inv, dir, &err) == 0, err.message);
    check(inv.nshards == 2, "expected two shards");
    check(inv.ntensors == 9, "expected nine tensors");
    check(inv.declared_total_size == inv.total_data_bytes, "total size mismatch");
    const DSV4TensorInfo *expert = dsv4_inventory_find(
        &inv, "model.layers.3.mlp.experts.7.gate_proj.weight");
    check(expert != NULL, "expert lookup");
    check(expert->dtype == DSV4_DT_I8, "packed expert dtype");
    check(expert->layer == 3 && expert->expert == 7, "parsed ids");
    check(expert->class_id == DSV4_CLASS_ROUTED_EXPERT, "expert class");

    DSV4Config cfg = flash_config();
    check(dsv4_inventory_validate_flash(&inv, &cfg, &err) == 0, err.message);
    check(inv.profile_validated, "profile flag");
    check(inv.count_by_class[DSV4_CLASS_MTP] == 2, "mtp classification");
    check(inv.count_by_dtype[DSV4_DT_I8] == 2, "packed fp4 count");

    FILE *json = tmpfile();
    check(json != NULL, "tmpfile json");
    check(dsv4_inventory_write_json(&inv, json, 1) == 0, "write json");
    rewind(json);
    char out[8192];
    size_t got = fread(out, 1, sizeof out - 1, json);
    out[got] = '\0';
    check(strstr(out, "\"profile_validated\":true") != NULL, "json profile");
    check(strstr(out, "gate_proj.weight") != NULL, "json tensors");
    fclose(json);

    FILE *tsv = tmpfile();
    check(tsv != NULL, "tmpfile tsv");
    check(dsv4_inventory_write_tsv(&inv, tsv) == 0, "write tsv");
    rewind(tsv);
    got = fread(out, 1, sizeof out - 1, tsv);
    out[got] = '\0';
    check(strstr(out, "tensor\tshard") != NULL, "tsv header");
    fclose(tsv);

    dsv4_inventory_close(&inv);
    free(dir);
}

static void test_profile_failures(void)
{
    DSV4Inventory inv;
    DSV4InventoryError err;
    DSV4Config cfg = flash_config();
    char *dir = make_fixture(0, 0, 1, 0, 0);
    memset(&inv, 0, sizeof inv);
    memset(&err, 0, sizeof err);
    check(dsv4_inventory_open(&inv, dir, &err) == 0, err.message);
    check(dsv4_inventory_validate_flash(&inv, &cfg, &err) != 0,
          "bad fp4 scale must fail");
    check(strstr(err.message, "FP4 scale geometry") != NULL, err.message);
    dsv4_inventory_close(&inv);
    free(dir);

    dir = make_fixture(0, 0, 0, 1, 0);
    memset(&inv, 0, sizeof inv);
    memset(&err, 0, sizeof err);
    check(dsv4_inventory_open(&inv, dir, &err) == 0, err.message);
    check(dsv4_inventory_validate_flash(&inv, &cfg, &err) != 0,
          "bad fp8 scale must fail");
    check(strstr(err.message, "FP8 scale geometry") != NULL, err.message);
    dsv4_inventory_close(&inv);
    free(dir);
}

int main(void)
{
    test_valid();
    expect_open_failure(make_fixture(1, 0, 0, 0, 0), "missing shard");
    expect_open_failure(make_fixture(2, 0, 0, 0, 0), "mapped to");
    expect_open_failure(make_fixture(3, 0, 0, 0, 0), "unsafe shard name");
    expect_open_failure(make_fixture(4, 0, 0, 0, 0), "duplicate tensor");
    expect_open_failure(make_fixture(0, 1, 0, 0, 0), "overlap");
    expect_open_failure(make_fixture(0, 2, 0, 0, 0), "hole");
    expect_open_failure(make_fixture(0, 3, 0, 0, 0), "byte size");
    expect_open_failure(make_fixture(0, 0, 0, 0, 1), "unsupported dtype");
    test_profile_failures();
    test_name_guards();
    puts("dsv4 inventory tests passed");
    return 0;
}

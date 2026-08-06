/* SPDX-License-Identifier: Apache-2.0 */
/*
 * DeepSeek V4 Safetensors inventory.
 *
 * The parser deliberately reads index and shard metadata only. It does not map or
 * decode checkpoint payloads. The K3 production reader remains unchanged.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "dsv4_inventory.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    const char *p;
    const char *end;
    const char *label;
    DSV4InventoryError *error;
} Scan;

typedef struct {
    char *name;
    char *shard_name;
    size_t shard;
    int seen;
} IndexEntry;

typedef struct {
    uint64_t start;
    uint64_t end;
    const char *name;
} Span;

static void error_reset(DSV4InventoryError *error)
{
    if (error) error->message[0] = '\0';
}

static int error_set(DSV4InventoryError *error, const char *fmt, ...)
{
    if (error && error->message[0] == '\0') {
        va_list ap;
        va_start(ap, fmt);
        (void)vsnprintf(error->message, sizeof error->message, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static char *dup_string(const char *text)
{
    size_t length = strlen(text);
    if (length == SIZE_MAX) return NULL;
    size_t n = length + 1;
    char *copy = (char *)malloc(n);
    if (copy) memcpy(copy, text, n);
    return copy;
}

static void scan_ws(Scan *scan)
{
    while (scan->p < scan->end && isspace((unsigned char)*scan->p)) scan->p++;
}

static int scan_char(Scan *scan, char expected)
{
    scan_ws(scan);
    if (scan->p >= scan->end || *scan->p != expected)
        return error_set(scan->error, "%s: expected '%c'", scan->label, expected);
    scan->p++;
    return 0;
}

static int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    c = (char)tolower((unsigned char)c);
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int append_utf8(char **buffer, size_t *length, size_t *capacity, unsigned cp)
{
    unsigned char bytes[4];
    size_t count;
    if (cp <= 0x7f) {
        bytes[0] = (unsigned char)cp;
        count = 1;
    } else if (cp <= 0x7ff) {
        bytes[0] = (unsigned char)(0xc0 | (cp >> 6));
        bytes[1] = (unsigned char)(0x80 | (cp & 0x3f));
        count = 2;
    } else if (cp <= 0xffff) {
        bytes[0] = (unsigned char)(0xe0 | (cp >> 12));
        bytes[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
        bytes[2] = (unsigned char)(0x80 | (cp & 0x3f));
        count = 3;
    } else {
        bytes[0] = (unsigned char)(0xf0 | (cp >> 18));
        bytes[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3f));
        bytes[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3f));
        bytes[3] = (unsigned char)(0x80 | (cp & 0x3f));
        count = 4;
    }
    if (*length > SIZE_MAX - count - 1) return -1;
    size_t needed = *length + count + 1;
    if (needed > *capacity) {
        size_t next = *capacity ? *capacity : 64;
        while (needed > next) {
            if (next > SIZE_MAX / 2) {
                next = needed;
                break;
            }
            next *= 2;
        }
        char *grown = (char *)realloc(*buffer, next);
        if (!grown) return -1;
        *buffer = grown;
        *capacity = next;
    }
    for (size_t i = 0; i < count; i++) (*buffer)[(*length)++] = (char)bytes[i];
    (*buffer)[*length] = '\0';
    return 0;
}

static int scan_string(Scan *scan, char **out)
{
    scan_ws(scan);
    if (scan->p >= scan->end || *scan->p != '"')
        return error_set(scan->error, "%s: expected JSON string", scan->label);
    scan->p++;
    char *buffer = NULL;
    size_t length = 0, capacity = 0;
    while (scan->p < scan->end && *scan->p != '"') {
        unsigned cp = (unsigned char)*scan->p++;
        if (cp < 0x20) {
            free(buffer);
            return error_set(scan->error, "%s: control byte in JSON string", scan->label);
        }
        if (cp == '\\') {
            if (scan->p >= scan->end) {
                free(buffer);
                return error_set(scan->error, "%s: truncated JSON escape", scan->label);
            }
            char escape = *scan->p++;
            switch (escape) {
            case '"': cp = '"'; break;
            case '\\': cp = '\\'; break;
            case '/': cp = '/'; break;
            case 'b': cp = '\b'; break;
            case 'f': cp = '\f'; break;
            case 'n': cp = '\n'; break;
            case 'r': cp = '\r'; break;
            case 't': cp = '\t'; break;
            case 'u': {
                if (scan->end - scan->p < 4) {
                    free(buffer);
                    return error_set(scan->error, "%s: truncated unicode escape", scan->label);
                }
                unsigned high = 0;
                for (int i = 0; i < 4; i++) {
                    int value = hex_value(scan->p[i]);
                    if (value < 0) {
                        free(buffer);
                        return error_set(scan->error, "%s: invalid unicode escape", scan->label);
                    }
                    high = (high << 4) | (unsigned)value;
                }
                scan->p += 4;
                cp = high;
                if (high >= 0xd800 && high <= 0xdbff) {
                    if (scan->end - scan->p < 6 || scan->p[0] != '\\' || scan->p[1] != 'u') {
                        free(buffer);
                        return error_set(scan->error, "%s: unpaired high surrogate", scan->label);
                    }
                    scan->p += 2;
                    unsigned low = 0;
                    for (int i = 0; i < 4; i++) {
                        int value = hex_value(scan->p[i]);
                        if (value < 0) {
                            free(buffer);
                            return error_set(scan->error, "%s: invalid low surrogate", scan->label);
                        }
                        low = (low << 4) | (unsigned)value;
                    }
                    scan->p += 4;
                    if (low < 0xdc00 || low > 0xdfff) {
                        free(buffer);
                        return error_set(scan->error, "%s: invalid low surrogate", scan->label);
                    }
                    cp = 0x10000u + ((high - 0xd800u) << 10) + (low - 0xdc00u);
                } else if (high >= 0xdc00 && high <= 0xdfff) {
                    free(buffer);
                    return error_set(scan->error, "%s: unpaired low surrogate", scan->label);
                }
                break;
            }
            default:
                free(buffer);
                return error_set(scan->error, "%s: invalid JSON escape", scan->label);
            }
        }
        if (append_utf8(&buffer, &length, &capacity, cp) != 0) {
            free(buffer);
            return error_set(scan->error, "%s: out of memory", scan->label);
        }
    }
    if (scan->p >= scan->end) {
        free(buffer);
        return error_set(scan->error, "%s: unterminated JSON string", scan->label);
    }
    scan->p++;
    if (!buffer) {
        buffer = dup_string("");
        if (!buffer) return error_set(scan->error, "%s: out of memory", scan->label);
    }
    *out = buffer;
    return 0;
}

static int scan_u64(Scan *scan, uint64_t *out)
{
    scan_ws(scan);
    if (scan->p >= scan->end || !isdigit((unsigned char)*scan->p))
        return error_set(scan->error, "%s: expected non-negative integer", scan->label);
    uint64_t value = 0;
    do {
        unsigned digit = (unsigned)(*scan->p - '0');
        if (value > (UINT64_MAX - digit) / 10)
            return error_set(scan->error, "%s: integer overflow", scan->label);
        value = value * 10 + digit;
        scan->p++;
    } while (scan->p < scan->end && isdigit((unsigned char)*scan->p));
    *out = value;
    return 0;
}

static int scan_literal(Scan *scan, const char *literal)
{
    size_t n = strlen(literal);
    if ((size_t)(scan->end - scan->p) < n || memcmp(scan->p, literal, n) != 0)
        return error_set(scan->error, "%s: invalid JSON literal", scan->label);
    scan->p += n;
    return 0;
}

static int skip_value_depth(Scan *scan, int depth);

static int skip_array(Scan *scan, int depth)
{
    if (scan_char(scan, '[') != 0) return -1;
    scan_ws(scan);
    if (scan->p < scan->end && *scan->p == ']') {
        scan->p++;
        return 0;
    }
    for (;;) {
        if (skip_value_depth(scan, depth + 1) != 0) return -1;
        scan_ws(scan);
        if (scan->p < scan->end && *scan->p == ',') {
            scan->p++;
            continue;
        }
        return scan_char(scan, ']');
    }
}

static int skip_object(Scan *scan, int depth)
{
    if (scan_char(scan, '{') != 0) return -1;
    scan_ws(scan);
    if (scan->p < scan->end && *scan->p == '}') {
        scan->p++;
        return 0;
    }
    for (;;) {
        char *key = NULL;
        if (scan_string(scan, &key) != 0) return -1;
        free(key);
        if (scan_char(scan, ':') != 0) return -1;
        if (skip_value_depth(scan, depth + 1) != 0) return -1;
        scan_ws(scan);
        if (scan->p < scan->end && *scan->p == ',') {
            scan->p++;
            continue;
        }
        return scan_char(scan, '}');
    }
}

static int skip_value_depth(Scan *scan, int depth)
{
    if (depth > 64) return error_set(scan->error, "%s: JSON nesting too deep", scan->label);
    scan_ws(scan);
    if (scan->p >= scan->end) return error_set(scan->error, "%s: missing JSON value", scan->label);
    if (*scan->p == '"') {
        char *value = NULL;
        int rc = scan_string(scan, &value);
        free(value);
        return rc;
    }
    if (*scan->p == '{') return skip_object(scan, depth);
    if (*scan->p == '[') return skip_array(scan, depth);
    if (*scan->p == 't') return scan_literal(scan, "true");
    if (*scan->p == 'f') return scan_literal(scan, "false");
    if (*scan->p == 'n') return scan_literal(scan, "null");
    if (*scan->p == '-' || isdigit((unsigned char)*scan->p)) {
        if (*scan->p == '-') scan->p++;
        if (scan->p >= scan->end || !isdigit((unsigned char)*scan->p))
            return error_set(scan->error, "%s: invalid JSON number", scan->label);
        while (scan->p < scan->end &&
               (isdigit((unsigned char)*scan->p) || strchr(".eE+-", *scan->p)))
            scan->p++;
        return 0;
    }
    return error_set(scan->error, "%s: invalid JSON value", scan->label);
}

static int scan_finish(Scan *scan)
{
    scan_ws(scan);
    if (scan->p != scan->end)
        return error_set(scan->error, "%s: trailing bytes after JSON value", scan->label);
    return 0;
}

static int read_whole_file(const char *path, uint64_t limit, char **out, size_t *size,
                           DSV4InventoryError *error)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return error_set(error, "%s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) {
        int saved = errno;
        close(fd);
        return error_set(error, "%s: %s", path, strerror(saved));
    }
    if (st.st_size < 2 || (uint64_t)st.st_size > limit || (uint64_t)st.st_size > SIZE_MAX - 1) {
        close(fd);
        return error_set(error, "%s: implausible file size", path);
    }
    size_t n = (size_t)st.st_size;
    char *buffer = (char *)malloc(n + 1);
    if (!buffer) {
        close(fd);
        return error_set(error, "%s: out of memory", path);
    }
    size_t got = 0;
    while (got < n) {
        ssize_t rc = read(fd, buffer + got, n - got);
        if (rc <= 0) {
            int saved = errno;
            free(buffer);
            close(fd);
            return error_set(error, "%s: short read (%s)", path,
                             rc < 0 ? strerror(saved) : "end of file");
        }
        got += (size_t)rc;
    }
    close(fd);
    buffer[n] = '\0';
    *out = buffer;
    *size = n;
    return 0;
}

static int safe_shard_name(const char *name)
{
    size_t n = strlen(name);
    if (n < sizeof ".safetensors" || name[0] == '.' ||
        strstr(name, "..") != NULL || strchr(name, '/') != NULL ||
        strchr(name, '\\') != NULL)
        return 0;
    return n >= 12 && strcmp(name + n - 12, ".safetensors") == 0;
}

static int join_path(char **out, const char *dir, const char *name,
                     DSV4InventoryError *error)
{
    size_t a = strlen(dir), b = strlen(name);
    if (a > SIZE_MAX - b - 2) return error_set(error, "path length overflow");
    char *path = (char *)malloc(a + b + 2);
    if (!path) return error_set(error, "out of memory building path");
    memcpy(path, dir, a);
    path[a] = '/';
    memcpy(path + a + 1, name, b + 1);
    *out = path;
    return 0;
}

static int index_entry_cmp(const void *left, const void *right)
{
    const IndexEntry *a = (const IndexEntry *)left;
    const IndexEntry *b = (const IndexEntry *)right;
    return strcmp(a->name, b->name);
}

static int shard_cmp(const void *left, const void *right)
{
    const DSV4ShardInfo *a = (const DSV4ShardInfo *)left;
    const DSV4ShardInfo *b = (const DSV4ShardInfo *)right;
    return strcmp(a->name, b->name);
}

static int span_cmp(const void *left, const void *right)
{
    const Span *a = (const Span *)left;
    const Span *b = (const Span *)right;
    if (a->start < b->start) return -1;
    if (a->start > b->start) return 1;
    if (a->end < b->end) return -1;
    if (a->end > b->end) return 1;
    return strcmp(a->name, b->name);
}

static IndexEntry *index_find(IndexEntry *entries, size_t count, const char *name)
{
    size_t low = 0, high = count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = strcmp(name, entries[mid].name);
        if (cmp == 0) return &entries[mid];
        if (cmp < 0) high = mid;
        else low = mid + 1;
    }
    return NULL;
}

static size_t shard_find(const DSV4Inventory *inventory, const char *name)
{
    size_t low = 0, high = inventory->nshards;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = strcmp(name, inventory->shards[mid].name);
        if (cmp == 0) return mid;
        if (cmp < 0) high = mid;
        else low = mid + 1;
    }
    return SIZE_MAX;
}

static int push_index(IndexEntry **entries, size_t *count, size_t *capacity,
                      char *name, char *shard, DSV4InventoryError *error)
{
    if (*count == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 4096;
        if (next < *capacity || next > SIZE_MAX / sizeof **entries)
            return error_set(error, "index entry capacity overflow");
        IndexEntry *grown = (IndexEntry *)realloc(*entries, next * sizeof **entries);
        if (!grown) return error_set(error, "out of memory reading weight_map");
        *entries = grown;
        *capacity = next;
    }
    (*entries)[*count].name = name;
    (*entries)[*count].shard_name = shard;
    (*entries)[*count].shard = SIZE_MAX;
    (*entries)[*count].seen = 0;
    (*count)++;
    return 0;
}

static int parse_metadata(Scan *scan, uint64_t *total_size)
{
    if (scan_char(scan, '{') != 0) return -1;
    int seen_total = 0;
    scan_ws(scan);
    if (scan->p < scan->end && *scan->p == '}') {
        scan->p++;
        return error_set(scan->error, "%s: metadata.total_size is required", scan->label);
    }
    for (;;) {
        char *key = NULL;
        if (scan_string(scan, &key) != 0) return -1;
        if (scan_char(scan, ':') != 0) {
            free(key);
            return -1;
        }
        if (!strcmp(key, "total_size")) {
            if (seen_total) {
                free(key);
                return error_set(scan->error, "%s: duplicate metadata.total_size", scan->label);
            }
            seen_total = 1;
            if (scan_u64(scan, total_size) != 0) {
                free(key);
                return -1;
            }
        } else if (skip_value_depth(scan, 0) != 0) {
            free(key);
            return -1;
        }
        free(key);
        scan_ws(scan);
        if (scan->p < scan->end && *scan->p == ',') {
            scan->p++;
            continue;
        }
        if (scan_char(scan, '}') != 0) return -1;
        break;
    }
    if (!seen_total) return error_set(scan->error, "%s: metadata.total_size is required", scan->label);
    return 0;
}

static int parse_weight_map(Scan *scan, IndexEntry **entries, size_t *count,
                            size_t *capacity)
{
    if (scan_char(scan, '{') != 0) return -1;
    scan_ws(scan);
    if (scan->p < scan->end && *scan->p == '}') {
        scan->p++;
        return error_set(scan->error, "%s: empty weight_map", scan->label);
    }
    for (;;) {
        char *name = NULL, *shard = NULL;
        if (scan_string(scan, &name) != 0) return -1;
        if (scan_char(scan, ':') != 0 || scan_string(scan, &shard) != 0) {
            free(name);
            free(shard);
            return -1;
        }
        if (strlen(name) >= DSV4_INV_MAX_NAME) {
            int rc = error_set(scan->error, "%s: tensor name too long", scan->label);
            free(name);
            free(shard);
            return rc;
        }
        if (!safe_shard_name(shard)) {
            int rc = error_set(scan->error, "%s: unsafe shard name '%s'", scan->label, shard);
            free(name);
            free(shard);
            return rc;
        }
        if (push_index(entries, count, capacity, name, shard, scan->error) != 0) {
            free(name);
            free(shard);
            return -1;
        }
        scan_ws(scan);
        if (scan->p < scan->end && *scan->p == ',') {
            scan->p++;
            continue;
        }
        return scan_char(scan, '}');
    }
}

static void free_index(IndexEntry *entries, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        free(entries[i].name);
        free(entries[i].shard_name);
    }
    free(entries);
}

static int parse_index(const char *path, IndexEntry **entries_out, size_t *count_out,
                       uint64_t *total_size, DSV4InventoryError *error)
{
    char *text = NULL;
    size_t size = 0;
    if (read_whole_file(path, 128u * 1024u * 1024u, &text, &size, error) != 0) return -1;
    Scan scan = {text, text + size, path, error};
    IndexEntry *entries = NULL;
    size_t count = 0, capacity = 0;
    int seen_metadata = 0, seen_map = 0;
    if (scan_char(&scan, '{') != 0) goto fail;
    for (;;) {
        scan_ws(&scan);
        if (scan.p < scan.end && *scan.p == '}') {
            scan.p++;
            break;
        }
        char *key = NULL;
        if (scan_string(&scan, &key) != 0) goto fail;
        if (scan_char(&scan, ':') != 0) {
            free(key);
            goto fail;
        }
        if (!strcmp(key, "metadata")) {
            if (seen_metadata) {
                free(key);
                error_set(error, "%s: duplicate metadata object", path);
                goto fail;
            }
            seen_metadata = 1;
            if (parse_metadata(&scan, total_size) != 0) {
                free(key);
                goto fail;
            }
        } else if (!strcmp(key, "weight_map")) {
            if (seen_map) {
                free(key);
                error_set(error, "%s: duplicate weight_map object", path);
                goto fail;
            }
            seen_map = 1;
            if (parse_weight_map(&scan, &entries, &count, &capacity) != 0) {
                free(key);
                goto fail;
            }
        } else {
            free(key);
            error_set(error, "%s: unsupported root key in index", path);
            goto fail;
        }
        free(key);
        scan_ws(&scan);
        if (scan.p < scan.end && *scan.p == ',') {
            scan.p++;
            continue;
        }
        if (scan_char(&scan, '}') != 0) goto fail;
        break;
    }
    if (scan_finish(&scan) != 0) goto fail;
    if (!seen_metadata || !seen_map) {
        error_set(error, "%s: index requires metadata and weight_map", path);
        goto fail;
    }
    qsort(entries, count, sizeof *entries, index_entry_cmp);
    for (size_t i = 1; i < count; i++) {
        if (!strcmp(entries[i - 1].name, entries[i].name)) {
            error_set(error, "%s: duplicate tensor '%s' in weight_map", path, entries[i].name);
            goto fail;
        }
    }
    free(text);
    *entries_out = entries;
    *count_out = count;
    return 0;

fail:
    free(text);
    free_index(entries, count);
    return -1;
}

const char *dsv4_dtype_name(DSV4Dtype dtype)
{
    static const char *names[DSV4_DT_COUNT] = {
        "UNKNOWN", "BOOL", "F4", "F6_E2M3", "F6_E3M2", "U8", "I8",
        "F8_E5M2", "F8_E4M3", "F8_E8M0", "F8_E4M3FNUZ",
        "F8_E5M2FNUZ", "I16", "U16", "F16", "BF16", "I32", "U32",
        "F32", "C64", "F64", "I64", "U64"
    };
    return dtype >= 0 && dtype < DSV4_DT_COUNT ? names[dtype] : "UNKNOWN";
}

unsigned dsv4_dtype_bits(DSV4Dtype dtype)
{
    static const unsigned bits[DSV4_DT_COUNT] = {
        0, 8, 4, 6, 6, 8, 8, 8, 8, 8, 8, 8, 16, 16, 16, 16,
        32, 32, 32, 64, 64, 64, 64
    };
    return dtype >= 0 && dtype < DSV4_DT_COUNT ? bits[dtype] : 0;
}

static DSV4Dtype parse_dtype(const char *name)
{
    for (int i = 1; i < DSV4_DT_COUNT; i++)
        if (!strcmp(name, dsv4_dtype_name((DSV4Dtype)i))) return (DSV4Dtype)i;
    return DSV4_DT_UNKNOWN;
}

const char *dsv4_tensor_class_name(DSV4TensorClass class_id)
{
    static const char *names[DSV4_CLASS_COUNT] = {
        "other", "embedding", "head", "norm", "hyper_connection",
        "attention", "compressor_index", "router_hash", "routed_expert",
        "shared_expert", "mtp"
    };
    return class_id >= 0 && class_id < DSV4_CLASS_COUNT ? names[class_id] : "other";
}

static int parse_component_id(const char *name, const char *marker, int *out)
{
    const char *p = strstr(name, marker);
    if (!p) {
        *out = -1;
        return 0;
    }
    p += strlen(marker);
    if (!isdigit((unsigned char)*p)) return -1;
    unsigned value = 0;
    while (isdigit((unsigned char)*p)) {
        unsigned digit = (unsigned)(*p - '0');
        if (value > ((unsigned)INT_MAX - digit) / 10) return -1;
        value = value * 10 + digit;
        p++;
    }
    if (*p != '.' && *p != '\0') return -1;
    *out = (int)value;
    return 0;
}

static DSV4TensorClass classify_name(const char *name, int expert)
{
    if (expert >= 0) return DSV4_CLASS_ROUTED_EXPERT;
    if (strstr(name, "shared_expert") || strstr(name, "shared_experts"))
        return DSV4_CLASS_SHARED_EXPERT;
    if (strstr(name, "embed_tokens") || !strncmp(name, "embed.", 6))
        return DSV4_CLASS_EMBEDDING;
    if (strstr(name, "lm_head") || !strncmp(name, "head.", 5))
        return DSV4_CLASS_HEAD;
    if (strstr(name, "compress") || strstr(name, "indexer") ||
        strstr(name, ".index.") || strstr(name, "weights_proj"))
        return DSV4_CLASS_COMPRESSOR_INDEX;
    if (strstr(name, "tid2eid") || strstr(name, "correction_bias") ||
        strstr(name, ".gate.") || strstr(name, ".router."))
        return DSV4_CLASS_ROUTER_HASH;
    if (strstr(name, "self_attn") || strstr(name, ".attn."))
        return DSV4_CLASS_ATTENTION;
    if (strstr(name, "hc_") || strstr(name, ".hc.") || strstr(name, "hyper"))
        return DSV4_CLASS_HYPER_CONNECTION;
    if (strstr(name, "norm")) return DSV4_CLASS_NORM;
    return DSV4_CLASS_OTHER;
}

static int push_tensor(DSV4Inventory *inventory, size_t *capacity,
                       DSV4TensorInfo *tensor, DSV4InventoryError *error)
{
    if (inventory->ntensors == *capacity) {
        size_t next = *capacity ? *capacity * 2 : 4096;
        if (next < *capacity || next > SIZE_MAX / sizeof *inventory->tensors)
            return error_set(error, "tensor capacity overflow");
        DSV4TensorInfo *grown = (DSV4TensorInfo *)realloc(
            inventory->tensors, next * sizeof *inventory->tensors);
        if (!grown) return error_set(error, "out of memory indexing tensors");
        inventory->tensors = grown;
        *capacity = next;
    }
    inventory->tensors[inventory->ntensors++] = *tensor;
    return 0;
}

static int parse_shape(Scan *scan, DSV4TensorInfo *tensor)
{
    if (scan_char(scan, '[') != 0) return -1;
    tensor->ndim = 0;
    scan_ws(scan);
    if (scan->p < scan->end && *scan->p == ']') {
        scan->p++;
        return 0;
    }
    for (;;) {
        if (tensor->ndim == DSV4_INV_MAX_DIMS)
            return error_set(scan->error, "%s: tensor rank exceeds %d",
                             scan->label, DSV4_INV_MAX_DIMS);
        if (scan_u64(scan, &tensor->shape[tensor->ndim]) != 0) return -1;
        tensor->ndim++;
        scan_ws(scan);
        if (scan->p < scan->end && *scan->p == ',') {
            scan->p++;
            continue;
        }
        return scan_char(scan, ']');
    }
}

static int parse_offsets(Scan *scan, DSV4TensorInfo *tensor)
{
    if (scan_char(scan, '[') != 0 ||
        scan_u64(scan, &tensor->data_start) != 0 ||
        scan_char(scan, ',') != 0 ||
        scan_u64(scan, &tensor->data_end) != 0 ||
        scan_char(scan, ']') != 0)
        return -1;
    if (tensor->data_end < tensor->data_start)
        return error_set(scan->error, "%s: tensor has reversed data_offsets", scan->label);
    tensor->nbytes = tensor->data_end - tensor->data_start;
    return 0;
}

static int tensor_expected_bytes(const DSV4TensorInfo *tensor, uint64_t *out,
                                 DSV4InventoryError *error)
{
    uint64_t numel = 1;
    for (int i = 0; i < tensor->ndim; i++) {
        if (tensor->shape[i] == 0) {
            numel = 0;
            break;
        }
        if (numel > UINT64_MAX / tensor->shape[i])
            return error_set(error, "%s: element count overflow", tensor->name);
        numel *= tensor->shape[i];
    }
    if (tensor->bits_per_element == 0)
        return error_set(error, "%s: unsupported dtype", tensor->name);
    if (numel > UINT64_MAX / tensor->bits_per_element)
        return error_set(error, "%s: bit size overflow", tensor->name);
    uint64_t bits = numel * tensor->bits_per_element;
    if (bits % 8 != 0)
        return error_set(error, "%s: sub-byte tensor is not byte aligned", tensor->name);
    *out = bits / 8;
    return 0;
}

static int parse_tensor_object(Scan *scan, DSV4TensorInfo *tensor)
{
    if (scan_char(scan, '{') != 0) return -1;
    int have_dtype = 0, have_shape = 0, have_offsets = 0;
    for (;;) {
        scan_ws(scan);
        if (scan->p < scan->end && *scan->p == '}') {
            scan->p++;
            break;
        }
        char *key = NULL;
        if (scan_string(scan, &key) != 0) return -1;
        if (scan_char(scan, ':') != 0) {
            free(key);
            return -1;
        }
        if (!strcmp(key, "dtype")) {
            char *dtype_name = NULL;
            if (have_dtype || scan_string(scan, &dtype_name) != 0) {
                free(key);
                free(dtype_name);
                if (have_dtype) error_set(scan->error, "%s: duplicate dtype", scan->label);
                return -1;
            }
            have_dtype = 1;
            tensor->dtype = parse_dtype(dtype_name);
            tensor->bits_per_element = dsv4_dtype_bits(tensor->dtype);
            if (tensor->dtype == DSV4_DT_UNKNOWN) {
                int rc = error_set(scan->error, "%s: unsupported dtype '%s'",
                                   tensor->name, dtype_name);
                free(dtype_name);
                free(key);
                return rc;
            }
            free(dtype_name);
        } else if (!strcmp(key, "shape")) {
            if (have_shape) {
                free(key);
                return error_set(scan->error, "%s: duplicate shape", tensor->name);
            }
            have_shape = 1;
            if (parse_shape(scan, tensor) != 0) {
                free(key);
                return -1;
            }
        } else if (!strcmp(key, "data_offsets")) {
            if (have_offsets) {
                free(key);
                return error_set(scan->error, "%s: duplicate data_offsets", tensor->name);
            }
            have_offsets = 1;
            if (parse_offsets(scan, tensor) != 0) {
                free(key);
                return -1;
            }
        } else {
            int rc = error_set(scan->error, "%s: unsupported tensor field '%s'",
                               tensor->name, key);
            free(key);
            return rc;
        }
        free(key);
        scan_ws(scan);
        if (scan->p < scan->end && *scan->p == ',') {
            scan->p++;
            continue;
        }
        if (scan_char(scan, '}') != 0) return -1;
        break;
    }
    if (!have_dtype || !have_shape || !have_offsets)
        return error_set(scan->error, "%s: tensor descriptor is incomplete", tensor->name);
    uint64_t expected = 0;
    if (tensor_expected_bytes(tensor, &expected, scan->error) != 0) return -1;
    if (expected != tensor->nbytes)
        return error_set(scan->error,
                         "%s: byte size mismatch, offsets=%" PRIu64
                         " shape/dtype=%" PRIu64,
                         tensor->name, tensor->nbytes, expected);
    return 0;
}

static int parse_metadata_strings(Scan *scan)
{
    if (scan_char(scan, '{') != 0) return -1;
    scan_ws(scan);
    if (scan->p < scan->end && *scan->p == '}') {
        scan->p++;
        return 0;
    }
    for (;;) {
        char *key = NULL, *value = NULL;
        if (scan_string(scan, &key) != 0 ||
            scan_char(scan, ':') != 0 ||
            scan_string(scan, &value) != 0) {
            free(key);
            free(value);
            return -1;
        }
        free(key);
        free(value);
        scan_ws(scan);
        if (scan->p < scan->end && *scan->p == ',') {
            scan->p++;
            continue;
        }
        return scan_char(scan, '}');
    }
}

static int read_header(int fd, const char *path, char **json, size_t *length,
                       uint64_t *file_size, DSV4InventoryError *error)
{
    struct stat st;
    if (fstat(fd, &st) != 0) return error_set(error, "%s: %s", path, strerror(errno));
    if (st.st_size < 10) return error_set(error, "%s: file is too short", path);
    *file_size = (uint64_t)st.st_size;
    unsigned char bytes[8];
    ssize_t got = pread(fd, bytes, sizeof bytes, 0);
    if (got != (ssize_t)sizeof bytes)
        return error_set(error, "%s: cannot read header length", path);
    uint64_t header = 0;
    for (int i = 7; i >= 0; i--) header = (header << 8) | bytes[i];
    if (header < 2 || header > *file_size - 8 || header > SIZE_MAX - 1 ||
        header > 1024u * 1024u * 1024u)
        return error_set(error, "%s: impossible header length", path);
    char *buffer = (char *)malloc((size_t)header + 1);
    if (!buffer) return error_set(error, "%s: out of memory reading header", path);
    size_t offset = 0;
    while (offset < (size_t)header) {
        got = pread(fd, buffer + offset, (size_t)header - offset, (off_t)(8 + offset));
        if (got <= 0) {
            free(buffer);
            return error_set(error, "%s: short header read", path);
        }
        offset += (size_t)got;
    }
    buffer[header] = '\0';
    *json = buffer;
    *length = (size_t)header;
    return 0;
}

static int scan_shard(DSV4Inventory *inventory, size_t shard_index,
                      IndexEntry *index, size_t index_count, size_t *tensor_capacity,
                      DSV4InventoryError *error)
{
    DSV4ShardInfo *shard = &inventory->shards[shard_index];
    int fd = open(shard->path, O_RDONLY);
    if (fd < 0) return error_set(error, "missing shard %s: %s", shard->name, strerror(errno));
    char *json = NULL;
    size_t header_length = 0;
    uint64_t file_size = 0;
    if (read_header(fd, shard->path, &json, &header_length, &file_size, error) != 0) {
        close(fd);
        return -1;
    }
    close(fd);
    shard->file_size = file_size;
    shard->header_size = 8 + header_length;
    shard->data_size = file_size - shard->header_size;
    shard->first_tensor = inventory->ntensors;

    Scan scan = {json, json + header_length, shard->path, error};
    Span *spans = NULL;
    size_t span_count = 0, span_capacity = 0;
    int seen_metadata = 0;
    if (scan_char(&scan, '{') != 0) goto fail;
    for (;;) {
        scan_ws(&scan);
        if (scan.p < scan.end && *scan.p == '}') {
            scan.p++;
            break;
        }
        char *name = NULL;
        if (scan_string(&scan, &name) != 0 || scan_char(&scan, ':') != 0) {
            free(name);
            goto fail;
        }
        if (!strcmp(name, "__metadata__")) {
            if (seen_metadata) {
                free(name);
                error_set(error, "%s: duplicate __metadata__", shard->path);
                goto fail;
            }
            seen_metadata = 1;
            free(name);
            if (parse_metadata_strings(&scan) != 0) goto fail;
        } else {
            if (strlen(name) >= DSV4_INV_MAX_NAME) {
                error_set(error, "%s: tensor name too long", shard->name);
                free(name);
                goto fail;
            }
            IndexEntry *mapped = index_find(index, index_count, name);
            if (!mapped) {
                error_set(error, "%s: tensor '%s' is absent from weight_map",
                          shard->name, name);
                free(name);
                goto fail;
            }
            if (mapped->shard != shard_index) {
                error_set(error, "%s: tensor '%s' mapped to %s",
                          shard->name, name, inventory->shards[mapped->shard].name);
                free(name);
                goto fail;
            }
            if (mapped->seen) {
                error_set(error, "%s: duplicate tensor '%s' across shard headers",
                          shard->name, name);
                free(name);
                goto fail;
            }
            DSV4TensorInfo tensor;
            memset(&tensor, 0, sizeof tensor);
            tensor.name = name;
            tensor.shard = shard_index;
            if (parse_component_id(name, "layers.", &tensor.layer) != 0) {
                error_set(error, "%s: malformed layer index", name);
                free(name);
                goto fail;
            }
            if (parse_component_id(name, ".experts.", &tensor.expert) != 0) {
                error_set(error, "%s: malformed expert index", name);
                free(name);
                goto fail;
            }
            tensor.class_id = classify_name(name, tensor.expert);
            if (parse_tensor_object(&scan, &tensor) != 0) {
                free(name);
                goto fail;
            }
            if (tensor.data_end > shard->data_size ||
                shard->header_size > UINT64_MAX - tensor.data_start) {
                error_set(error, "%s: tensor '%s' extends beyond shard",
                          shard->name, name);
                free(name);
                goto fail;
            }
            tensor.file_offset = shard->header_size + tensor.data_start;
            if (push_tensor(inventory, tensor_capacity, &tensor, error) != 0) {
                free(name);
                goto fail;
            }
            mapped->seen = 1;
            if (span_count == span_capacity) {
                size_t next = span_capacity ? span_capacity * 2 : 4096;
                if (next < span_capacity || next > SIZE_MAX / sizeof *spans) {
                    error_set(error, "%s: span capacity overflow", shard->name);
                    goto fail;
                }
                Span *grown = (Span *)realloc(spans, next * sizeof *spans);
                if (!grown) {
                    error_set(error, "%s: out of memory validating spans", shard->name);
                    goto fail;
                }
                spans = grown;
                span_capacity = next;
            }
            spans[span_count++] = (Span){tensor.data_start, tensor.data_end, tensor.name};
        }
        scan_ws(&scan);
        if (scan.p < scan.end && *scan.p == ',') {
            scan.p++;
            continue;
        }
        if (scan_char(&scan, '}') != 0) goto fail;
        break;
    }
    if (scan_finish(&scan) != 0) goto fail;
    qsort(spans, span_count, sizeof *spans, span_cmp);
    uint64_t cursor = 0;
    for (size_t i = 0; i < span_count; i++) {
        if (spans[i].start < cursor) {
            error_set(error, "%s: tensor '%s' overlap at byte %" PRIu64,
                      shard->name, spans[i].name, spans[i].start);
            goto fail;
        }
        if (spans[i].start > cursor) {
            error_set(error, "%s: hole before tensor '%s' at byte %" PRIu64,
                      shard->name, spans[i].name, cursor);
            goto fail;
        }
        cursor = spans[i].end;
    }
    if (cursor != shard->data_size) {
        error_set(error, "%s: hole at end of data buffer (%" PRIu64 " vs %" PRIu64 ")",
                  shard->name, cursor, shard->data_size);
        goto fail;
    }
    shard->tensor_count = span_count;
    free(spans);
    free(json);
    return 0;

fail:
    free(spans);
    free(json);
    return -1;
}

static uint64_t fnv1a(const char *text)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int build_lookup(DSV4Inventory *inventory, DSV4InventoryError *error)
{
    if (inventory->ntensors > (size_t)INT32_MAX ||
        inventory->ntensors > SIZE_MAX / 2)
        return error_set(error, "too many tensors for lookup");
    size_t target = inventory->ntensors * 2;
    size_t capacity = 16;
    while (capacity < target) {
        if (capacity > SIZE_MAX / 2)
            return error_set(error, "lookup capacity overflow");
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof *inventory->lookup)
        return error_set(error, "lookup allocation overflow");
    inventory->lookup = (int32_t *)malloc(capacity * sizeof *inventory->lookup);
    if (!inventory->lookup) return error_set(error, "out of memory building lookup");
    inventory->lookup_capacity = capacity;
    for (size_t i = 0; i < capacity; i++) inventory->lookup[i] = -1;
    for (size_t i = 0; i < inventory->ntensors; i++) {
        size_t slot = (size_t)fnv1a(inventory->tensors[i].name) & (capacity - 1);
        while (inventory->lookup[slot] >= 0) {
            int32_t prior = inventory->lookup[slot];
            if (!strcmp(inventory->tensors[(size_t)prior].name, inventory->tensors[i].name))
                return error_set(error, "duplicate tensor '%s'", inventory->tensors[i].name);
            slot = (slot + 1) & (capacity - 1);
        }
        if (i > INT32_MAX) return error_set(error, "too many tensors for lookup");
        inventory->lookup[slot] = (int32_t)i;
    }
    return 0;
}

const DSV4TensorInfo *dsv4_inventory_find(const DSV4Inventory *inventory,
                                          const char *name)
{
    if (!inventory || !name || !inventory->lookup || inventory->lookup_capacity == 0)
        return NULL;
    size_t mask = inventory->lookup_capacity - 1;
    size_t slot = (size_t)fnv1a(name) & mask;
    for (;;) {
        int32_t index = inventory->lookup[slot];
        if (index < 0) return NULL;
        const DSV4TensorInfo *tensor = &inventory->tensors[(size_t)index];
        if (!strcmp(tensor->name, name)) return tensor;
        slot = (slot + 1) & mask;
    }
}

static int recompute_totals(DSV4Inventory *inventory, DSV4InventoryError *error)
{
    memset(inventory->bytes_by_dtype, 0, sizeof inventory->bytes_by_dtype);
    memset(inventory->bytes_by_class, 0, sizeof inventory->bytes_by_class);
    memset(inventory->count_by_dtype, 0, sizeof inventory->count_by_dtype);
    memset(inventory->count_by_class, 0, sizeof inventory->count_by_class);
    inventory->total_data_bytes = 0;
    for (size_t i = 0; i < inventory->ntensors; i++) {
        DSV4TensorInfo *tensor = &inventory->tensors[i];
        if (inventory->total_data_bytes > UINT64_MAX - tensor->nbytes ||
            inventory->bytes_by_dtype[tensor->dtype] > UINT64_MAX - tensor->nbytes ||
            inventory->bytes_by_class[tensor->class_id] > UINT64_MAX - tensor->nbytes)
            return error_set(error, "tensor byte totals overflow");
        inventory->total_data_bytes += tensor->nbytes;
        inventory->bytes_by_dtype[tensor->dtype] += tensor->nbytes;
        inventory->bytes_by_class[tensor->class_id] += tensor->nbytes;
        if (inventory->count_by_dtype[tensor->dtype] == SIZE_MAX ||
            inventory->count_by_class[tensor->class_id] == SIZE_MAX)
            return error_set(error, "tensor count totals overflow");
        inventory->count_by_dtype[tensor->dtype]++;
        inventory->count_by_class[tensor->class_id]++;
    }
    return 0;
}

int dsv4_inventory_open(DSV4Inventory *inventory, const char *model_dir,
                        DSV4InventoryError *error)
{
    if (!inventory || !model_dir) return error_set(error, "invalid inventory arguments");
    memset(inventory, 0, sizeof *inventory);
    error_reset(error);
    char *index_path = NULL;
    if (join_path(&index_path, model_dir, "model.safetensors.index.json", error) != 0)
        return -1;
    IndexEntry *index = NULL;
    size_t index_count = 0;
    if (parse_index(index_path, &index, &index_count,
                    &inventory->declared_total_size, error) != 0) {
        free(index_path);
        return -1;
    }
    free(index_path);

    size_t shard_capacity = 0;
    for (size_t i = 0; i < index_count; i++) {
        int exists = 0;
        for (size_t j = 0; j < inventory->nshards; j++) {
            if (!strcmp(inventory->shards[j].name, index[i].shard_name)) {
                exists = 1;
                break;
            }
        }
        if (exists) continue;
        if (inventory->nshards == shard_capacity) {
            size_t next = shard_capacity ? shard_capacity * 2 : 64;
            if (next < shard_capacity || next > SIZE_MAX / sizeof *inventory->shards) {
                error_set(error, "shard capacity overflow");
                goto fail;
            }
            DSV4ShardInfo *grown = (DSV4ShardInfo *)realloc(
                inventory->shards, next * sizeof *inventory->shards);
            if (!grown) {
                error_set(error, "out of memory collecting shards");
                goto fail;
            }
            inventory->shards = grown;
            shard_capacity = next;
        }
        DSV4ShardInfo *shard = &inventory->shards[inventory->nshards++];
        memset(shard, 0, sizeof *shard);
        shard->name = dup_string(index[i].shard_name);
        if (!shard->name || join_path(&shard->path, model_dir, shard->name, error) != 0)
            goto fail;
    }
    qsort(inventory->shards, inventory->nshards, sizeof *inventory->shards, shard_cmp);
    for (size_t i = 0; i < index_count; i++) {
        index[i].shard = shard_find(inventory, index[i].shard_name);
        if (index[i].shard == SIZE_MAX) {
            error_set(error, "internal shard lookup failure");
            goto fail;
        }
    }

    size_t tensor_capacity = 0;
    for (size_t i = 0; i < inventory->nshards; i++) {
        if (scan_shard(inventory, i, index, index_count, &tensor_capacity, error) != 0)
            goto fail;
    }
    for (size_t i = 0; i < index_count; i++) {
        if (!index[i].seen) {
            error_set(error, "weight_map tensor '%s' is missing from shard %s",
                      index[i].name, index[i].shard_name);
            goto fail;
        }
    }
    if (recompute_totals(inventory, error) != 0) goto fail;
    if (inventory->declared_total_size != inventory->total_data_bytes) {
        error_set(error, "metadata.total_size=%" PRIu64
                  " but shard descriptors contain %" PRIu64 " bytes",
                  inventory->declared_total_size, inventory->total_data_bytes);
        goto fail;
    }
    if (build_lookup(inventory, error) != 0) goto fail;
    free_index(index, index_count);
    return 0;

fail:
    free_index(index, index_count);
    dsv4_inventory_close(inventory);
    return -1;
}

void dsv4_inventory_close(DSV4Inventory *inventory)
{
    if (!inventory) return;
    for (size_t i = 0; i < inventory->ntensors; i++) free(inventory->tensors[i].name);
    for (size_t i = 0; i < inventory->nshards; i++) {
        free(inventory->shards[i].name);
        free(inventory->shards[i].path);
    }
    free(inventory->tensors);
    free(inventory->shards);
    free(inventory->lookup);
    memset(inventory, 0, sizeof *inventory);
}

static int ends_with(const char *text, const char *suffix)
{
    size_t a = strlen(text), b = strlen(suffix);
    return a >= b && !strcmp(text + a - b, suffix);
}

static int scale_weight_name(const char *scale_name, char out[DSV4_INV_MAX_NAME])
{
    const char *suffix = NULL;
    if (ends_with(scale_name, ".weight_scale_inv")) suffix = ".weight_scale_inv";
    else if (ends_with(scale_name, ".scale")) suffix = ".scale";
    else return 0;
    size_t prefix = strlen(scale_name) - strlen(suffix);
    if (prefix > DSV4_INV_MAX_NAME - sizeof ".weight") return -1;
    memcpy(out, scale_name, prefix);
    memcpy(out + prefix, ".weight", sizeof ".weight");
    return 1;
}

static int weight_scale_name(const char *weight_name, const DSV4Inventory *inventory,
                             char out[DSV4_INV_MAX_NAME])
{
    if (!ends_with(weight_name, ".weight")) return 0;
    size_t prefix = strlen(weight_name) - strlen(".weight");
    const char *suffixes[] = {".weight_scale_inv", ".scale"};
    for (size_t i = 0; i < sizeof suffixes / sizeof suffixes[0]; i++) {
        size_t suffix_length = strlen(suffixes[i]);
        if (prefix > DSV4_INV_MAX_NAME - suffix_length - 1) return -1;
        memcpy(out, weight_name, prefix);
        memcpy(out + prefix, suffixes[i], suffix_length + 1);
        if (dsv4_inventory_find(inventory, out)) return 1;
    }
    return 0;
}

static int dtype_is_fp8_weight(DSV4Dtype dtype)
{
    return dtype == DSV4_DT_F8_E4M3 || dtype == DSV4_DT_F8_E4M3FNUZ;
}

static int tensor_is_fp4_weight(const DSV4TensorInfo *tensor)
{
    if (!ends_with(tensor->name, ".weight")) return 0;
    if (tensor->class_id != DSV4_CLASS_ROUTED_EXPERT &&
        tensor->class_id != DSV4_CLASS_SHARED_EXPERT &&
        tensor->class_id != DSV4_CLASS_MTP)
        return 0;
    return tensor->dtype == DSV4_DT_F4 || tensor->dtype == DSV4_DT_I8;
}

static int is_scale_dtype(DSV4Dtype dtype)
{
    return dtype == DSV4_DT_F8_E8M0;
}

static int validate_fp4_pair(const DSV4TensorInfo *weight,
                             const DSV4TensorInfo *scale,
                             const DSV4Config *config,
                             DSV4InventoryError *error)
{
    if (!scale || !is_scale_dtype(scale->dtype))
        return error_set(error, "%s: FP4 weight lacks F8_E8M0 scale", weight->name);
    if (weight->ndim < 2 || scale->ndim != weight->ndim)
        return error_set(error, "%s: FP4 scale geometry rank mismatch", weight->name);
    for (int i = 0; i < weight->ndim - 1; i++) {
        if (scale->shape[i] != weight->shape[i])
            return error_set(error, "%s: FP4 scale geometry mismatch", weight->name);
    }
    uint64_t k = weight->shape[weight->ndim - 1];
    if (weight->dtype == DSV4_DT_I8) {
        if (k > UINT64_MAX / 2)
            return error_set(error, "%s: packed FP4 logical width overflow", weight->name);
        k *= 2; /* official HF checkpoint stores two FP4 values in each I8 byte */
    }
    uint64_t expected = (k + (uint64_t)config->fp4_block_size - 1) /
                        (uint64_t)config->fp4_block_size;
    if (k % (uint64_t)config->fp4_block_size != 0 ||
        scale->shape[scale->ndim - 1] != expected)
        return error_set(error, "%s: FP4 scale geometry mismatch", weight->name);
    return 0;
}

static int validate_fp8_pair(const DSV4TensorInfo *weight,
                             const DSV4TensorInfo *scale,
                             const DSV4Config *config,
                             DSV4InventoryError *error)
{
    if (!scale || !is_scale_dtype(scale->dtype))
        return error_set(error, "%s: FP8 weight lacks F8_E8M0 scale", weight->name);
    if (weight->ndim < 2 || scale->ndim != weight->ndim)
        return error_set(error, "%s: FP8 scale geometry rank mismatch", weight->name);
    for (int i = 0; i < weight->ndim - 2; i++) {
        if (scale->shape[i] != weight->shape[i])
            return error_set(error, "%s: FP8 scale geometry mismatch", weight->name);
    }
    uint64_t n = weight->shape[weight->ndim - 2];
    uint64_t k = weight->shape[weight->ndim - 1];
    uint64_t rows = (n + (uint64_t)config->fp8_block_rows - 1) /
                    (uint64_t)config->fp8_block_rows;
    uint64_t cols = (k + (uint64_t)config->fp8_block_cols - 1) /
                    (uint64_t)config->fp8_block_cols;
    if (scale->shape[scale->ndim - 2] != rows ||
        scale->shape[scale->ndim - 1] != cols)
        return error_set(error, "%s: FP8 scale geometry mismatch", weight->name);
    return 0;
}

int dsv4_inventory_validate_flash(DSV4Inventory *inventory,
                                  const DSV4Config *config,
                                  DSV4InventoryError *error)
{
    if (!inventory || !config) return error_set(error, "invalid profile arguments");
    error_reset(error);
    inventory->profile_validated = 0;
    if (config->n_layers <= 0 || config->n_mtp_layers != 1 ||
        config->n_routed_experts != 256 || config->n_activated_experts != 6 ||
        config->fp8_block_rows != 128 || config->fp8_block_cols != 128 ||
        config->fp4_block_size != 32 || strcmp(config->dtype, "fp8") ||
        strcmp(config->expert_dtype, "fp4") || strcmp(config->scale_fmt, "ue8m0"))
        return error_set(error, "configuration is not the supported Flash quantization profile");

    for (size_t i = 0; i < inventory->ntensors; i++) {
        DSV4TensorInfo *tensor = &inventory->tensors[i];
        if (tensor->layer >= config->n_layers) {
            if (tensor->layer >= config->n_layers + config->n_mtp_layers)
                return error_set(error, "%s: layer index is outside base+MTP range",
                                 tensor->name);
            tensor->class_id = DSV4_CLASS_MTP;
        }
        if (tensor->expert >= config->n_routed_experts)
            return error_set(error, "%s: expert index is outside configured range",
                             tensor->name);

        if (strstr(tensor->name, "tid2eid")) {
            if (tensor->ndim < 1 ||
                tensor->shape[tensor->ndim - 1] !=
                    (uint64_t)config->n_activated_experts)
                return error_set(error, "%s: learned top-6 routing geometry mismatch",
                                 tensor->name);
            if (tensor->dtype != DSV4_DT_I8 && tensor->dtype != DSV4_DT_U8 &&
                tensor->dtype != DSV4_DT_I32 && tensor->dtype != DSV4_DT_I64)
                return error_set(error, "%s: routing index dtype is not integer",
                                 tensor->name);
        }

        if (tensor_is_fp4_weight(tensor) || dtype_is_fp8_weight(tensor->dtype)) {
            char scale_name[DSV4_INV_MAX_NAME];
            int found = weight_scale_name(tensor->name, inventory, scale_name);
            if (found < 0) return error_set(error, "%s: scale name is too long", tensor->name);
            const DSV4TensorInfo *scale =
                found > 0 ? dsv4_inventory_find(inventory, scale_name) : NULL;
            if (tensor_is_fp4_weight(tensor)) {
                if (validate_fp4_pair(tensor, scale, config, error) != 0) return -1;
            } else if (validate_fp8_pair(tensor, scale, config, error) != 0) {
                return -1;
            }
        } else {
            char weight_name[DSV4_INV_MAX_NAME];
            int is_scale = scale_weight_name(tensor->name, weight_name);
            if (is_scale < 0) return error_set(error, "%s: weight name is too long", tensor->name);
            if (is_scale > 0) {
                const DSV4TensorInfo *weight = dsv4_inventory_find(inventory, weight_name);
                if (!weight || (!tensor_is_fp4_weight(weight) &&
                                !dtype_is_fp8_weight(weight->dtype)))
                    return error_set(error, "%s: orphan quantization scale", tensor->name);
                if (!is_scale_dtype(tensor->dtype))
                    return error_set(error, "%s: quantization scale is not F8_E8M0",
                                     tensor->name);
            }
        }
    }
    if (recompute_totals(inventory, error) != 0) return -1;
    inventory->profile_validated = 1;
    return 0;
}

static void json_string(FILE *stream, const char *text)
{
    fputc('"', stream);
    for (; *text; text++) {
        unsigned char c = (unsigned char)*text;
        switch (c) {
        case '"': fputs("\\\"", stream); break;
        case '\\': fputs("\\\\", stream); break;
        case '\b': fputs("\\b", stream); break;
        case '\f': fputs("\\f", stream); break;
        case '\n': fputs("\\n", stream); break;
        case '\r': fputs("\\r", stream); break;
        case '\t': fputs("\\t", stream); break;
        default:
            if (c < 0x20) fprintf(stream, "\\u%04x", c);
            else fputc(c, stream);
        }
    }
    fputc('"', stream);
}

int dsv4_inventory_write_json(const DSV4Inventory *inventory, FILE *stream,
                              int include_tensors)
{
    if (!inventory || !stream) return -1;
    fprintf(stream,
            "{\"shards\":%zu,\"tensors\":%zu,\"declared_total_size\":%" PRIu64
            ",\"total_data_bytes\":%" PRIu64 ",\"profile_validated\":%s",
            inventory->nshards, inventory->ntensors,
            inventory->declared_total_size, inventory->total_data_bytes,
            inventory->profile_validated ? "true" : "false");
    fputs(",\"dtypes\":{", stream);
    int first = 1;
    for (int i = 1; i < DSV4_DT_COUNT; i++) {
        if (!inventory->count_by_dtype[i]) continue;
        if (!first) fputc(',', stream);
        first = 0;
        json_string(stream, dsv4_dtype_name((DSV4Dtype)i));
        fprintf(stream, ":{\"count\":%zu,\"bytes\":%" PRIu64 "}",
                inventory->count_by_dtype[i], inventory->bytes_by_dtype[i]);
    }
    fputs("},\"classes\":{", stream);
    first = 1;
    for (int i = 0; i < DSV4_CLASS_COUNT; i++) {
        if (!inventory->count_by_class[i]) continue;
        if (!first) fputc(',', stream);
        first = 0;
        json_string(stream, dsv4_tensor_class_name((DSV4TensorClass)i));
        fprintf(stream, ":{\"count\":%zu,\"bytes\":%" PRIu64 "}",
                inventory->count_by_class[i], inventory->bytes_by_class[i]);
    }
    fputc('}', stream);
    if (include_tensors) {
        fputs(",\"tensor_inventory\":[", stream);
        for (size_t i = 0; i < inventory->ntensors; i++) {
            const DSV4TensorInfo *tensor = &inventory->tensors[i];
            if (i) fputc(',', stream);
            fputs("{\"name\":", stream);
            json_string(stream, tensor->name);
            fputs(",\"shard\":", stream);
            json_string(stream, inventory->shards[tensor->shard].name);
            fputs(",\"dtype\":", stream);
            json_string(stream, dsv4_dtype_name(tensor->dtype));
            fprintf(stream, ",\"shape\":[");
            for (int d = 0; d < tensor->ndim; d++)
                fprintf(stream, "%s%" PRIu64, d ? "," : "", tensor->shape[d]);
            fprintf(stream,
                    "],\"offset\":%" PRIu64 ",\"nbytes\":%" PRIu64
                    ",\"class\":",
                    tensor->file_offset, tensor->nbytes);
            json_string(stream, dsv4_tensor_class_name(tensor->class_id));
            fprintf(stream, ",\"layer\":%d,\"expert\":%d}", tensor->layer,
                    tensor->expert);
        }
        fputc(']', stream);
    }
    fputs("}\n", stream);
    return ferror(stream) ? -1 : 0;
}

int dsv4_inventory_write_tsv(const DSV4Inventory *inventory, FILE *stream)
{
    if (!inventory || !stream) return -1;
    fputs("tensor\tshard\tdtype\tshape\toffset\tnbytes\tclass\tlayer\texpert\n",
          stream);
    for (size_t i = 0; i < inventory->ntensors; i++) {
        const DSV4TensorInfo *tensor = &inventory->tensors[i];
        fprintf(stream, "%s\t%s\t%s\t", tensor->name,
                inventory->shards[tensor->shard].name,
                dsv4_dtype_name(tensor->dtype));
        if (tensor->ndim == 0) {
            fputs("[]", stream);
        } else {
            for (int d = 0; d < tensor->ndim; d++)
                fprintf(stream, "%s%" PRIu64, d ? "x" : "", tensor->shape[d]);
        }
        fprintf(stream, "\t%" PRIu64 "\t%" PRIu64 "\t%s\t%d\t%d\n",
                tensor->file_offset, tensor->nbytes,
                dsv4_tensor_class_name(tensor->class_id),
                tensor->layer, tensor->expert);
    }
    return ferror(stream) ? -1 : 0;
}

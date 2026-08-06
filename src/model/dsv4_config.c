/* SPDX-License-Identifier: Apache-2.0 */
/*
 * DeepSeek V4 Flash configuration normalizer.
 *
 * Source contract:
 *   deepseek-ai/DeepSeek-V4-Flash
 *   60d8d70770c6776ff598c94bb586a859a38244f1
 *
 * The root Transformers config and inference/config.json omit different fields.
 * Values filled from explicit constants in inference/model.py are marked in
 * contract_mask. Nothing else is defaulted.
 */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "dsv4_config.h"

typedef struct {
    jval *root;
    DSV4Config *cfg;
    DSV4ConfigError *err;
    int has_hf;
    int has_inference;
} ConfigCtx;

static void error_reset(DSV4ConfigError *err)
{
    if (!err) return;
    err->count = 0;
    err->message[0] = '\0';
}

static void error_add(DSV4ConfigError *err, const char *fmt, ...)
{
    if (!err) return;
    size_t used = strlen(err->message);
    if (used >= sizeof err->message - 1) {
        err->count++;
        return;
    }
    if (used) {
        int n = snprintf(err->message + used, sizeof err->message - used, "; ");
        if (n < 0) return;
        used += (size_t)n;
        if (used >= sizeof err->message - 1) {
            err->count++;
            return;
        }
    }
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(err->message + used, sizeof err->message - used, fmt, ap);
    va_end(ap);
    err->count++;
}

static void json_free_value(jval *value)
{
    if (!value) return;
    if (value->t == J_OBJ) {
        for (int i = 0; i < value->len; i++) {
            free(value->keys[i]);
            json_free_value(value->kids[i]);
        }
        free(value->keys);
        free(value->kids);
    } else if (value->t == J_ARR) {
        for (int i = 0; i < value->len; i++) json_free_value(value->kids[i]);
        free(value->kids);
    } else if (value->t == J_STR) {
        free(value->str);
    }
    free(value);
}

static char *read_file(const char *path, DSV4ConfigError *err)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        error_add(err, "%s: %s", path, strerror(errno));
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        error_add(err, "%s: seek failed", path);
        fclose(file);
        return NULL;
    }
    long size = ftell(file);
    if (size < 2 || size > (1L << 24)) {
        error_add(err, "%s: implausible config size %ld", path, size);
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        error_add(err, "%s: rewind failed", path);
        fclose(file);
        return NULL;
    }
    char *text = (char *)malloc((size_t)size + 1);
    if (!text) {
        error_add(err, "%s: out of memory", path);
        fclose(file);
        return NULL;
    }
    size_t got = fread(text, 1, (size_t)size, file);
    fclose(file);
    if (got != (size_t)size) {
        error_add(err, "%s: short read", path);
        free(text);
        return NULL;
    }
    text[got] = '\0';
    return text;
}

static jval *nested(jval *root, const char *object, const char *key)
{
    jval *parent = json_get(root, object);
    return json_get(parent, key);
}

static int number_value(jval *value, const char *name, double *out, DSV4ConfigError *err)
{
    if (!value) return 0;
    if (value->t != J_NUM || !isfinite(value->num)) {
        error_add(err, "%s must be a finite number", name);
        return -1;
    }
    *out = value->num;
    return 1;
}

static int int_value(jval *value, const char *name, int64_t *out, DSV4ConfigError *err)
{
    double number = 0.0;
    int state = number_value(value, name, &number, err);
    if (state <= 0) return state;
    double whole = 0.0;
    if (modf(number, &whole) != 0.0 || whole < (double)INT64_MIN ||
        whole > (double)INT64_MAX) {
        error_add(err, "%s must be an integer", name);
        return -1;
    }
    *out = (int64_t)whole;
    return 1;
}

static int string_value(jval *value, const char *name, const char **out,
                        DSV4ConfigError *err)
{
    if (!value) return 0;
    if (value->t != J_STR || !value->str) {
        error_add(err, "%s must be a string", name);
        return -1;
    }
    *out = value->str;
    return 1;
}

static int bool_value(jval *value, const char *name, int *out, DSV4ConfigError *err)
{
    if (!value) return 0;
    if (value->t == J_BOOL) {
        *out = value->boolean ? 1 : 0;
        return 1;
    }
    error_add(err, "%s must be a boolean", name);
    return -1;
}

static int close_number(double left, double right)
{
    double scale = fmax(1.0, fmax(fabs(left), fabs(right)));
    return fabs(left - right) <= scale * 1e-12;
}

static int alias_int(ConfigCtx *ctx, jval *left, const char *left_name,
                     jval *right, const char *right_name, int *out)
{
    int64_t a = 0, b = 0;
    int sa = int_value(left, left_name, &a, ctx->err);
    int sb = int_value(right, right_name, &b, ctx->err);
    if (sa == 0 && sb == 0) {
        error_add(ctx->err, "missing required field %s/%s", left_name, right_name);
        return 0;
    }
    if (sa < 0 || sb < 0) return 0;
    if (sa > 0 && sb > 0 && a != b) {
        error_add(ctx->err, "conflict: %s=%lld but %s=%lld", left_name,
                  (long long)a, right_name, (long long)b);
        return 0;
    }
    int64_t chosen = sa > 0 ? a : b;
    if (chosen < INT_MIN || chosen > INT_MAX) {
        error_add(ctx->err, "%s/%s is outside int range", left_name, right_name);
        return 0;
    }
    *out = (int)chosen;
    return 1;
}

static int alias_i64(ConfigCtx *ctx, jval *left, const char *left_name,
                     jval *right, const char *right_name, int64_t *out)
{
    int64_t a = 0, b = 0;
    int sa = int_value(left, left_name, &a, ctx->err);
    int sb = int_value(right, right_name, &b, ctx->err);
    if (sa == 0 && sb == 0) {
        error_add(ctx->err, "missing required field %s/%s", left_name, right_name);
        return 0;
    }
    if (sa < 0 || sb < 0) return 0;
    if (sa > 0 && sb > 0 && a != b) {
        error_add(ctx->err, "conflict: %s=%lld but %s=%lld", left_name,
                  (long long)a, right_name, (long long)b);
        return 0;
    }
    *out = sa > 0 ? a : b;
    return 1;
}

static int alias_double(ConfigCtx *ctx, jval *left, const char *left_name,
                        jval *right, const char *right_name, double *out)
{
    double a = 0.0, b = 0.0;
    int sa = number_value(left, left_name, &a, ctx->err);
    int sb = number_value(right, right_name, &b, ctx->err);
    if (sa == 0 && sb == 0) {
        error_add(ctx->err, "missing required field %s/%s", left_name, right_name);
        return 0;
    }
    if (sa < 0 || sb < 0) return 0;
    if (sa > 0 && sb > 0 && !close_number(a, b)) {
        error_add(ctx->err, "conflict: %s=%.17g but %s=%.17g",
                  left_name, a, right_name, b);
        return 0;
    }
    *out = sa > 0 ? a : b;
    return 1;
}

static int copy_string(char *dst, size_t cap, const char *src,
                       const char *name, DSV4ConfigError *err)
{
    if (strlen(src) >= cap) {
        error_add(err, "%s is too long", name);
        return 0;
    }
    memcpy(dst, src, strlen(src) + 1);
    return 1;
}

static int alias_string(ConfigCtx *ctx, jval *left, const char *left_name,
                        jval *right, const char *right_name,
                        char *out, size_t out_cap)
{
    const char *a = NULL, *b = NULL;
    int sa = string_value(left, left_name, &a, ctx->err);
    int sb = string_value(right, right_name, &b, ctx->err);
    if (sa == 0 && sb == 0) {
        error_add(ctx->err, "missing required field %s/%s", left_name, right_name);
        return 0;
    }
    if (sa < 0 || sb < 0) return 0;
    if (sa > 0 && sb > 0 && strcmp(a, b) != 0) {
        error_add(ctx->err, "conflict: %s=%s but %s=%s",
                  left_name, a, right_name, b);
        return 0;
    }
    return copy_string(out, out_cap, sa > 0 ? a : b,
                       sa > 0 ? left_name : right_name, ctx->err);
}

static int required_int(ConfigCtx *ctx, const char *name, int *out)
{
    int64_t value = 0;
    int state = int_value(json_get(ctx->root, name), name, &value, ctx->err);
    if (state == 0) {
        error_add(ctx->err, "missing required field %s", name);
        return 0;
    }
    if (state < 0) return 0;
    if (value < INT_MIN || value > INT_MAX) {
        error_add(ctx->err, "%s is outside int range", name);
        return 0;
    }
    *out = (int)value;
    return 1;
}

static int required_double(ConfigCtx *ctx, const char *name, double *out)
{
    int state = number_value(json_get(ctx->root, name), name, out, ctx->err);
    if (state == 0) {
        error_add(ctx->err, "missing required field %s", name);
        return 0;
    }
    return state > 0;
}

static int required_string(ConfigCtx *ctx, const char *name,
                           char *out, size_t out_cap)
{
    const char *value = NULL;
    int state = string_value(json_get(ctx->root, name), name, &value, ctx->err);
    if (state == 0) {
        error_add(ctx->err, "missing required field %s", name);
        return 0;
    }
    return state > 0 && copy_string(out, out_cap, value, name, ctx->err);
}

static void load_optional_contract_fields(ConfigCtx *ctx)
{
    DSV4Config *cfg = ctx->cfg;
    int64_t ignored = 0;
    int sm = int_value(json_get(ctx->root, "num_nextn_predict_layers"),
                       "num_nextn_predict_layers", &ignored, ctx->err);
    int si = int_value(json_get(ctx->root, "n_mtp_layers"),
                       "n_mtp_layers", &ignored, ctx->err);
    if (sm < 0 || si < 0) {
        /* Errors already recorded. */
    } else if (sm == 0 && si == 0) {
        if (ctx->has_inference) {
            cfg->n_mtp_layers = 1;
            cfg->contract_mask |= DSV4_CONTRACT_MTP_LAYERS;
        } else {
            error_add(ctx->err, "missing required field num_nextn_predict_layers");
        }
    } else {
        int64_t hf_value = 0, inference_value = 0;
        if (sm > 0) (void)int_value(json_get(ctx->root, "num_nextn_predict_layers"),
                                    "num_nextn_predict_layers", &hf_value, ctx->err);
        if (si > 0) (void)int_value(json_get(ctx->root, "n_mtp_layers"),
                                    "n_mtp_layers", &inference_value, ctx->err);
        if (sm > 0 && si > 0 && hf_value != inference_value) {
            error_add(ctx->err,
                      "conflict: num_nextn_predict_layers=%lld but n_mtp_layers=%lld",
                      (long long)hf_value, (long long)inference_value);
        } else {
            int64_t chosen = sm > 0 ? hf_value : inference_value;
            if (chosen < INT_MIN || chosen > INT_MAX) {
                error_add(ctx->err, "MTP layer count outside int range");
            } else {
                cfg->n_mtp_layers = (int)chosen;
            }
        }
    }

    double value = 0.0;
    int state = number_value(json_get(ctx->root, "rms_norm_eps"),
                             "rms_norm_eps", &value, ctx->err);
    if (state == 0)
        state = number_value(json_get(ctx->root, "norm_eps"),
                             "norm_eps", &value, ctx->err);
    if (state > 0) {
        cfg->norm_eps = value;
    } else if (state == 0 && ctx->has_inference) {
        cfg->norm_eps = 1e-6;
        cfg->contract_mask |= DSV4_CONTRACT_NORM_EPS;
    } else if (state == 0) {
        error_add(ctx->err, "missing required field rms_norm_eps");
    }

    state = number_value(json_get(ctx->root, "hc_eps"), "hc_eps", &value, ctx->err);
    if (state > 0) {
        cfg->hc_eps = value;
    } else if (state == 0 && ctx->has_inference) {
        cfg->hc_eps = 1e-6;
        cfg->contract_mask |= DSV4_CONTRACT_HC_EPS;
    } else if (state == 0) {
        error_add(ctx->err, "missing required field hc_eps");
    }
}

static void load_routing_contract(ConfigCtx *ctx)
{
    DSV4Config *cfg = ctx->cfg;
    const char *method = NULL;
    int state = string_value(json_get(ctx->root, "topk_method"),
                             "topk_method", &method, ctx->err);
    if (state > 0) {
        (void)copy_string(cfg->topk_method, sizeof cfg->topk_method,
                          method, "topk_method", ctx->err);
    } else if (state == 0 && ctx->has_inference) {
        (void)copy_string(cfg->topk_method, sizeof cfg->topk_method,
                          "noaux_tc", "model.py topk method", ctx->err);
        cfg->contract_mask |= DSV4_CONTRACT_TOPK_METHOD;
    } else if (state == 0) {
        error_add(ctx->err, "missing required field topk_method");
    }

    state = bool_value(json_get(ctx->root, "norm_topk_prob"),
                       "norm_topk_prob", &cfg->norm_topk_prob, ctx->err);
    if (state == 0 && ctx->has_inference) {
        cfg->norm_topk_prob = 1;
        cfg->contract_mask |= DSV4_CONTRACT_NORM_TOPK;
    } else if (state == 0) {
        error_add(ctx->err, "missing required field norm_topk_prob");
    }

    int64_t kv_heads = 0;
    state = int_value(json_get(ctx->root, "num_key_value_heads"),
                      "num_key_value_heads", &kv_heads, ctx->err);
    if (state > 0) {
        if (kv_heads < INT_MIN || kv_heads > INT_MAX)
            error_add(ctx->err, "num_key_value_heads is outside int range");
        else
            cfg->n_kv_heads = (int)kv_heads;
    } else if (state == 0 && ctx->has_inference) {
        cfg->n_kv_heads = 1;
        cfg->contract_mask |= DSV4_CONTRACT_KV_HEADS;
    } else if (state == 0) {
        error_add(ctx->err, "missing required field num_key_value_heads");
    }
}

static void load_quantization(ConfigCtx *ctx)
{
    DSV4Config *cfg = ctx->cfg;
    jval *quant = json_get(ctx->root, "quantization_config");
    (void)alias_string(ctx, json_get(quant, "quant_method"), "quantization_config.quant_method",
                       json_get(ctx->root, "dtype"), "dtype",
                       cfg->dtype, sizeof cfg->dtype);
    (void)required_string(ctx, "expert_dtype",
                          cfg->expert_dtype, sizeof cfg->expert_dtype);
    (void)alias_string(ctx, json_get(quant, "scale_fmt"),
                       "quantization_config.scale_fmt",
                       json_get(ctx->root, "scale_fmt"), "scale_fmt",
                       cfg->scale_fmt, sizeof cfg->scale_fmt);

    jval *block = json_get(quant, "weight_block_size");
    if (block) {
        if (block->t != J_ARR || block->len != 2) {
            error_add(ctx->err,
                      "quantization_config.weight_block_size must contain two integers");
        } else {
            int64_t rows = 0, cols = 0;
            int sr = int_value(block->kids[0],
                               "quantization_config.weight_block_size[0]",
                               &rows, ctx->err);
            int sc = int_value(block->kids[1],
                               "quantization_config.weight_block_size[1]",
                               &cols, ctx->err);
            if (sr > 0 && sc > 0) {
                if (rows < INT_MIN || rows > INT_MAX || cols < INT_MIN || cols > INT_MAX) {
                    error_add(ctx->err, "FP8 block dimensions are outside int range");
                } else {
                    cfg->fp8_block_rows = (int)rows;
                    cfg->fp8_block_cols = (int)cols;
                }
            }
        }
    } else if (ctx->has_inference) {
        cfg->fp8_block_rows = 128;
        cfg->fp8_block_cols = 128;
        cfg->contract_mask |= DSV4_CONTRACT_FP8_BLOCK;
    } else {
        error_add(ctx->err,
                  "missing required field quantization_config.weight_block_size");
    }

    cfg->fp4_block_size = 32;
    cfg->contract_mask |= DSV4_CONTRACT_FP4_BLOCK;
}

static void load_compression(ConfigCtx *ctx)
{
    DSV4Config *cfg = ctx->cfg;
    jval *ratios = json_get(ctx->root, "compress_ratios");
    if (!ratios) {
        error_add(ctx->err, "missing required field compress_ratios");
        return;
    }
    if (ratios->t != J_ARR) {
        error_add(ctx->err, "compress_ratios must be an array");
        return;
    }
    if (ratios->len > DSV4_MAX_COMPRESS_RATIOS) {
        error_add(ctx->err, "compress_ratios has %d entries, maximum is %d",
                  ratios->len, DSV4_MAX_COMPRESS_RATIOS);
        return;
    }
    cfg->n_compress_ratios = ratios->len;
    for (int i = 0; i < ratios->len; i++) {
        int64_t ratio = 0;
        if (int_value(ratios->kids[i], "compress_ratios entry",
                      &ratio, ctx->err) > 0) {
            if (ratio < INT_MIN || ratio > INT_MAX)
                error_add(ctx->err, "compress_ratios entry is outside int range");
            else
                cfg->compress_ratios[i] = (int)ratio;
        }
    }
}

static void validate_structure(DSV4Config *cfg, DSV4ConfigError *err)
{
#define POSITIVE(field) \
    do { if ((cfg->field) <= 0) error_add(err, #field " must be positive"); } while (0)
    POSITIVE(vocab_size);
    POSITIVE(dim);
    POSITIVE(moe_inter_dim);
    POSITIVE(n_layers);
    POSITIVE(n_heads);
    POSITIVE(n_kv_heads);
    POSITIVE(n_routed_experts);
    POSITIVE(n_shared_experts);
    POSITIVE(n_activated_experts);
    POSITIVE(q_lora_rank);
    POSITIVE(head_dim);
    POSITIVE(rope_head_dim);
    POSITIVE(o_groups);
    POSITIVE(o_lora_rank);
    POSITIVE(window_size);
    POSITIVE(index_n_heads);
    POSITIVE(index_head_dim);
    POSITIVE(index_topk);
    POSITIVE(hc_mult);
    POSITIVE(hc_sinkhorn_iters);
#undef POSITIVE

    if (cfg->n_hash_layers < 0 || cfg->n_hash_layers > cfg->n_layers)
        error_add(err, "n_hash_layers must be in 0..n_layers");
    if (cfg->n_mtp_layers < 0)
        error_add(err, "n_mtp_layers must be non-negative");
    if (cfg->n_activated_experts > cfg->n_routed_experts)
        error_add(err, "n_activated_experts exceeds n_routed_experts");
    if (cfg->rope_head_dim > cfg->head_dim)
        error_add(err, "rope_head_dim exceeds head_dim");
    if (cfg->original_seq_len <= 0 || cfg->max_seq_len < cfg->original_seq_len)
        error_add(err, "invalid original/max sequence lengths");
    if (cfg->rope_factor <= 0.0 || cfg->rope_theta <= 0.0 ||
        cfg->compress_rope_theta <= 0.0)
        error_add(err, "rope factors and theta values must be positive");
    if (cfg->norm_eps <= 0.0 || cfg->hc_eps <= 0.0)
        error_add(err, "normalization epsilons must be positive");
    if (cfg->route_scale <= 0.0 || cfg->swiglu_limit < 0.0)
        error_add(err, "invalid route scale or SwiGLU limit");

    if (strcmp(cfg->score_func, "sqrtsoftplus") != 0)
        error_add(err, "score_func must be sqrtsoftplus");
    if (strcmp(cfg->topk_method, "noaux_tc") != 0)
        error_add(err, "topk_method must be noaux_tc");
    if (!cfg->norm_topk_prob)
        error_add(err, "norm_topk_prob must be true");
    if (strcmp(cfg->dtype, "fp8") != 0)
        error_add(err, "dtype must be fp8");
    if (strcmp(cfg->expert_dtype, "fp4") != 0)
        error_add(err, "expert_dtype must be fp4");
    if (strcmp(cfg->scale_fmt, "ue8m0") != 0)
        error_add(err, "scale_fmt must be ue8m0");
    if (cfg->fp8_block_rows != 128 || cfg->fp8_block_cols != 128)
        error_add(err, "FP8 block must be 128x128");
    if (cfg->fp4_block_size != 32)
        error_add(err, "FP4 block must be 32");

    int expected = cfg->n_layers + cfg->n_mtp_layers;
    if (cfg->n_compress_ratios != expected)
        error_add(err, "compress_ratios has %d entries, expected %d",
                  cfg->n_compress_ratios, expected);
    for (int i = 0; i < cfg->n_compress_ratios; i++) {
        int ratio = cfg->compress_ratios[i];
        if (ratio != 0 && ratio != 4 && ratio != 128) {
            error_add(err, "compress_ratios[%d]=%d is unsupported", i, ratio);
            break;
        }
    }
}

static int load_config_tree(DSV4Config *cfg, jval *root, DSV4ConfigError *err)
{
    memset(cfg, 0, sizeof *cfg);
    error_reset(err);
    if (!root || root->t != J_OBJ) {
        error_add(err, "config root must be a JSON object");
        return 0;
    }

    ConfigCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.root = root;
    ctx.cfg = cfg;
    ctx.err = err;
    ctx.has_hf = json_get(root, "hidden_size") != NULL ||
                 json_get(root, "model_type") != NULL ||
                 json_get(root, "quantization_config") != NULL;
    ctx.has_inference = json_get(root, "dim") != NULL ||
                        json_get(root, "dtype") != NULL;

    if (!ctx.has_hf && !ctx.has_inference) {
        error_add(err, "unrecognized config layout");
        return 0;
    }
    cfg->layout = ctx.has_hf && ctx.has_inference
                      ? DSV4_CONFIG_MIXED
                      : (ctx.has_hf ? DSV4_CONFIG_HF : DSV4_CONFIG_INFERENCE);

    (void)required_int(&ctx, "vocab_size", &cfg->vocab_size);
    (void)alias_int(&ctx, json_get(root, "hidden_size"), "hidden_size",
                    json_get(root, "dim"), "dim", &cfg->dim);
    (void)alias_int(&ctx, json_get(root, "moe_intermediate_size"),
                    "moe_intermediate_size", json_get(root, "moe_inter_dim"),
                    "moe_inter_dim", &cfg->moe_inter_dim);
    (void)alias_int(&ctx, json_get(root, "num_hidden_layers"), "num_hidden_layers",
                    json_get(root, "n_layers"), "n_layers", &cfg->n_layers);
    (void)alias_int(&ctx, json_get(root, "num_hash_layers"), "num_hash_layers",
                    json_get(root, "n_hash_layers"), "n_hash_layers",
                    &cfg->n_hash_layers);
    (void)alias_int(&ctx, json_get(root, "num_attention_heads"),
                    "num_attention_heads", json_get(root, "n_heads"), "n_heads",
                    &cfg->n_heads);
    (void)required_int(&ctx, "n_routed_experts", &cfg->n_routed_experts);
    (void)required_int(&ctx, "n_shared_experts", &cfg->n_shared_experts);
    (void)alias_int(&ctx, json_get(root, "num_experts_per_tok"),
                    "num_experts_per_tok", json_get(root, "n_activated_experts"),
                    "n_activated_experts", &cfg->n_activated_experts);
    (void)alias_string(&ctx, json_get(root, "scoring_func"), "scoring_func",
                       json_get(root, "score_func"), "score_func",
                       cfg->score_func, sizeof cfg->score_func);
    (void)alias_double(&ctx, json_get(root, "routed_scaling_factor"),
                       "routed_scaling_factor", json_get(root, "route_scale"),
                       "route_scale", &cfg->route_scale);
    (void)required_double(&ctx, "swiglu_limit", &cfg->swiglu_limit);
    (void)required_int(&ctx, "q_lora_rank", &cfg->q_lora_rank);
    (void)required_int(&ctx, "head_dim", &cfg->head_dim);
    (void)alias_int(&ctx, json_get(root, "qk_rope_head_dim"),
                    "qk_rope_head_dim", json_get(root, "rope_head_dim"),
                    "rope_head_dim", &cfg->rope_head_dim);
    (void)required_int(&ctx, "o_groups", &cfg->o_groups);
    (void)required_int(&ctx, "o_lora_rank", &cfg->o_lora_rank);
    (void)alias_int(&ctx, json_get(root, "sliding_window"), "sliding_window",
                    json_get(root, "window_size"), "window_size",
                    &cfg->window_size);

    (void)alias_i64(&ctx, nested(root, "rope_scaling",
                                 "original_max_position_embeddings"),
                    "rope_scaling.original_max_position_embeddings",
                    json_get(root, "original_seq_len"), "original_seq_len",
                    &cfg->original_seq_len);
    (void)required_double(&ctx, "rope_theta", &cfg->rope_theta);
    (void)alias_double(&ctx, nested(root, "rope_scaling", "factor"),
                       "rope_scaling.factor", json_get(root, "rope_factor"),
                       "rope_factor", &cfg->rope_factor);
    (void)alias_double(&ctx, nested(root, "rope_scaling", "beta_fast"),
                       "rope_scaling.beta_fast", json_get(root, "beta_fast"),
                       "beta_fast", &cfg->beta_fast);
    (void)alias_double(&ctx, nested(root, "rope_scaling", "beta_slow"),
                       "rope_scaling.beta_slow", json_get(root, "beta_slow"),
                       "beta_slow", &cfg->beta_slow);
    (void)required_double(&ctx, "compress_rope_theta", &cfg->compress_rope_theta);

    int64_t maximum = 0;
    int max_state = int_value(json_get(root, "max_position_embeddings"),
                              "max_position_embeddings", &maximum, err);
    if (max_state > 0) {
        cfg->max_seq_len = maximum;
    } else if (max_state == 0 && ctx.has_inference) {
        double product = (double)cfg->original_seq_len * cfg->rope_factor;
        if (!isfinite(product) || product < 1.0 || product > (double)INT64_MAX ||
            floor(product) != product) {
            error_add(err, "cannot derive max sequence length");
        } else {
            cfg->max_seq_len = (int64_t)product;
            cfg->contract_mask |= DSV4_CONTRACT_MAX_SEQ_LEN;
        }
    } else if (max_state == 0) {
        error_add(err, "missing required field max_position_embeddings");
    }

    (void)required_int(&ctx, "index_n_heads", &cfg->index_n_heads);
    (void)required_int(&ctx, "index_head_dim", &cfg->index_head_dim);
    (void)required_int(&ctx, "index_topk", &cfg->index_topk);
    (void)required_int(&ctx, "hc_mult", &cfg->hc_mult);
    (void)required_int(&ctx, "hc_sinkhorn_iters", &cfg->hc_sinkhorn_iters);

    load_optional_contract_fields(&ctx);
    load_routing_contract(&ctx);
    load_quantization(&ctx);
    load_compression(&ctx);
    validate_structure(cfg, err);
    return !err || err->count == 0;
}

int dsv4_config_load_file(DSV4Config *cfg, const char *path, DSV4ConfigError *err)
{
    if (!cfg || !path) {
        error_reset(err);
        error_add(err, "config and path are required");
        return 0;
    }
    error_reset(err);
    char *text = read_file(path, err);
    if (!text) return 0;
    char *arena = NULL;
    jval *root = json_parse(text, &arena);
    free(text);
    free(arena);
    int ok = load_config_tree(cfg, root, err);
    json_free_value(root);
    return ok;
}

const char *dsv4_config_layout_name(DSV4ConfigLayout layout)
{
    switch (layout) {
    case DSV4_CONFIG_HF: return "hf";
    case DSV4_CONFIG_INFERENCE: return "inference";
    case DSV4_CONFIG_MIXED: return "mixed";
    default: return "unknown";
    }
}

static void expect_int(DSV4ConfigError *err, const char *name, int got, int want)
{
    if (got != want) error_add(err, "%s=%d, expected %d", name, got, want);
}

static void expect_i64(DSV4ConfigError *err, const char *name,
                       int64_t got, int64_t want)
{
    if (got != want)
        error_add(err, "%s=%lld, expected %lld", name,
                  (long long)got, (long long)want);
}

static void expect_double(DSV4ConfigError *err, const char *name,
                          double got, double want)
{
    if (!close_number(got, want))
        error_add(err, "%s=%.17g, expected %.17g", name, got, want);
}

static void expect_string(DSV4ConfigError *err, const char *name,
                          const char *got, const char *want)
{
    if (strcmp(got, want) != 0)
        error_add(err, "%s=%s, expected %s", name, got, want);
}

int dsv4_config_is_flash_profile(const DSV4Config *cfg, DSV4ConfigError *err)
{
    error_reset(err);
    if (!cfg) {
        error_add(err, "config is required");
        return 0;
    }

    expect_int(err, "vocab_size", cfg->vocab_size, 129280);
    expect_int(err, "dim", cfg->dim, 4096);
    expect_int(err, "moe_inter_dim", cfg->moe_inter_dim, 2048);
    expect_int(err, "n_layers", cfg->n_layers, 43);
    expect_int(err, "n_hash_layers", cfg->n_hash_layers, 3);
    expect_int(err, "n_mtp_layers", cfg->n_mtp_layers, 1);
    expect_int(err, "n_heads", cfg->n_heads, 64);
    expect_int(err, "n_kv_heads", cfg->n_kv_heads, 1);
    expect_int(err, "n_routed_experts", cfg->n_routed_experts, 256);
    expect_int(err, "n_shared_experts", cfg->n_shared_experts, 1);
    expect_int(err, "n_activated_experts", cfg->n_activated_experts, 6);
    expect_string(err, "score_func", cfg->score_func, "sqrtsoftplus");
    expect_string(err, "topk_method", cfg->topk_method, "noaux_tc");
    expect_int(err, "norm_topk_prob", cfg->norm_topk_prob, 1);
    expect_double(err, "route_scale", cfg->route_scale, 1.5);
    expect_double(err, "swiglu_limit", cfg->swiglu_limit, 10.0);
    expect_int(err, "q_lora_rank", cfg->q_lora_rank, 1024);
    expect_int(err, "head_dim", cfg->head_dim, 512);
    expect_int(err, "rope_head_dim", cfg->rope_head_dim, 64);
    expect_int(err, "o_groups", cfg->o_groups, 8);
    expect_int(err, "o_lora_rank", cfg->o_lora_rank, 1024);
    expect_int(err, "window_size", cfg->window_size, 128);
    expect_i64(err, "original_seq_len", cfg->original_seq_len, 65536);
    expect_i64(err, "max_seq_len", cfg->max_seq_len, 1048576);
    expect_double(err, "rope_theta", cfg->rope_theta, 10000.0);
    expect_double(err, "rope_factor", cfg->rope_factor, 16.0);
    expect_double(err, "beta_fast", cfg->beta_fast, 32.0);
    expect_double(err, "beta_slow", cfg->beta_slow, 1.0);
    expect_double(err, "compress_rope_theta", cfg->compress_rope_theta, 160000.0);
    expect_int(err, "index_n_heads", cfg->index_n_heads, 64);
    expect_int(err, "index_head_dim", cfg->index_head_dim, 128);
    expect_int(err, "index_topk", cfg->index_topk, 512);
    expect_int(err, "hc_mult", cfg->hc_mult, 4);
    expect_int(err, "hc_sinkhorn_iters", cfg->hc_sinkhorn_iters, 20);
    expect_double(err, "hc_eps", cfg->hc_eps, 1e-6);
    expect_double(err, "norm_eps", cfg->norm_eps, 1e-6);
    expect_string(err, "dtype", cfg->dtype, "fp8");
    expect_string(err, "expert_dtype", cfg->expert_dtype, "fp4");
    expect_string(err, "scale_fmt", cfg->scale_fmt, "ue8m0");
    expect_int(err, "fp8_block_rows", cfg->fp8_block_rows, 128);
    expect_int(err, "fp8_block_cols", cfg->fp8_block_cols, 128);
    expect_int(err, "fp4_block_size", cfg->fp4_block_size, 32);
    expect_int(err, "n_compress_ratios", cfg->n_compress_ratios, 44);

    if (cfg->n_compress_ratios == 44) {
        for (int i = 0; i < 44; i++) {
            int want = i < 2 || i == 43 ? 0 : (i % 2 == 0 ? 4 : 128);
            if (cfg->compress_ratios[i] != want) {
                error_add(err, "compress_ratios[%d]=%d, expected %d",
                          i, cfg->compress_ratios[i], want);
                break;
            }
        }
    }
    return !err || err->count == 0;
}

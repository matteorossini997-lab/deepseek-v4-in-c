#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "dsv4_full_model.h"
#include "dsv4_full_fixture_builders.h"
#include "dsv4_full_runtime_vectors.h"

#define H DSV4_MODEL_HIDDEN
#define HC DSV4_MODEL_HC
#define V DSV4_MODEL_VOCAB
#define K DSV4_MODEL_MOE_TOPK
#define L DSV4_MODEL_LAYERS

static int failures = 0;

static unsigned argmax(const float *values, size_t count)
{
    size_t i, best = 0u;
    for (i = 1u; i < count; ++i) if (values[i] > values[best]) best = i;
    return (unsigned)best;
}

static void close_value(const char *label, size_t pos, size_t slot,
                        float got, float expected, float tolerance)
{
    if (!isfinite(got) || fabsf(got - expected) > tolerance) {
        fprintf(stderr, "FAIL %s pos=%zu slot=%zu got=%.9g expected=%.9g\n",
                label, pos, slot, (double)got, (double)expected);
        ++failures;
    }
}

static void route_pairs(size_t pos, const uint32_t got[K], const float weights[K])
{
    size_t e;
    for (e = 0u; e < K; ++e) {
        uint32_t expected = dsv4_full_expected_mtp_route_indices[pos * K + e];
        float expected_weight = dsv4_full_expected_mtp_route_weights[pos * K + e];
        size_t g;
        int found = 0;
        for (g = 0u; g < K; ++g) if (got[g] == expected) {
            close_value("mtp-route-weight", pos, e, weights[g], expected_weight, 2e-5f);
            found = 1;
            break;
        }
        if (!found) {
            fprintf(stderr, "FAIL MTP route pos=%zu missing=%u\n", pos, expected);
            ++failures;
        }
    }
}

static void test_incremental_and_rollback(void)
{
    DSV4DecoderLayerConfig base_configs[L];
    DSV4DecoderLayerWeights base_weights[L];
    DSV4FullModelConfig config = fixture_full_config(base_configs);
    DSV4FullModelWeights weights = fixture_full_weights(base_weights);
    DSV4FullModelStatus status = DSV4_FULL_OK;
    DSV4FullModel *model = dsv4_full_model_create(&config, &status);
    size_t pos;

    if (model == NULL || status != DSV4_FULL_OK) {
        fprintf(stderr, "FAIL full create status=%d\n", (int)status);
        ++failures;
        return;
    }

    /* Late MTP failure must not commit the staged base token either. */
    {
        const float *saved_bias = weights.mtp.decoder.moe.correction_bias;
        float bs[HC * H], bl[V], ms[HC * H], ml[V];
        uint32_t ri[K]; float rw[K]; DSV4FullModelTrace tr, sentinel;
        size_t i;
        for (i=0u;i<HC*H;++i) { bs[i]=71.0f; ms[i]=72.0f; }
        for (i=0u;i<V;++i) { bl[i]=73.0f; ml[i]=74.0f; }
        for (i=0u;i<K;++i) { ri[i]=UINT32_MAX; rw[i]=75.0f; }
        memset(&tr,0xA5,sizeof(tr)); sentinel=tr;
        weights.mtp.decoder.moe.correction_bias = NULL;
        status = dsv4_full_model_step_f32(model,&weights,dsv4_full_token[0],bs,bl,ms,ml,ri,rw,K,&tr);
        if (status != DSV4_FULL_INVALID_ARGUMENT) { fprintf(stderr,"FAIL rollback status=%d\n",(int)status); ++failures; }
        for (i=0u;i<HC*H;++i) if (bs[i]!=71.0f || ms[i]!=72.0f) { fprintf(stderr,"FAIL rollback streams mutated\n"); ++failures; break; }
        for (i=0u;i<V;++i) if (bl[i]!=73.0f || ml[i]!=74.0f) { fprintf(stderr,"FAIL rollback logits mutated\n"); ++failures; break; }
        if (memcmp(&tr,&sentinel,sizeof(tr))!=0) { fprintf(stderr,"FAIL rollback trace mutated\n"); ++failures; }
        weights.mtp.decoder.moe.correction_bias = saved_bias;
    }

    for (pos = 0u; pos < DSV4_FULL_STEPS; ++pos) {
        float base_streams[HC * H], base_logits[V], mtp_streams[HC * H], mtp_logits[V];
        uint32_t route_indices[K]; float route_weights[K]; DSV4FullModelTrace trace;
        size_t i;
        status = dsv4_full_model_step_f32(model,&weights,dsv4_full_token[pos],base_streams,base_logits,mtp_streams,mtp_logits,route_indices,route_weights,K,&trace);
        if (status != DSV4_FULL_OK) { fprintf(stderr,"FAIL step pos=%zu status=%d\n",pos,(int)status); ++failures; break; }
        for (i=0u;i<V;++i) {
            close_value("base-logits",pos,i,base_logits[i],dsv4_full_expected_base_logits[pos*V+i],2e-3f);
            close_value("mtp-logits",pos,i,mtp_logits[i],dsv4_full_expected_mtp_logits[pos*V+i],2e-3f);
        }
        if (argmax(base_logits,V)!=dsv4_full_expected_base_argmax[pos]) { fprintf(stderr,"FAIL base argmax pos=%zu\n",pos); ++failures; }
        if (argmax(mtp_logits,V)!=dsv4_full_expected_mtp_argmax[pos]) { fprintf(stderr,"FAIL mtp argmax pos=%zu\n",pos); ++failures; }
        route_pairs(pos,route_indices,route_weights);
    }

    /* Shared embedding/head identity is part of the combined contract. */
    {
        DSV4FullModelWeights bad = weights;
        float dummy_embedding[V * H];
        float bs[HC*H], bl[V], ms[HC*H], ml[V]; uint32_t ri[K]; float rw[K]; DSV4FullModelTrace tr;
        memset(dummy_embedding,0,sizeof(dummy_embedding));
        bad.mtp.embedding_weight = dummy_embedding;
        status = dsv4_full_model_step_f32(model,&bad,0u,bs,bl,ms,ml,ri,rw,K,&tr);
        if (status != DSV4_FULL_INVALID_ARGUMENT) { fprintf(stderr,"FAIL shared pointer rejection=%d\n",(int)status); ++failures; }
    }
    dsv4_full_model_destroy(model);
}

static void test_greedy_generation(void)
{
    DSV4DecoderLayerConfig base_configs[L];
    DSV4DecoderLayerWeights base_weights[L];
    DSV4FullModelConfig config = fixture_full_config(base_configs);
    DSV4FullModelWeights weights = fixture_full_weights(base_weights);
    DSV4FullModelStatus status = DSV4_FULL_OK;
    DSV4FullModel *model = dsv4_full_model_create(&config,&status);
    uint64_t prompt[DSV4_FULL_PROMPT_LEN];
    uint64_t output[DSV4_FULL_GENERATED_LEN];
    size_t output_len = 0u, i;
    if (model == NULL || status != DSV4_FULL_OK) { ++failures; return; }
    for (i=0u;i<DSV4_FULL_PROMPT_LEN;++i) prompt[i]=dsv4_full_prompt[i];
    status = dsv4_full_model_generate_greedy(model,&weights,prompt,DSV4_FULL_PROMPT_LEN,DSV4_FULL_MAX_NEW,0,0u,output,DSV4_FULL_GENERATED_LEN,&output_len);
    if (status != DSV4_FULL_OK || output_len != DSV4_FULL_GENERATED_LEN) {
        fprintf(stderr,"FAIL generate status=%d len=%zu\n",(int)status,output_len); ++failures;
    } else for (i=0u;i<output_len;++i) if (output[i]!=dsv4_full_expected_generated[i]) {
        fprintf(stderr,"FAIL generated slot=%zu got=%llu expected=%u\n",i,(unsigned long long)output[i],dsv4_full_expected_generated[i]); ++failures;
    }
    dsv4_full_model_destroy(model);
}

int main(void)
{
    test_incremental_and_rollback();
    test_greedy_generation();
    if (failures != 0) { fprintf(stderr,"%d full-runtime failures\n",failures); return 1; }
    puts("DeepSeek full runtime + greedy generation: OK");
    return 0;
}

#!/usr/bin/env python3
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    model_path = Path('src/cpu/dsv4_model.c')
    model = model_path.read_text(encoding='utf-8')
    model_anchor = '''void dsv4_model_destroy(DSV4Model *model)\n'''
    model_clone = r'''DSV4Model *dsv4_model_clone(const DSV4Model *source,
                             DSV4ModelStatus *out_status)
{
    DSV4Model *copy = NULL;
    DSV4ModelStatus status = DSV4_MODEL_INVALID_ARGUMENT;
    size_t config_bytes;

    if (source == NULL || source->layer_configs == NULL || source->layers == NULL ||
        source->config.num_layers == 0u)
        goto done;
    copy = (DSV4Model *)calloc(1u, sizeof(*copy));
    if (copy == NULL) {
        status = DSV4_MODEL_ALLOCATION_FAILED;
        goto done;
    }
    if (!checked_mul_size(source->config.num_layers,
                          sizeof(*copy->layer_configs), &config_bytes)) {
        status = DSV4_MODEL_OVERFLOW;
        goto fail;
    }
    copy->layer_configs = (DSV4DecoderLayerConfig *)malloc(config_bytes);
    if (copy->layer_configs == NULL) {
        status = DSV4_MODEL_ALLOCATION_FAILED;
        goto fail;
    }
    memcpy(copy->layer_configs, source->layer_configs, config_bytes);
    copy->config = source->config;
    copy->config.layer_configs = copy->layer_configs;
    status = clone_layer_array(source->layers, source->config.num_layers, &copy->layers);
    if (status != DSV4_MODEL_OK) goto fail;
    copy->next_position = source->next_position;
    status = DSV4_MODEL_OK;
    goto done;

fail:
    destroy_layer_array(copy->layers,
                        source != NULL ? source->config.num_layers : 0u);
    free(copy->layer_configs);
    free(copy);
    copy = NULL;
done:
    if (out_status != NULL) *out_status = status;
    return copy;
}

'''
    model = replace_once(model, model_anchor, model_clone + model_anchor,
                         'base model clone insertion')
    model_path.write_text(model, encoding='utf-8')

    mtp_path = Path('src/cpu/dsv4_mtp.c')
    mtp = mtp_path.read_text(encoding='utf-8')
    mtp_anchor = '''void dsv4_mtp_destroy(DSV4MTP *mtp)\n'''
    mtp_clone = r'''DSV4MTP *dsv4_mtp_clone(const DSV4MTP *source,
                         DSV4MTPStatus *out_status)
{
    DSV4MTP *copy = NULL;
    DSV4MTPStatus status = DSV4_MTP_INVALID_ARGUMENT;
    DSV4DecoderLayerStatus decoder_status;

    if (source == NULL || source->decoder == NULL) goto done;
    copy = (DSV4MTP *)calloc(1u, sizeof(*copy));
    if (copy == NULL) {
        status = DSV4_MTP_ALLOCATION_FAILED;
        goto done;
    }
    copy->config = source->config;
    copy->decoder = dsv4_decoder_layer_clone(source->decoder, &decoder_status);
    if (copy->decoder == NULL) {
        status = map_decoder(decoder_status);
        free(copy);
        copy = NULL;
        goto done;
    }
    copy->next_position = source->next_position;
    status = DSV4_MTP_OK;

done:
    if (out_status != NULL) *out_status = status;
    return copy;
}

'''
    mtp = replace_once(mtp, mtp_anchor, mtp_clone + mtp_anchor,
                       'MTP clone insertion')
    mtp_path.write_text(mtp, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

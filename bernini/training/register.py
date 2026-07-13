def register_bernini_renderer_to_veomni():
    from transformers import AutoConfig, AutoModel

    from bernini.models.renderer import BerniniRendererConfig, BerniniRendererModel
    from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY

    try:
        AutoConfig.register("bernini_renderer", BerniniRendererConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(BerniniRendererConfig, BerniniRendererModel)
    except ValueError:
        pass

    if "bernini_renderer" not in MODEL_CONFIG_REGISTRY.valid_keys():
        MODEL_CONFIG_REGISTRY.register("bernini_renderer")(lambda: BerniniRendererConfig)
    if "bernini_renderer" not in MODELING_REGISTRY.valid_keys():
        MODELING_REGISTRY.register("bernini_renderer")(lambda architecture=None: BerniniRendererModel)

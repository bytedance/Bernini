# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

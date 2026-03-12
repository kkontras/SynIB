from synib.models.vlm.qwen_base_models import (
    _QwenVL_SynergyLegacyImpl,
    _QwenVL_PromptFrozenCLSImpl,
    _QwenVL_PromptFrozenCLSVisualEmbImpl,
    _QwenVL_PromptUnimodalImageImpl,
    _QwenVL_PromptUnimodalTextImpl,
)


class QwenVL_Synergy_FrozenCLS(_QwenVL_PromptFrozenCLSImpl):
    pass


class QwenVL_Synergy_FrozenCLS_VisualEmb(_QwenVL_PromptFrozenCLSVisualEmbImpl):
    pass


class QwenVL_Unimodal_Image(_QwenVL_PromptUnimodalImageImpl):
    pass


class QwenVL_Unimodal_Text(_QwenVL_PromptUnimodalTextImpl):
    pass


class QwenVL_Synergy(_QwenVL_SynergyLegacyImpl):
    pass

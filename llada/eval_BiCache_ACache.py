"""HF BiCache-style matched-budget layer-selection baseline."""
from eval_CacheBlend_ACache import LLaDACacheBlendAnchorEvalHarness
from bicache_layer_policy import BiCacheEvalMixin
from acache_eval_shared import prepare_cli_args_for_custom_fewshot
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("llada_bicache_acache")
class BiCacheEval(BiCacheEvalMixin, LLaDACacheBlendAnchorEvalHarness):
    bicache_family = "llada"


if __name__ == "__main__":
    print("Active policy: BiCache-style / matched-budget layer selection", flush=True)
    cli_evaluate(prepare_cli_args_for_custom_fewshot())

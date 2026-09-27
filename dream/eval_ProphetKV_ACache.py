"""Inherit the live CacheBlend prompt/eval/cache protocol, replace selection only."""
from eval_CacheBlend_ACache import DreamCacheBlendAnchorEvalHarness
from generate_ProphetKV_ACache import generate_with_prophetkv_anchor_attention
from acache_eval_shared import prepare_cli_args_for_custom_fewshot
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("dream_prophetkv_acache")
class DreamProphetKVAnchorEvalHarness(DreamCacheBlendAnchorEvalHarness):
    def _generate_with_affix_cache(
        self, input_ids, affix_start, affix_end, generation_start, affix_state
    ):
        return generate_with_prophetkv_anchor_attention(
            self.model, input_ids,
            steps=self.steps, gen_length=self.gen_length, block_length=self.block_length,
            temperature=0.0, remasking=self.remasking, mask_id=self.mask_id,
            threshold=self.threshold, factor=self.factor,
            affix_start=affix_start, affix_end=affix_end,
            generation_start=generation_start if self.affix_type == "suffix" else None,
            anchor_ratio=self.anchor_ratio, selection_mode=self.selection_mode,
            drop_non_anchor=self.drop_non_anchor,
            precomputed_affix_cache=affix_state["precomputed_affix_cache"],
        )


if __name__ == "__main__":
    print("Active selector: ProphetKV (CacheBlend scoring is NOT used).", flush=True)
    cli_evaluate(prepare_cli_args_for_custom_fewshot())

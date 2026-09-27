"""HF, batch-one dKV-Cache+ACache evaluator for Dream.

This is the Dream counterpart to the LLaDA HF dKV evaluator.  It keeps the
standard Dream ACache prompt harness and affix-cache construction intact, but
uses dKV cache refreshes while decoding affixed requests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acache_eval_shared import prepare_cli_args_for_custom_fewshot
from eval_ACache import DreamAnchorEvalHarness
from generate_dKV_ACache import generate_with_dkv_anchor_attention


@register_model("dream_dkv_acache_hf")
class DreamDKVACacheHFEvalHarness(DreamAnchorEvalHarness):
    """Dream ACache harness using dKV refreshes for affixed decoding."""

    def __init__(self, dkv_steps=0, dkv_cache_interval=8, **kwargs):
        super().__init__(**kwargs)
        self.dkv_steps = int(dkv_steps) or int(self.steps)
        self.dkv_cache_interval = int(dkv_cache_interval)
        if self.dkv_steps <= 0 or self.dkv_cache_interval <= 0:
            raise ValueError("dkv_steps and dkv_cache_interval must be positive.")

    def _generate_with_affix_cache(
        self,
        input_ids: torch.Tensor,
        affix_start: int,
        affix_end: int,
        generation_start: int,
        affix_state,
    ):
        return generate_with_dkv_anchor_attention(
            self.model,
            input_ids,
            steps=self.dkv_steps,
            gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=0.0,
            remasking=self.remasking,
            mask_id=self.mask_id,
            affix_start=affix_start,
            affix_end=affix_end,
            generation_start=generation_start if self.affix_type == "suffix" else None,
            anchor_ratio=self.anchor_ratio,
            selection_mode=self.selection_mode,
            drop_non_anchor=self.drop_non_anchor,
            dkv_cache_interval=self.dkv_cache_interval,
            precomputed_affix_cache=affix_state["precomputed_affix_cache"],
        )


if __name__ == "__main__":
    cli_evaluate(prepare_cli_args_for_custom_fewshot())

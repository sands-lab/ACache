"""
Evaluation entry for CacheBlend-style Anchor selection on top of LLaDA ACache.

This is a sidecar ablation script.  It registers ``llada_cacheblend_acache`` and
does not alter the existing ``llada_acache`` implementation.
"""

import accelerate
import sys
from pathlib import Path

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoConfig, AutoTokenizer

from generate import generate_with_dual_cache
from generate_CacheBlend_ACache import generate_with_cacheblend_anchor_attention
from model.modeling_llada import LLaDAModelLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acache_eval_shared import ACacheEvalHarnessMixin, prepare_cli_args_for_custom_fewshot, set_seed


@register_model("llada_cacheblend_acache")
class LLaDACacheBlendAnchorEvalHarness(ACacheEvalHarnessMixin, LM):
    def __init__(
        self,
        model_path="",
        mask_id=126336,
        max_length=4096,
        batch_size=1,
        mc_num=128,
        is_check_greedy=False,
        steps=128,
        gen_length=256,
        block_length=32,
        remasking="low_confidence",
        device="cuda",
        threshold=0.9,
        factor=None,
        save_dir=None,
        show_speed=True,
        anchor_ratio=0.1,
        affix_type="prefix",
        selection_mode="top",
        drop_non_anchor=False,
        cacheblend_score_metric="l2",
        cacheblend_include_values=True,
        seed=0,
        print_fewshot_affix=True,
        fewshot_num_examples=0,
        fewshot_dataset_path="gsm8k",
        fewshot_dataset_name="main",
        fewshot_split="train",
        fewshot_question_key="question",
        fewshot_answer_key="answer",
        **kwargs,
    ):
        super().__init__()
        if "num_anchor" in kwargs:
            raise ValueError("`num_anchor` has been renamed to `anchor_ratio` (range [0, 1]).")

        self.seed = None if seed is None else int(seed)
        if self.seed is not None:
            set_seed(self.seed)
            print(f"Manual seed set in eval_CacheBlend_ACache: {self.seed}")

        if not model_path:
            raise ValueError("`model_path` must be provided.")

        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs["device_map"] = {"": f"{self.accelerator.device}"}

        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            config=config,
            **model_kwargs,
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.mask_id = int(mask_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.is_instruct = "instruct" in model_path.lower()
        self.print_fewshot_affix = print_fewshot_affix

        self.cacheblend_score_metric = str(cacheblend_score_metric).strip().lower()
        if self.cacheblend_score_metric not in {"l2", "l1", "relative_l2"}:
            raise ValueError(
                "cacheblend_score_metric must be one of: l2, l1, relative_l2. "
                f"Got {cacheblend_score_metric!r}."
            )
        self.cacheblend_include_values = cacheblend_include_values

        self._configure_acache_common(
            batch_size=batch_size,
            max_length=max_length,
            mc_num=mc_num,
            is_check_greedy=is_check_greedy,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            save_dir=save_dir,
            show_speed=show_speed,
            anchor_ratio=anchor_ratio,
            affix_type=affix_type,
            selection_mode=selection_mode,
            drop_non_anchor=drop_non_anchor,
            fewshot_num_examples=fewshot_num_examples,
            fewshot_dataset_path=fewshot_dataset_path,
            fewshot_dataset_name=fewshot_dataset_name,
            fewshot_split=fewshot_split,
            fewshot_question_key=fewshot_question_key,
            fewshot_answer_key=fewshot_answer_key,
            model_label="llada_acache",
        )
        print("CacheBlend-style Selector Config:")
        print(f"  cacheblend_score_metric: {self.cacheblend_score_metric}")
        print(f"  cacheblend_include_values: {self.cacheblend_include_values}")

    def _generate_with_affix_cache(
        self,
        input_ids: torch.Tensor,
        affix_start: int,
        affix_end: int,
        generation_start: int,
        affix_state,
    ):
        generation_kwargs = dict(
            steps=self.steps,
            gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=0.0,
            remasking=self.remasking,
            mask_id=self.mask_id,
            threshold=self.threshold,
            factor=self.factor,
            affix_start=affix_start,
            affix_end=affix_end,
            anchor_ratio=self.anchor_ratio,
            selection_mode=self.selection_mode,
            drop_non_anchor=self.drop_non_anchor,
            precomputed_affix_cache=affix_state["precomputed_affix_cache"],
            cacheblend_score_metric=self.cacheblend_score_metric,
            cacheblend_include_values=self.cacheblend_include_values,
        )
        if self.affix_type == "suffix":
            generation_kwargs["generation_start"] = generation_start
        return generate_with_cacheblend_anchor_attention(self.model, input_ids, **generation_kwargs)

    def _generate_without_affix(self, input_ids: torch.Tensor):
        return generate_with_dual_cache(
            self.model,
            input_ids,
            steps=self.steps,
            gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=0.0,
            remasking=self.remasking,
            mask_id=self.mask_id,
            threshold=self.threshold,
            factor=self.factor,
        )


if __name__ == "__main__":
    cli_evaluate(prepare_cli_args_for_custom_fewshot())

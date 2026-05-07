"""
Evaluation entry for CacheBlend-style Anchor selection on top of Dream ACache.

This is a sidecar ablation script.  It registers ``dream_cacheblend_acache`` and
does not alter the existing ``dream_acache`` implementation.
"""

import accelerate
import sys
import types
from pathlib import Path

import torch
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer

from generate_CacheBlend_ACache import generate_with_cacheblend_anchor_attention
from model.generation_utils_block import DreamGenerationMixin as DreamGenerationMixinWithCache
from model.modeling_dream import DreamModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acache_eval_shared import ACacheEvalHarnessMixin, prepare_cli_args_for_custom_fewshot, set_seed


@register_model("dream_cacheblend_acache")
class DreamCacheBlendAnchorEvalHarness(ACacheEvalHarnessMixin, LM):
    def __init__(
        self,
        pretrained="",
        mask_id=None,
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
        removed_knobs = {
            "model_path": "`model_path` has been removed from dream_cacheblend_acache; use `pretrained=...`.",
            "use_cache": (
                "`use_cache` has been removed from dream_cacheblend_acache; cached block decoding is always enabled."
            ),
            "dual_cache": (
                "`dual_cache` has been removed from dream_cacheblend_acache; dual-cache decoding is always enabled."
            ),
            "use_anchor": (
                "`use_anchor` has been removed from dream_cacheblend_acache; use `anchor_ratio=0` for the "
                "frozen-affix ablation or `affix_type=none` for the no-affix baseline."
            ),
            "print_fewshot_affix": "`print_fewshot_affix` has been removed from dream_cacheblend_acache.",
        }
        for knob, message in removed_knobs.items():
            if knob in kwargs:
                raise ValueError(message)

        self.seed = None if seed is None else int(seed)
        if self.seed is not None:
            set_seed(self.seed)
            print(f"Manual seed set in eval_CacheBlend_ACache: {self.seed}")

        if not pretrained:
            raise ValueError("`pretrained` must be provided.")

        accelerator = accelerate.Accelerator()
        self.accelerator = accelerator if accelerator.num_processes > 1 else None

        self.model = DreamModel.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
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

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenizer_mask_id = self.tokenizer.mask_token_id
        generation_mask_id = getattr(getattr(self.model, "generation_config", None), "mask_token_id", None)
        resolved_mask_id = mask_id if mask_id is not None else tokenizer_mask_id
        if resolved_mask_id is None:
            resolved_mask_id = generation_mask_id
        if resolved_mask_id is None:
            raise ValueError("Unable to resolve mask token id. Pass `mask_id` explicitly in --model_args.")
        self.mask_id = int(resolved_mask_id)

        self.is_instruct = "instruct" in pretrained.lower()
        self.model.diffusion_generate = types.MethodType(DreamGenerationMixinWithCache.diffusion_generate, self.model)
        self.model._sample = types.MethodType(DreamGenerationMixinWithCache._sample, self.model)

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
            model_label="dream_acache",
        )
        print("CacheBlend-style Selector Config:")
        print(f"  cacheblend_score_metric: {self.cacheblend_score_metric}")
        print(f"  cacheblend_include_values: {self.cacheblend_include_values}")

    def _uses_chat_template_for_prompts(self) -> bool:
        return True

    def _build_input_ids_with_affix(self, question: str, affix_state):
        if self.affix_type != "none":
            return super()._build_input_ids_with_affix(question, affix_state)

        input_ids = self._tokenize_chat_messages(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
        )
        generation_start = len(input_ids)
        generation_end = generation_start + self.gen_length
        return input_ids, 0, 0, generation_start, generation_end

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
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        generation_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=self.gen_length,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.steps,
            temperature=0.0,
            top_p=None,
            top_k=None,
            alg="confidence_threshold",
            threshold=self.threshold,
            block_length=self.block_length,
            dual_cache=True,
        )

        generated = self.model.diffusion_generate(input_ids, **generation_kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        return sequences, int(self.steps)


if __name__ == "__main__":
    cli_evaluate(prepare_cli_args_for_custom_fewshot())

import accelerate
import json
import os
import random
import re
import time

import torch
from datasets import load_dataset
from lm_eval.__main__ import cli_evaluate, parse_eval_args, setup_parser
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from config import Config
from model.modeling_llada import LLaDAModelLM
from model_runner import ModelRunner
from sequence import Sequence
from utils import set_seed


GSM8K_PROMPT_STYLE = "gsm8k"
MBPP_PROMPT_STYLE = "mbpp"

PREFIX_QUERY_TEMPLATE = "Question: {question}\nLet's think step by step.\nAnswer:"
MBPP_PREFIX_QUERY_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {question} "
    "Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
)

QUESTION_FROM_PROMPT_PATTERN = re.compile(
    r"Question:\s*(.*?)(?:\nAnswer:|\nA:|$)",
    flags=re.DOTALL,
)
MBPP_FROM_PROMPT_PATTERN = re.compile(
    r"You are an expert Python programmer, and here is your task:\s*(.*?)\s+"
    r"Your code should pass these tests:\n\n(.*?)\n\[BEGIN\]\s*$",
    flags=re.DOTALL,
)


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path="",
        mask_id=126336,
        recompute_batch_size=4,
        anchor_selection_batch_size=1,
        gen_length=256,
        block_length=32,
        cache_block_size=256,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.9,
        temperature=0.0,
        remasking="low_confidence",
        device="cuda",
        threshold=0.9,
        save_dir=None,
        show_speed=False,
        variable_fewshot=False,
        seed=0,
        batch_size=0,
        acache=False,
        anchor_ratio=0.05,
        selection_mode="top",
        reference_attention=False,
        profile_timing=False,
        fewshot_num_examples=0,
        fewshot_dataset_path="gsm8k",
        fewshot_dataset_name="main",
        fewshot_split="train",
        fewshot_question_key="question",
        fewshot_answer_key="answer",
        model_type="llada",
    ):
        super().__init__()

        self.model_type = str(model_type).strip().lower()
        if self.model_type not in {"llada", "dream"}:
            raise ValueError(f"Unsupported model_type={model_type!r}; expected 'llada' or 'dream'.")

        self.seed = None if seed is None else int(seed)
        if self.seed is not None:
            set_seed(self.seed)

        recompute_batch_size = int(recompute_batch_size)
        assert recompute_batch_size > 0, "recompute_batch_size must be larger than 0"
        anchor_selection_batch_size = int(anchor_selection_batch_size)
        assert anchor_selection_batch_size > 0, "anchor_selection_batch_size must be larger than 0"

        batch_size = int(batch_size)
        assert batch_size >= 0, "batch_size must be 0 (no limit) or a positive integer"
        max_num_seqs = batch_size

        self.variable_fewshot = coerce_bool(variable_fewshot)

        if isinstance(gen_length, str):
            gen_length = gen_length.strip().strip("'\"")
        if isinstance(gen_length, (list, tuple)):
            raise ValueError(f"gen_length must be a single integer value, got {type(gen_length).__name__}")
        self.gen_length = int(gen_length)

        assert self.gen_length > 0, f"gen_length must be larger than 0. Got {self.gen_length}"
        assert self.gen_length % block_length == 0, (
            f"gen_length must be multiple of block_length. Got {self.gen_length} "
            f"which is not multiple of {block_length}"
        )

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
        if self.model_type == "dream":
            from model.configuration_dream import DreamConfig
            from model.modeling_dream import DreamModel

            hf_config = DreamConfig.from_pretrained(model_path, trust_remote_code=True)
            if int(mask_id) == 126336:
                mask_id = getattr(hf_config, "mask_token_id", mask_id)
            model = DreamModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                config=hf_config,
                **model_kwargs,
            )
        else:
            hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = LLaDAModelLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                config=hf_config,
                **model_kwargs,
            )
        model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            model = self.accelerator.prepare(model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            model = model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.is_instruct = "instruct" in model_path.lower()
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.profile_timing = coerce_bool(profile_timing)

        self.fewshot_num_examples = int(fewshot_num_examples)
        self.fewshot_dataset_path = fewshot_dataset_path
        self.fewshot_dataset_name = fewshot_dataset_name
        self.fewshot_split = fewshot_split
        self.fewshot_question_key = fewshot_question_key
        self.fewshot_answer_key = fewshot_answer_key
        self.prompt_style = self._resolve_prompt_style()
        self.sampled_fewshot_examples = []
        self.prefix_fewshot_messages = self._build_prefix_fewshot_messages()
        self.prefix_affix_token_ids = self._build_prefix_affix_token_ids()
        self.prefix_affix_len = len(self.prefix_affix_token_ids)
        self.use_model_side_prefix_fewshot = (
            self.fewshot_num_examples > 0
            or self.prompt_style == MBPP_PROMPT_STYLE
        )

        self.serve_config = Config(
            hf_config=hf_config,
            mask_id=mask_id,
            recompute_batch_size=recompute_batch_size,
            anchor_selection_batch_size=anchor_selection_batch_size,
            gen_length=self.gen_length,
            block_length=block_length,
            remasking=remasking,
            threshold=threshold,
            cache_block_size=cache_block_size,
            temperature=temperature,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_acache=coerce_bool(acache),
            anchor_ratio=float(anchor_ratio),
            selection_mode=selection_mode,
            use_reference_attention=coerce_bool(reference_attention),
            profile_timing=self.profile_timing,
            model_type=self.model_type,
        )

        self.runner = ModelRunner(model, self.serve_config)

    @property
    def rank(self):
        return self._rank if hasattr(self, "_rank") else 0

    @property
    def world_size(self):
        return self._world_size if hasattr(self, "_world_size") else 1

    def _profile_timing_mark(self):
        if not self.profile_timing:
            return None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _print_profile_timing(self, tokenization_s, sequence_s, generation_s, postprocess_s):
        print("Timing breakdown (seconds; CUDA-synchronized when profile_timing=True):")
        print(f"  eval.tokenization: {tokenization_s:.6f}")
        print(f"  eval.sequence_build: {sequence_s:.6f}")
        print(f"  eval.runner_generate: {generation_s:.6f}")
        print(f"  eval.postprocess_decode: {postprocess_s:.6f} (not included in TPS timer)")

        runner_timing = getattr(self.runner, "last_timing", {}) or {}
        if not runner_timing:
            return

        print("  runner mode:", runner_timing.get("mode", "unknown"))
        print("  note: runner.total is authoritative; setup rows may be nested inside scheduler/admission.")
        timing_order = [
            "total",
            "acache_metadata",
            "shared_prefix_prepare",
            "shared_prefix_forward",
            "anchor_selection_prepare",
            "anchor_selection_forward",
            "anchor_selection_topk",
            "scheduler",
            "cache_prepare",
            "cache_forward",
            "cache_update",
            "decode_prepare",
            "decode_forward",
            "decode_update",
            "kv_cache_peak_used_gb",
            "kv_cache_peak_capacity_gb",
        ]
        count_order = [
            "num_prompts",
            "total_nfe",
            "shared_prefix_forward_calls",
            "anchor_selection_forward_calls",
            "anchor_selection_sequences",
            "anchor_selection_query_tokens",
            "anchor_selection_kv_tokens",
            "cache_forward_calls",
            "cache_forward_nfe",
            "decode_forward_calls",
            "decode_forward_nfe",
            "kv_cache_block_size",
            "kv_cache_block_bytes",
            "kv_cache_slot_bytes",
            "kv_cache_peak_used_blocks",
            "kv_cache_peak_used_slots",
            "kv_cache_peak_used_bytes",
            "kv_cache_peak_capacity_blocks",
            "kv_cache_peak_capacity_slots",
            "kv_cache_peak_capacity_bytes",
        ]
        for key in timing_order:
            value = runner_timing.get(key)
            if isinstance(value, (int, float)):
                print(f"  runner.{key}: {value:.6f}")
        for key in count_order:
            value = runner_timing.get(key)
            if value is not None:
                print(f"  runner.{key}: {value}")
        cache_forward_calls = runner_timing.get("cache_forward_calls")
        cache_forward_nfe = runner_timing.get("cache_forward_nfe")
        cache_forward_s = runner_timing.get("cache_forward")
        cache_prepare_s = runner_timing.get("cache_prepare", 0.0)
        cache_update_s = runner_timing.get("cache_update", 0.0)
        if cache_forward_calls and cache_forward_s is not None:
            cache_total_s = cache_prepare_s + cache_forward_s + cache_update_s
            print(f"  runner.cache_recompute_forward_avg_per_call: {cache_forward_s / cache_forward_calls:.6f}")
            print(f"  runner.cache_recompute_total_avg_per_call: {cache_total_s / cache_forward_calls:.6f}")
        if cache_forward_nfe and cache_forward_s is not None:
            cache_total_s = cache_prepare_s + cache_forward_s + cache_update_s
            print(f"  runner.cache_recompute_forward_avg_per_nfe: {cache_forward_s / cache_forward_nfe:.6f}")
            print(f"  runner.cache_recompute_total_avg_per_nfe: {cache_total_s / cache_forward_nfe:.6f}")

    def _uses_mbpp_fewshot_dataset(self) -> bool:
        dataset_path = str(self.fewshot_dataset_path or "").strip().lower()
        if dataset_path.startswith("lm_eval:"):
            dataset_path = dataset_path[len("lm_eval:"):]
        return dataset_path in {"mbpp", "google-research-datasets/mbpp"} or dataset_path.endswith("/mbpp")

    def _resolved_fewshot_keys(self):
        question_key = self.fewshot_question_key
        answer_key = self.fewshot_answer_key
        if self._uses_mbpp_fewshot_dataset():
            if question_key == "question":
                question_key = "text"
            if answer_key == "answer":
                answer_key = "code"
        return question_key, answer_key

    def _resolve_prompt_style(self) -> str:
        question_key, answer_key = self._resolved_fewshot_keys()
        if self._uses_mbpp_fewshot_dataset():
            return MBPP_PROMPT_STYLE
        if question_key == "text" and answer_key == "code":
            return MBPP_PROMPT_STYLE
        return GSM8K_PROMPT_STYLE

    def _is_mbpp_task(self, req) -> bool:
        task_name = getattr(req, "task_name", None)
        return isinstance(task_name, str) and task_name == "mbpp"

    def _coerce_prompt_input(self, question, row=None):
        if self.prompt_style == MBPP_PROMPT_STYLE:
            text = None
            test_list = None
            if row is not None:
                question_key, _ = self._resolved_fewshot_keys()
                text = row.get(question_key, row.get("text"))
                test_list = row.get("test_list")
            elif isinstance(question, dict):
                text = question.get("text")
                if text is None:
                    text = question.get("question", question.get("problem"))
                test_list = question.get("test_list")
            else:
                text = question

            text = "" if text is None else str(text).strip()
            if not text:
                raise ValueError("MBPP prompt text is empty.")
            if not isinstance(test_list, (list, tuple)) or len(test_list) < 3:
                raise KeyError("MBPP prompt requires `test_list` with at least 3 tests.")
            return {
                "text": text,
                "test_list": [str(test) for test in test_list[:3]],
            }

        if isinstance(question, dict):
            candidate_keys = []
            for key in (self.fewshot_question_key, "question", "problem", "text", "prompt"):
                if key and key not in candidate_keys:
                    candidate_keys.append(key)
            for key in candidate_keys:
                value = question.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return str(question)

        return str(question).strip()

    def _format_query_prompt(self, question) -> str:
        prompt_input = self._coerce_prompt_input(question)
        if self.prompt_style == MBPP_PROMPT_STYLE:
            return MBPP_PREFIX_QUERY_TEMPLATE.format(
                question=prompt_input["text"],
                tests="\n".join(prompt_input["test_list"]),
            )
        return PREFIX_QUERY_TEMPLATE.format(question=prompt_input)

    def _load_fewshot_rows(self):
        dataset = load_dataset(
            path=self.fewshot_dataset_path,
            name=self.fewshot_dataset_name,
            split=self.fewshot_split,
        )
        dataset_desc = f"{self.fewshot_dataset_path}/{self.fewshot_dataset_name}:{self.fewshot_split}"
        return dataset, dataset_desc

    def _extract_question_text(self, req) -> str:
        if hasattr(req, "doc") and isinstance(req.doc, dict):
            resolved_question_key, _ = self._resolved_fewshot_keys()
            candidate_keys = []
            for key in (resolved_question_key, self.fewshot_question_key, "question", "problem", "text", "prompt"):
                if key and key not in candidate_keys:
                    candidate_keys.append(key)

            if self.prompt_style == MBPP_PROMPT_STYLE or self._is_mbpp_task(req):
                question = None
                for key in candidate_keys:
                    value = req.doc.get(key)
                    if isinstance(value, str) and value.strip():
                        question = value.strip()
                        break
                test_list = req.doc.get("test_list")
                if question and isinstance(test_list, (list, tuple)) and len(test_list) >= 3:
                    return {
                        "text": question,
                        "test_list": [str(test) for test in test_list[:3]],
                    }

            for key in candidate_keys:
                question = req.doc.get(key)
                if isinstance(question, str) and question.strip():
                    return question.strip()

        prompt_text = req.args[0]
        if not isinstance(prompt_text, str):
            return str(prompt_text)

        prompt_text = prompt_text.strip()
        if self.prompt_style == MBPP_PROMPT_STYLE or self._is_mbpp_task(req):
            match = MBPP_FROM_PROMPT_PATTERN.search(prompt_text)
            if match:
                question = match.group(1).strip()
                tests_block = match.group(2)
                test_list = [line for line in tests_block.splitlines() if line.strip()]
                if question and len(test_list) >= 3:
                    return {
                        "text": question,
                        "test_list": test_list[:3],
                    }

        match = QUESTION_FROM_PROMPT_PATTERN.search(prompt_text)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
        return prompt_text

    def _normalize_fewshot_answer(self, answer: str) -> str:
        if self.prompt_style == MBPP_PROMPT_STYLE:
            text = str(answer).rstrip()
            if text.endswith("[DONE]"):
                return text
            return f"{text}\n[DONE]"

        answer = str(answer)
        text = re.sub(r"</?reasoning>", "", answer, flags=re.IGNORECASE).strip()
        if "####" not in text:
            return text

        prefix, suffix = text.rsplit("####", 1)
        reasoning = prefix.strip()
        final_answer = suffix.strip()
        if not final_answer:
            return reasoning
        if not reasoning:
            return f"The answer is {final_answer}"
        return f"{reasoning}\nThe answer is {final_answer}"

    def _coerce_fewshot_answer_text(self, answer) -> str:
        if isinstance(answer, dict):
            return str(answer.get("value") or answer).strip()
        return str(answer).strip()

    def _build_prefix_fewshot_messages(self):
        if self.fewshot_num_examples <= 0:
            print("  prefix few-shot: disabled (fewshot_num_examples <= 0)")
            return []

        dataset, dataset_desc = self._load_fewshot_rows()
        if len(dataset) == 0:
            raise ValueError(
                "fewshot dataset is empty "
                f"({dataset_desc})."
            )

        question_key, answer_key = self._resolved_fewshot_keys()
        sample_count = min(self.fewshot_num_examples, len(dataset))
        if sample_count < self.fewshot_num_examples:
            print(
                "  prefix few-shot: requested "
                f"{self.fewshot_num_examples} examples, but split only has {len(dataset)}; using {sample_count}."
            )

        sampled_indices = random.sample(range(len(dataset)), sample_count)
        selection_desc = "global random state"
        messages = []
        sampled_examples = []
        normalized_count = 0
        for idx in sampled_indices:
            row = dataset[int(idx)]
            if question_key not in row or answer_key not in row:
                raise KeyError(
                    "fewshot row is missing expected keys "
                    f"('{question_key}', '{answer_key}'). "
                    f"Available keys: {sorted(row.keys())}"
                )

            question = str(row[question_key]).strip()
            raw_answer = row[answer_key]
            answer = self._normalize_fewshot_answer(raw_answer)
            raw_answer_text = self._coerce_fewshot_answer_text(raw_answer)
            if answer != raw_answer_text:
                normalized_count += 1
            question_prompt_input = self._coerce_prompt_input(question, row=row)
            sampled_example = {"index": int(idx), "question": question_prompt_input, "answer": answer}
            if "task_id" in row:
                sampled_example["task_id"] = int(row["task_id"])
            sampled_examples.append(sampled_example)
            messages.append({"role": "user", "content": self._format_query_prompt(question_prompt_input)})
            messages.append({"role": "assistant", "content": answer})

        preview_indices = sampled_indices[:10]
        preview_suffix = "" if len(sampled_indices) <= 10 else "..."
        print(
            "  prefix few-shot: selected "
            f"{sample_count} examples from {dataset_desc} "
            f"using {selection_desc}"
        )
        print(f"  sampled indices: {preview_indices}{preview_suffix}")
        print(f"  normalized sampled answers in {normalized_count}/{sample_count} sampled answers")
        self.sampled_fewshot_examples = sampled_examples
        return messages

    def _build_prefix_fewshot_chat(self, question):
        messages = list(self.prefix_fewshot_messages) if self.prefix_fewshot_messages is not None else []
        messages.append({"role": "user", "content": self._format_query_prompt(question)})
        return messages

    def _build_prefix_affix_token_ids(self):
        if not self.prefix_fewshot_messages:
            return []
        chat = self.tokenizer.apply_chat_template(
            self.prefix_fewshot_messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        return self.tokenizer(chat)["input_ids"]

    def _build_prefix_input_ids(self, question):
        messages = self._build_prefix_fewshot_chat(question)
        chat = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.tokenizer(chat)["input_ids"]

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def extract_and_modify_fewshot(self, prompt, task_name):
        pattern = r"(Question:.*?Answer:.*?)(?=Question:|$)"
        examples = re.findall(pattern, prompt, re.DOTALL)

        if len(examples) > 1:
            actual_question = examples[-1]
            few_shot_examples = examples[:-1]
            num_examples = random.randint(0, len(few_shot_examples))
            selected_examples = random.sample(few_shot_examples, num_examples) if num_examples > 0 else []
            return "".join(selected_examples) + actual_question

        return prompt

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"rank_{self.rank}.jsonl")
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, "r", encoding="utf-8") as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")

        unprocessed_requests = requests[processed_count:]
        if not unprocessed_requests:
            return output

        start_time = time.time()
        tokenization_start = self._profile_timing_mark()

        all_prompts = []
        for req in tqdm(unprocessed_requests, desc="Tokenizing..."):
            if self.use_model_side_prefix_fewshot:
                question = self._extract_question_text(req)
                input_ids = self._build_prefix_input_ids(question)
            else:
                question = req.args[0]
                task_name = req.doc.get("task", "unknown") if hasattr(req, "doc") and req.doc else "unknown"
                if self.variable_fewshot:
                    question = self.extract_and_modify_fewshot(question, task_name)

                if self.is_instruct:
                    messages = [{"role": "user", "content": question}]
                    user_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                else:
                    user_input = question
                input_ids = self.tokenizer(user_input)["input_ids"]
            all_prompts.append(input_ids)
        tokenization_end = self._profile_timing_mark()

        sequence_start = self._profile_timing_mark()
        all_sequences = []
        for prompt in all_prompts:
            seq = Sequence(
                prompt,
                block_length=self.serve_config.block_length,
                gen_length=self.gen_length,
                cache_block_size=self.serve_config.cache_block_size,
                mask_id=self.serve_config.mask_id,
                prompt_affix_len=self.prefix_affix_len if self.serve_config.enable_acache else 0,
            )
            all_sequences.append(seq)
        sequence_end = self._profile_timing_mark()

        generation_start = self._profile_timing_mark()
        if self.serve_config.enable_acache:
            all_generated_parts, nfe = self.runner.generate_with_acache(all_sequences)
        else:
            all_generated_parts, nfe = self.runner.generate_with_dual_cache(all_sequences)
        generation_end = self._profile_timing_mark()

        num_nfe += nfe
        end_time = time.time()

        postprocess_start = self._profile_timing_mark()
        newly_generated_answers = []
        for i, gen_token_ids in enumerate(all_generated_parts):
            if self.show_speed:
                num_tokens += len(gen_token_ids)
            req = unprocessed_requests[i]
            generation_kwargs = req.args[1] if len(req.args) > 1 and isinstance(req.args[1], dict) else {}
            stop_tokens = generation_kwargs.get("until", [])
            if stop_tokens is None:
                stop_tokens = []
            elif isinstance(stop_tokens, str):
                stop_tokens = [stop_tokens]
            else:
                stop_tokens = list(stop_tokens)
            if self.prompt_style == MBPP_PROMPT_STYLE and "[DONE]" not in stop_tokens:
                stop_tokens.append("[DONE]")

            if (
                self.is_instruct
                and hasattr(req, "doc")
                and isinstance(req.doc, dict)
                and "task_id" in req.doc
                and str(req.doc["task_id"]).lower().startswith("humaneval")
            ):
                generated_answer_i = self.tokenizer.decode(gen_token_ids, skip_special_tokens=True)
            else:
                generated_answer_i = self.tokenizer.decode(gen_token_ids, skip_special_tokens=False)
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer_i:
                        generated_answer_i = generated_answer_i.split(stop_seq)[0]

                generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])

                generated_answer_i = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)

            newly_generated_answers.append(generated_answer_i)
        postprocess_end = self._profile_timing_mark()

        if self.save_dir is not None:
            with open(save_path, "a", encoding="utf-8") as f:
                for answer in newly_generated_answers:
                    f.write(json.dumps(answer, ensure_ascii=False) + "\n")

        print("nfe: ", nfe)
        total_output_count = processed_count + len(newly_generated_answers)
        print("avg nfe: ", num_nfe / total_output_count)
        print("=" * 20, end="\n\n")

        output.extend(newly_generated_answers)

        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time)}")
            print(f"Total NFE is {num_nfe}")
            if self.profile_timing:
                self._print_profile_timing(
                    tokenization_s=tokenization_end - tokenization_start,
                    sequence_s=sequence_end - sequence_start,
                    generation_s=generation_end - generation_start,
                    postprocess_s=postprocess_end - postprocess_start,
                )

        return output


@register_model("dream_dist")
class DreamEvalHarness(LLaDAEvalHarness):
    def __init__(
        self,
        model_path="Dream-org/Dream-v0-Instruct-7B",
        mask_id=151666,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            mask_id=mask_id,
            model_type="dream",
            **kwargs,
        )


def _has_model_arg(model_args: str, key: str) -> bool:
    return re.search(rf"(?:^|,)\s*{re.escape(key)}\s*=", model_args or "") is not None


def _append_model_arg(model_args: str, key: str, value) -> str:
    arg = f"{key}={value}"
    if not model_args:
        return arg
    return f"{model_args},{arg}"


def _normalize_eval_task_name(task) -> str:
    return str(task).strip()


def _normalize_eval_tasks(tasks):
    if tasks is None:
        return None
    if isinstance(tasks, str):
        task_names = [
            _normalize_eval_task_name(piece)
            for piece in tasks.split(",")
            if str(piece).strip()
        ]
        return ",".join(task_names)
    return [_normalize_eval_task_name(piece) for piece in tasks if str(piece).strip()]


def _tasks_are_mbpp(tasks) -> bool:
    if tasks is None:
        return False
    if isinstance(tasks, str):
        task_names = [piece.strip() for piece in tasks.split(",") if piece.strip()]
    else:
        task_names = [str(piece).strip() for piece in tasks if str(piece).strip()]
    return bool(task_names) and all(task_name == "mbpp" for task_name in task_names)


def _prepare_cli_args_for_custom_fewshot():
    parser = setup_parser()
    args = parse_eval_args(parser)
    args.tasks = _normalize_eval_tasks(args.tasks)

    model_side_fewshot_arg = "fewshot_num_examples"
    model_side_seed_arg = "seed"
    requested_num_fewshot = args.num_fewshot
    requested_seed = args.seed[0] if getattr(args, "seed", None) else None

    if isinstance(args.model_args, dict):
        if requested_num_fewshot is not None and model_side_fewshot_arg not in args.model_args:
            args.model_args[model_side_fewshot_arg] = requested_num_fewshot
        if requested_seed is not None and model_side_seed_arg not in args.model_args:
            args.model_args[model_side_seed_arg] = requested_seed
        if _tasks_are_mbpp(args.tasks):
            if "fewshot_dataset_path" not in args.model_args:
                args.model_args["fewshot_dataset_path"] = "google-research-datasets/mbpp"
            if "fewshot_dataset_name" not in args.model_args:
                args.model_args["fewshot_dataset_name"] = "full"
            if "fewshot_split" not in args.model_args:
                args.model_args["fewshot_split"] = "prompt"
            if "fewshot_question_key" not in args.model_args:
                args.model_args["fewshot_question_key"] = "text"
            if "fewshot_answer_key" not in args.model_args:
                args.model_args["fewshot_answer_key"] = "code"
    else:
        model_args_str = args.model_args or ""
        if requested_num_fewshot is not None and not _has_model_arg(model_args_str, model_side_fewshot_arg):
            model_args_str = _append_model_arg(model_args_str, model_side_fewshot_arg, requested_num_fewshot)
        if requested_seed is not None and not _has_model_arg(model_args_str, model_side_seed_arg):
            model_args_str = _append_model_arg(model_args_str, model_side_seed_arg, requested_seed)
        if _tasks_are_mbpp(args.tasks):
            if not _has_model_arg(model_args_str, "fewshot_dataset_path"):
                model_args_str = _append_model_arg(model_args_str, "fewshot_dataset_path", "google-research-datasets/mbpp")
            if not _has_model_arg(model_args_str, "fewshot_dataset_name"):
                model_args_str = _append_model_arg(model_args_str, "fewshot_dataset_name", "full")
            if not _has_model_arg(model_args_str, "fewshot_split"):
                model_args_str = _append_model_arg(model_args_str, "fewshot_split", "prompt")
            if not _has_model_arg(model_args_str, "fewshot_question_key"):
                model_args_str = _append_model_arg(model_args_str, "fewshot_question_key", "text")
            if not _has_model_arg(model_args_str, "fewshot_answer_key"):
                model_args_str = _append_model_arg(model_args_str, "fewshot_answer_key", "code")
        args.model_args = model_args_str

    if requested_num_fewshot is not None:
        print(
            f"[eval_llada] mapping --num_fewshot={requested_num_fewshot} to model arg `{model_side_fewshot_arg}` "
            "and disabling lm_eval few-shot."
        )
    if requested_seed is not None:
        print(
            f"[eval_llada] mapping --seed={requested_seed} to model arg `{model_side_seed_arg}` "
            "and disabling lm_eval seeding."
        )
    if _tasks_are_mbpp(args.tasks):
        print(
            "[eval_llada] auto-selected MBPP prompt-split few-shot samples "
            "(`fewshot_dataset_path=google-research-datasets/mbpp`, `full:prompt`, `text`/`code` keys)."
        )
    args.num_fewshot = 0
    args.seed = [None, None, None, None]
    return args


if __name__ == "__main__":
    cli_evaluate(_prepare_cli_args_for_custom_fewshot())

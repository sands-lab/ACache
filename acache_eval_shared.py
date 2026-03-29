import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from lm_eval.__main__ import parse_eval_args, setup_parser
from tqdm import tqdm


CUSTOM_LM_EVAL_TASKS_PATH = Path(__file__).resolve().parent / "lm_eval_tasks"

GSM8K_PROMPT_STYLE = "gsm8k"
MBPP_PROMPT_STYLE = "mbpp"
BABILONG_QA1_PROMPT_STYLE = "babilong_qa1"

PREFIX_QUERY_TEMPLATE = "Question: {question}\nLet's think step by step.\nAnswer:"
INFIX_QUERY_TEMPLATE = "Question: {question}\n\nExample(s):"
INFIX_FINAL_ANSWER_PROMPT = "Let's come back to the question in the beginning and think step by step.\nAnswer:"
MBPP_PREFIX_QUERY_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {question} "
    "Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
)
MBPP_INFIX_QUERY_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {question} "
    "Your code should pass these tests:\n\n{tests}\n\nExample(s):"
)
MBPP_INFIX_FINAL_ANSWER_PROMPT = (
    "Let's come back to the task in the beginning and write the Python code.\n"
    "Do not include any explanation, comments outside the code, markdown fences, or test cases.\n[BEGIN]\n"
)
BABILONG_QA1_PREFIX_QUERY_TEMPLATE = (
    "Story:\n{story}\nQuestion: {question}\nAnswer with one location:"
)
BABILONG_QA1_INFIX_QUERY_TEMPLATE = (
    "Story:\n{story}\nQuestion: {question}\n\nExample(s):"
)
BABILONG_QA1_INFIX_FINAL_ANSWER_PROMPT = (
    "Let's come back to the story and question in the beginning and answer with one location."
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


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def coerce_bool(value, default: bool, arg_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{arg_name} must be a boolean-like value, got {value!r}.")


def has_model_arg(model_args: str, key: str) -> bool:
    return re.search(rf"(?:^|,)\s*{re.escape(key)}\s*=", model_args or "") is not None


def append_model_arg(model_args: str, key: str, value) -> str:
    arg = f"{key}={value}"
    if not model_args:
        return arg
    return f"{model_args},{arg}"


def parse_model_args_dict(model_args):
    if isinstance(model_args, dict):
        return dict(model_args)

    parsed = {}
    for part in (model_args or "").split(","):
        piece = part.strip()
        if not piece or "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def sanitize_path_component(value, default="unknown") -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-")
    return text or default


def int_or_default(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def ceil_to_int(value: float) -> int:
    integer = int(value)
    if value == integer:
        return integer
    return integer + 1


def format_ratio(value, default="0.1") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return sanitize_path_component(value, default=default)


def derive_auto_output_path(model_args, fallback_fewshot, fallback_seed) -> str:
    model_args_dict = parse_model_args_dict(model_args)
    drop_non_anchor = coerce_bool(
        model_args_dict.get("drop_non_anchor", False),
        default=False,
        arg_name="drop_non_anchor",
    )

    mode = sanitize_path_component(model_args_dict.get("selection_mode", "top"), default="top")
    ratio = format_ratio(model_args_dict.get("anchor_ratio"), default="0")
    affix = sanitize_path_component(model_args_dict.get("affix_type", "prefix"), default="prefix")
    fewshot = int_or_default(model_args_dict.get("fewshot_num_examples"), fallback_fewshot)
    seed = int_or_default(model_args_dict.get("seed"), fallback_seed)
    non_anchor_mode = "dropna" if drop_non_anchor else "keepna"

    run_name = f"anchor_{mode}_{ratio}_{non_anchor_mode}_{fewshot}shot_{affix}_seed_{seed}"
    return str(Path("evals_results") / run_name)


def normalize_eval_task_name(task) -> str:
    return str(task).strip()


def normalize_eval_tasks(tasks):
    if tasks is None:
        return None
    if isinstance(tasks, str):
        task_names = [
            normalize_eval_task_name(piece)
            for piece in tasks.split(",")
            if str(piece).strip()
        ]
        return ",".join(task_names)
    return [normalize_eval_task_name(piece) for piece in tasks if str(piece).strip()]


def ensure_custom_task_include_path(args):
    if not CUSTOM_LM_EVAL_TASKS_PATH.is_dir():
        return args

    custom_path = str(CUSTOM_LM_EVAL_TASKS_PATH)
    include_path = getattr(args, "include_path", None)
    if include_path in (None, ""):
        args.include_path = custom_path
        return args

    if isinstance(include_path, str):
        if include_path != custom_path:
            args.include_path = [include_path, custom_path]
        return args

    include_paths = list(include_path)
    if custom_path not in include_paths:
        include_paths.append(custom_path)
    args.include_path = include_paths
    return args


def tasks_are_mbpp(tasks) -> bool:
    if tasks is None:
        return False
    if isinstance(tasks, str):
        task_names = [piece.strip() for piece in tasks.split(",") if piece.strip()]
    else:
        task_names = [str(piece).strip() for piece in tasks if str(piece).strip()]
    return bool(task_names) and all(task_name == "mbpp" for task_name in task_names)


def tasks_are_babilong_qa1(tasks) -> bool:
    if tasks is None:
        return False
    if isinstance(tasks, str):
        task_names = [piece.strip() for piece in tasks.split(",") if piece.strip()]
    else:
        task_names = [str(piece).strip() for piece in tasks if str(piece).strip()]

    valid_task_names = {"babilong"}
    return bool(task_names) and all(task_name in valid_task_names for task_name in task_names)


def tasks_include_humaneval(tasks) -> bool:
    if tasks is None:
        return False
    if isinstance(tasks, str):
        task_names = [piece.strip() for piece in tasks.split(",") if piece.strip()]
    else:
        task_names = [str(piece).strip() for piece in tasks if str(piece).strip()]
    return any(task_name.lower().startswith("humaneval") for task_name in task_names)


def prepare_cli_args_for_custom_fewshot():
    parser = setup_parser()
    args = parse_eval_args(parser)
    args.tasks = normalize_eval_tasks(args.tasks)
    args = ensure_custom_task_include_path(args)

    if tasks_include_humaneval(args.tasks):
        raise ValueError(
            "HumanEval tasks are no longer supported in this repository. "
            "Use GSM8K or MBPP tasks instead."
        )

    model_side_fewshot_arg = "fewshot_num_examples"
    model_side_seed_arg = "seed"
    requested_num_fewshot = args.num_fewshot
    requested_seed = args.seed[0]

    if isinstance(args.model_args, dict):
        if requested_num_fewshot is not None and model_side_fewshot_arg not in args.model_args:
            args.model_args[model_side_fewshot_arg] = requested_num_fewshot
        if requested_seed is not None and model_side_seed_arg not in args.model_args:
            args.model_args[model_side_seed_arg] = requested_seed
        if tasks_are_mbpp(args.tasks):
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
        if tasks_are_babilong_qa1(args.tasks):
            if "fewshot_dataset_path" not in args.model_args:
                args.model_args["fewshot_dataset_path"] = "RMT-team/babilong-1k-samples"
            if "fewshot_dataset_name" not in args.model_args:
                args.model_args["fewshot_dataset_name"] = "0k"
            if "fewshot_split" not in args.model_args:
                args.model_args["fewshot_split"] = "qa1"
            if "fewshot_question_key" not in args.model_args:
                args.model_args["fewshot_question_key"] = "question"
            if "fewshot_answer_key" not in args.model_args:
                args.model_args["fewshot_answer_key"] = "target"
    else:
        model_args_str = args.model_args or ""
        if requested_num_fewshot is not None and not has_model_arg(model_args_str, model_side_fewshot_arg):
            model_args_str = append_model_arg(model_args_str, model_side_fewshot_arg, requested_num_fewshot)
        if requested_seed is not None and not has_model_arg(model_args_str, model_side_seed_arg):
            model_args_str = append_model_arg(model_args_str, model_side_seed_arg, requested_seed)
        if tasks_are_mbpp(args.tasks):
            if not has_model_arg(model_args_str, "fewshot_dataset_path"):
                model_args_str = append_model_arg(model_args_str, "fewshot_dataset_path", "google-research-datasets/mbpp")
            if not has_model_arg(model_args_str, "fewshot_dataset_name"):
                model_args_str = append_model_arg(model_args_str, "fewshot_dataset_name", "full")
            if not has_model_arg(model_args_str, "fewshot_split"):
                model_args_str = append_model_arg(model_args_str, "fewshot_split", "prompt")
            if not has_model_arg(model_args_str, "fewshot_question_key"):
                model_args_str = append_model_arg(model_args_str, "fewshot_question_key", "text")
            if not has_model_arg(model_args_str, "fewshot_answer_key"):
                model_args_str = append_model_arg(model_args_str, "fewshot_answer_key", "code")
        if tasks_are_babilong_qa1(args.tasks):
            if not has_model_arg(model_args_str, "fewshot_dataset_path"):
                model_args_str = append_model_arg(model_args_str, "fewshot_dataset_path", "RMT-team/babilong-1k-samples")
            if not has_model_arg(model_args_str, "fewshot_dataset_name"):
                model_args_str = append_model_arg(model_args_str, "fewshot_dataset_name", "0k")
            if not has_model_arg(model_args_str, "fewshot_split"):
                model_args_str = append_model_arg(model_args_str, "fewshot_split", "qa1")
            if not has_model_arg(model_args_str, "fewshot_question_key"):
                model_args_str = append_model_arg(model_args_str, "fewshot_question_key", "question")
            if not has_model_arg(model_args_str, "fewshot_answer_key"):
                model_args_str = append_model_arg(model_args_str, "fewshot_answer_key", "target")
        args.model_args = model_args_str

    if requested_num_fewshot is not None:
        print(
            f"[eval_ACache] mapping --num_fewshot={requested_num_fewshot} to model arg `{model_side_fewshot_arg}` "
            "and disabling lm_eval few-shot."
        )
    if requested_seed is not None:
        print(
            f"[eval_ACache] mapping --seed={requested_seed} to model arg `{model_side_seed_arg}` "
            "and disabling lm_eval seeding."
        )
    if tasks_are_mbpp(args.tasks):
        print(
            "[eval_ACache] auto-selected MBPP prompt-split few-shot samples "
            "(`fewshot_dataset_path=google-research-datasets/mbpp`, `full:prompt`, `text`/`code` keys)."
        )
    if tasks_are_babilong_qa1(args.tasks):
        print(
            "[eval_ACache] auto-selected BABILONG-QA1 few-shot samples "
            "(`fewshot_dataset_path=RMT-team/babilong-1k-samples`, `0k:qa1`, `question`/`target` keys)."
        )
    if args.output_path in (None, "", "auto"):
        fallback_fewshot = 0 if requested_num_fewshot is None else int(requested_num_fewshot)
        fallback_seed = 0 if requested_seed is None else int(requested_seed)
        args.output_path = derive_auto_output_path(
            model_args=args.model_args,
            fallback_fewshot=fallback_fewshot,
            fallback_seed=fallback_seed,
        )
        print(f"[eval_ACache] auto-derived --output_path={args.output_path}")
    args.num_fewshot = 0
    args.seed = [None, None, None, None]
    return args


class ACacheEvalHarnessMixin:
    @property
    def rank(self):
        return self._rank if hasattr(self, "_rank") else 0

    @property
    def world_size(self):
        return self._world_size if hasattr(self, "_world_size") else 1

    def _configure_acache_common(
        self,
        *,
        batch_size,
        max_length,
        mc_num,
        is_check_greedy,
        steps,
        gen_length,
        block_length,
        remasking,
        threshold,
        factor,
        save_dir,
        show_speed,
        anchor_ratio,
        affix_type,
        selection_mode,
        drop_non_anchor,
        fewshot_num_examples,
        fewshot_dataset_path,
        fewshot_dataset_name,
        fewshot_split,
        fewshot_question_key,
        fewshot_answer_key,
        model_label,
    ):
        self.model_label = model_label
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        self.sampling_eps = 0.0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.threshold = threshold
        self.factor = factor
        self.save_dir = save_dir
        self.show_speed = coerce_bool(show_speed, default=True, arg_name="show_speed")

        self.anchor_ratio = float(anchor_ratio)
        if not (0.0 <= self.anchor_ratio <= 1.0):
            raise ValueError(f"anchor_ratio must be between 0 and 1 inclusive, got {self.anchor_ratio}.")
        self.affix_type = affix_type
        if self.batch_size != 1:
            if self.affix_type == "suffix":
                raise ValueError("Real suffix mode currently supports batch_size=1.")
            raise ValueError(f"{model_label} currently supports batch_size=1.")
        self.selection_mode = selection_mode
        self.drop_non_anchor = coerce_bool(drop_non_anchor, default=False, arg_name="drop_non_anchor")
        self.fewshot_num_examples = int(fewshot_num_examples)
        self.fewshot_dataset_path = fewshot_dataset_path
        self.fewshot_dataset_name = fewshot_dataset_name
        self.fewshot_split = fewshot_split
        self.fewshot_question_key = fewshot_question_key
        self.fewshot_answer_key = fewshot_answer_key
        self.prompt_style = self._resolve_prompt_style()
        print("ACache Config:")
        print(f"  anchor_ratio: {self.anchor_ratio}")
        print(f"  affix_type: {self.affix_type}")
        print(f"  prompt_style: {self.prompt_style}")
        print(f"  selection_mode: {self.selection_mode}")
        print(f"  drop_non_anchor: {self.drop_non_anchor}")
        if self.affix_type not in {"prefix", "suffix", "infix", "none"}:
            raise ValueError(
                f"Unsupported affix_type={affix_type!r}. Expected 'prefix', 'suffix', 'infix', or 'none'."
            )

        self.default_affix_state = self._make_empty_affix_state(state_name="default")
        self._initialize_affix_states()

    def _initialize_affix_states(self):
        if self.affix_type not in {"prefix", "suffix", "infix"}:
            print("  affix_bank_mode: none")
            return

        if self._uses_mbpp_fewshot_dataset() and str(self.fewshot_split or "").strip().lower() == "prompt":
            print("  affix_bank_mode: global MBPP prompt-split few-shot examples")
        elif self._uses_babilong_qa1_fewshot_dataset():
            print("  affix_bank_mode: global BABILONG-QA1 few-shot examples")
        else:
            print("  affix_bank_mode: global")
        self.default_affix_state = self._build_affix_state(
            state_name="default",
        )

    def _make_empty_affix_state(self, state_name: str):
        return {
            "state_name": state_name,
            "sampled_fewshot_examples": [],
            "prefix_fewshot_messages": None,
            "infix_plaintext_messages": None,
            "infix_affix_token_ids": None,
            "suffix_affix_token_ids": None,
            "affix_length": 0,
            "precomputed_affix_cache": None,
        }

    def _resolve_affix_state_for_request(self, req):
        return self.default_affix_state

    def _build_affix_state(self, state_name="default"):
        state = self._make_empty_affix_state(state_name)

        prefix_fewshot_messages, sampled_examples = self._build_prefix_fewshot_messages(
            state_name=state_name,
        )
        state["sampled_fewshot_examples"] = sampled_examples
        state["prefix_fewshot_messages"] = prefix_fewshot_messages
        prefix_affix_token_ids = None

        if self.affix_type == "infix":
            state["infix_plaintext_messages"] = self._build_infix_plaintext_messages(sampled_examples)
            if state["infix_plaintext_messages"]:
                probe_question = self._build_affix_probe_question(sampled_examples)
                probe_input_ids, affix_start, affix_end, _, _ = self._build_input_ids_with_affix(
                    probe_question,
                    state,
                )
                state["infix_affix_token_ids"] = probe_input_ids[affix_start:affix_end]
                state["affix_length"] = len(state["infix_affix_token_ids"])
        elif self.affix_type == "suffix":
            if prefix_fewshot_messages:
                probe_assistant_turn = self.tokenizer.decode([self.mask_id], skip_special_tokens=False)
                suffix_probe_messages = [{"role": "assistant", "content": probe_assistant_turn}]
                suffix_probe_messages.extend(prefix_fewshot_messages)
                suffix_probe_ids = self._tokenize_chat_messages(
                    suffix_probe_messages,
                    add_generation_prompt=False,
                )
                probe_mask_idx = suffix_probe_ids.index(self.mask_id)
                state["suffix_affix_token_ids"] = suffix_probe_ids[probe_mask_idx + 1:]
                state["affix_length"] = len(state["suffix_affix_token_ids"])
        else:
            if prefix_fewshot_messages:
                prefix_affix_token_ids = self._tokenize_chat_messages(
                    prefix_fewshot_messages,
                    add_generation_prompt=False,
                )
                state["affix_length"] = len(prefix_affix_token_ids)

        if state["affix_length"] > 0:
            if self.affix_type == "infix":
                affix_ids = state["infix_affix_token_ids"]
            elif self.affix_type == "suffix":
                affix_ids = state["suffix_affix_token_ids"]
            else:
                affix_ids = prefix_affix_token_ids
            affix_ids = torch.tensor(affix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            with torch.inference_mode():
                affix_out = self.model(affix_ids, use_cache=True)
            state["precomputed_affix_cache"] = affix_out.past_key_values

        self._log_affix_state_summary(state)
        return state

    def _log_affix_state_summary(self, affix_state):
        affix_length = int(affix_state["affix_length"])
        resolved_num_anchor = min(ceil_to_int(self.anchor_ratio * affix_length), affix_length)
        print(f"Affix State `{affix_state['state_name']}`:")
        print(f"  num_anchor (resolved K): {resolved_num_anchor}")
        print(f"  affix_length: {affix_length}")
        if affix_state["precomputed_affix_cache"] is not None:
            print(f"  precomputed_affix_cache: enabled (len={affix_length})")
        else:
            print("  precomputed_affix_cache: disabled")

    def _uses_chat_template_for_prompts(self) -> bool:
        return True

    def _prompt_text_delimiter(self) -> str:
        return "\n\n"

    def _prepare_plain_prompt_text(self, prompt_text: str) -> str:
        return prompt_text

    def _render_plain_prompt_from_messages(self, messages, add_generation_prompt: bool) -> str:
        del add_generation_prompt

        units = []
        idx = 0
        while idx < len(messages):
            message = messages[idx]
            role = message.get("role")
            content = message.get("content", "")

            if role == "user":
                if idx + 1 < len(messages) and messages[idx + 1].get("role") == "assistant":
                    content = f"{content}{messages[idx + 1].get('content', '')}"
                    idx += 2
                else:
                    idx += 1
                units.append(content)
                continue

            units.append(content)
            idx += 1

        return self._prompt_text_delimiter().join(units)

    def _tokenize_plain_prompt(self, prompt_text: str):
        prompt_text = self._prepare_plain_prompt_text(prompt_text)
        return self.tokenizer(prompt_text)["input_ids"]

    def _uses_mbpp_fewshot_dataset(self) -> bool:
        dataset_path = str(self.fewshot_dataset_path or "").strip().lower()
        if dataset_path.startswith("lm_eval:"):
            dataset_path = dataset_path[len("lm_eval:"):]
        return dataset_path in {"mbpp", "google-research-datasets/mbpp"} or dataset_path.endswith("/mbpp")

    def _uses_babilong_qa1_fewshot_dataset(self) -> bool:
        dataset_path = str(self.fewshot_dataset_path or "").strip().lower()
        if dataset_path.startswith("lm_eval:"):
            dataset_path = dataset_path[len("lm_eval:"):]
        dataset_name = str(self.fewshot_dataset_name or "").strip().lower()
        return "babilong" in dataset_path or dataset_name in {"0k", "babilong"}

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
        if self._uses_babilong_qa1_fewshot_dataset():
            return BABILONG_QA1_PROMPT_STYLE
        if question_key == "text" and answer_key == "code":
            return MBPP_PROMPT_STYLE
        return GSM8K_PROMPT_STYLE

    def _is_mbpp_task(self, req) -> bool:
        task_name = getattr(req, "task_name", None)
        return isinstance(task_name, str) and task_name == "mbpp"

    def _is_babilong_qa1_task(self, req) -> bool:
        task_name = getattr(req, "task_name", None)
        return isinstance(task_name, str) and task_name == "babilong"

    def _build_affix_probe_question(self, sampled_fewshot_examples):
        if self.prompt_style == MBPP_PROMPT_STYLE:
            if sampled_fewshot_examples:
                first_question = sampled_fewshot_examples[0].get("question")
                if isinstance(first_question, dict):
                    test_list = first_question.get("test_list")
                    if isinstance(test_list, (list, tuple)) and len(test_list) >= 3:
                        return {
                            "text": "__acache_probe__",
                            "test_list": [str(test) for test in test_list[:3]],
                        }
            return {
                "text": "__acache_probe__",
                "test_list": [
                    "assert True",
                    "assert 1 == 1",
                    "assert isinstance(1, int)",
                ],
            }
        return "__acache_probe__"

    def _coerce_prompt_input(self, question, row=None):
        if self.prompt_style == BABILONG_QA1_PROMPT_STYLE:
            if row is not None:
                question_key, _ = self._resolved_fewshot_keys()
                story = row.get("input", "")
                question_value = row.get(question_key, question)
                if story is None:
                    story = ""
                if question_value is None:
                    question_value = ""
                return {
                    "story": str(story).strip(),
                    "question": str(question_value).strip(),
                }

            if isinstance(question, dict):
                story = question.get("story", question.get("input", ""))
                question_value = question.get("question")
                if story is None:
                    story = ""
                if question_value is None:
                    question_value = question.get("text", question.get("query", ""))
                if question_value is None:
                    raise ValueError("BABILONG-QA1 prompt input is missing a question.")
                return {
                    "story": str(story).strip(),
                    "question": str(question_value).strip(),
                }

            return {
                "story": "",
                "question": str(question).strip(),
            }

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

    def _format_query_prompt(self, question: str) -> str:
        prompt_input = self._coerce_prompt_input(question)
        if self.prompt_style == MBPP_PROMPT_STYLE:
            return MBPP_PREFIX_QUERY_TEMPLATE.format(
                question=prompt_input["text"],
                tests="\n".join(prompt_input["test_list"]),
            )
        if self.prompt_style == BABILONG_QA1_PROMPT_STYLE:
            return BABILONG_QA1_PREFIX_QUERY_TEMPLATE.format(
                story=prompt_input["story"],
                question=prompt_input["question"],
            )
        return PREFIX_QUERY_TEMPLATE.format(question=prompt_input)

    def _format_infix_query_prompt(self, question: str) -> str:
        prompt_input = self._coerce_prompt_input(question)
        if self.prompt_style == MBPP_PROMPT_STYLE:
            return MBPP_INFIX_QUERY_TEMPLATE.format(
                question=prompt_input["text"],
                tests="\n".join(prompt_input["test_list"]),
            )
        if self.prompt_style == BABILONG_QA1_PROMPT_STYLE:
            return BABILONG_QA1_INFIX_QUERY_TEMPLATE.format(
                story=prompt_input["story"],
                question=prompt_input["question"],
            )
        return INFIX_QUERY_TEMPLATE.format(question=prompt_input)

    def _format_infix_final_answer_prompt(self) -> str:
        if self.prompt_style == MBPP_PROMPT_STYLE:
            return MBPP_INFIX_FINAL_ANSWER_PROMPT
        if self.prompt_style == BABILONG_QA1_PROMPT_STYLE:
            return BABILONG_QA1_INFIX_FINAL_ANSWER_PROMPT
        return INFIX_FINAL_ANSWER_PROMPT

    def _load_mbpp_rows(self, dataset_path: str, dataset_name: str, split_name: str):
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_patterns = [
            f"datasets/mbpp/{dataset_name}/*/*/mbpp-{split_name}.arrow",
            f"datasets/google-research-datasets___mbpp/{dataset_name}/*/*/mbpp-{split_name}.arrow",
            f"hub/datasets--mbpp/snapshots/*/{dataset_name}/{split_name}-*.parquet",
            f"hub/datasets--google-research-datasets--mbpp/snapshots/*/{dataset_name}/{split_name}-*.parquet",
        ]

        for pattern in cache_patterns:
            for cached_path in sorted(hf_home.glob(pattern)):
                if cached_path.suffix == ".arrow":
                    return Dataset.from_file(str(cached_path))
                if cached_path.suffix == ".parquet":
                    return load_dataset("parquet", data_files=str(cached_path), split="train")

        return load_dataset(
            path=dataset_path,
            name=dataset_name,
            split=split_name,
        )

    def _load_mbpp_fewshot_rows(self):
        dataset_path = str(self.fewshot_dataset_path or "google-research-datasets/mbpp").strip() or "google-research-datasets/mbpp"
        dataset_name = str(self.fewshot_dataset_name or "full").strip() or "full"
        split_name = str(self.fewshot_split or "prompt").strip() or "prompt"
        normalized_dataset_path = dataset_path.lower()
        if normalized_dataset_path.startswith("lm_eval:"):
            normalized_dataset_path = normalized_dataset_path[len("lm_eval:"):]
        if normalized_dataset_path == "mbpp":
            dataset_path = "google-research-datasets/mbpp"
        return self._load_mbpp_rows(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            split_name=split_name,
        )

    def _load_fewshot_rows(self):
        if self._uses_mbpp_fewshot_dataset():
            dataset = self._load_mbpp_fewshot_rows()
            dataset_desc = f"{self.fewshot_dataset_path}/{self.fewshot_dataset_name}:{self.fewshot_split}"
            return dataset, dataset_desc

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
            if self.prompt_style == BABILONG_QA1_PROMPT_STYLE or self._is_babilong_qa1_task(req):
                question = req.doc.get("question")
                story = req.doc.get("input", req.doc.get("story", req.doc.get("context")))
                if question is None:
                    raise ValueError("BABILONG-QA1 request is missing the `question` field.")
                if story is None:
                    raise ValueError("BABILONG-QA1 request is missing the `input` field.")
                return {
                    "story": str(story).strip(),
                    "question": str(question).strip(),
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
            text = answer.rstrip()
            if text.endswith("[DONE]"):
                return text
            return f"{text}\n[DONE]"

        if self.prompt_style == BABILONG_QA1_PROMPT_STYLE:
            return str(answer).strip()

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

    def _build_prefix_fewshot_messages(self, state_name=None):
        state_prefix = f"[{state_name}] " if state_name else ""
        if self.fewshot_num_examples <= 0:
            print(f"  {state_prefix}prefix few-shot: disabled (fewshot_num_examples <= 0)")
            return [], []

        dataset, dataset_desc = self._load_fewshot_rows()
        if len(dataset) == 0:
            raise ValueError(
                "fewshot dataset is empty "
                f"({dataset_desc})."
            )

        question_key, answer_key = self._resolved_fewshot_keys()
        candidate_indices = list(range(len(dataset)))
        available_count = len(candidate_indices)
        available_desc = f"split only has {available_count}"

        sample_count = min(self.fewshot_num_examples, available_count)
        if sample_count < self.fewshot_num_examples:
            print(
                f"  {state_prefix}prefix few-shot: requested "
                f"{self.fewshot_num_examples} examples, but {available_desc}; using {sample_count}."
            )

        if self._uses_mbpp_fewshot_dataset() and str(self.fewshot_split or "").strip().lower() == "prompt":
            selected_indices = random.sample(candidate_indices, sample_count)
            selection_desc = "MBPP prompt split random state"
        else:
            selected_indices = random.sample(candidate_indices, sample_count)
            selection_desc = "global random state"
        messages = []
        sampled_examples = []
        normalized_count = 0
        for idx in selected_indices:
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

        preview_indices = selected_indices[:10]
        preview_suffix = "" if len(selected_indices) <= 10 else "..."
        print(
            f"  {state_prefix}prefix few-shot: selected "
            f"{sample_count} examples from {dataset_desc} "
            f"using {selection_desc}"
        )
        print(f"  {state_prefix}sampled indices: {preview_indices}{preview_suffix}")
        print(
            f"  {state_prefix}normalized sampled answers in "
            f"{normalized_count}/{sample_count} sampled answers"
        )
        return messages, sampled_examples

    def _build_infix_plaintext_messages(self, sampled_fewshot_examples):
        if not sampled_fewshot_examples:
            return []

        example_blocks = []
        for i, example in enumerate(sampled_fewshot_examples, start=1):
            example_blocks.append(
                f"Example {i}:\n"
                f"{self._format_query_prompt(example['question'])}\n"
                f"{example['answer']}"
            )
        return [{"role": "user", "content": "\n\n".join(example_blocks)}]

    def _build_infix_user_content(
        self,
        question: str,
        include_examples: bool,
        include_answer_prompt: bool,
        affix_state,
    ) -> str:
        content = self._format_infix_query_prompt(question)
        infix_plaintext_messages = affix_state["infix_plaintext_messages"]
        if include_examples and infix_plaintext_messages:
            examples_text = infix_plaintext_messages[0]["content"]
            content = f"{content}\n\n{examples_text}"
        if include_answer_prompt:
            content = f"{content}\n\n{self._format_infix_final_answer_prompt()}"
        return content

    def _build_prefix_fewshot_chat(self, question: str, affix_state):
        prefix_fewshot_messages = affix_state["prefix_fewshot_messages"]
        messages = list(prefix_fewshot_messages) if prefix_fewshot_messages is not None else []
        messages.append({"role": "user", "content": self._format_query_prompt(question)})
        return messages

    def _build_suffix_question_response_messages(self, question: str):
        return [{"role": "user", "content": self._format_query_prompt(question)}]

    def _format_no_affix_prompt(self, question: str) -> str:
        return self._format_query_prompt(question)

    def _build_suffix_fewshot_chat(self, question: str, affix_state):
        messages = self._build_suffix_question_response_messages(question)
        prefix_fewshot_messages = affix_state["prefix_fewshot_messages"]
        if prefix_fewshot_messages:
            messages.extend(prefix_fewshot_messages)
        return messages

    def _build_infix_fewshot_chat(self, question: str, affix_state):
        content = self._build_infix_user_content(
            question=question,
            include_examples=True,
            include_answer_prompt=True,
            affix_state=affix_state,
        )
        return [{"role": "user", "content": content}]

    def _tokenize_chat_messages(self, messages, add_generation_prompt: bool):
        if self._uses_chat_template_for_prompts():
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
            return self.tokenizer(prompt_text)["input_ids"]

        prompt_text = self._render_plain_prompt_from_messages(
            messages,
            add_generation_prompt=add_generation_prompt,
        )
        return self._tokenize_plain_prompt(prompt_text)

    def _encode_text_with_optional_offsets(self, prompt_text: str):
        try:
            return self.tokenizer(prompt_text, return_offsets_mapping=True)
        except (TypeError, NotImplementedError):
            return self.tokenizer(prompt_text)

    def _find_token_span_for_substring(self, prompt_text: str, substring: str, offsets):
        if not substring or offsets is None:
            return None

        span_start = prompt_text.find(substring)
        if span_start < 0:
            return None
        span_end = span_start + len(substring)

        contained = []
        for idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if start >= span_start and end <= span_end:
                contained.append(idx)

        if not contained:
            return None

        return contained[0], contained[-1] + 1

    def _find_token_span_via_decoded_prefixes(self, input_ids, substring: str):
        if not substring:
            return None

        decoded_full = self._decode_token_ids(input_ids)
        span_start = decoded_full.find(substring)
        if span_start < 0:
            return None
        span_end = span_start + len(substring)

        prefix_lengths = [0]
        for end in range(1, len(input_ids) + 1):
            prefix_lengths.append(len(self._decode_token_ids(input_ids[:end])))

        contained = []
        for idx in range(len(input_ids)):
            token_start = prefix_lengths[idx]
            token_end = prefix_lengths[idx + 1]
            if token_start == token_end:
                continue
            if token_start >= span_start and token_end <= span_end:
                contained.append(idx)

        if not contained:
            return None

        return contained[0], contained[-1] + 1

    def _build_generation_span(self, input_ids):
        generation_start = len(input_ids)
        return generation_start, generation_start + self.gen_length

    def _resolve_infix_affix_span_from_prompt_text(self, input_ids, affix_state, prompt_text: str, encoded):
        affix_text = affix_state["infix_plaintext_messages"][0]["content"]
        affix_span = self._find_token_span_for_substring(
            prompt_text,
            affix_text,
            encoded.get("offset_mapping"),
        )
        if affix_span is not None:
            return affix_span

        # Slow tokenizers may not expose offsets; decoded-prefix fallback keeps the
        # affix slice aligned to whole tokens only.
        return self._find_token_span_via_decoded_prefixes(
            input_ids,
            affix_text,
        )

    def _resolve_infix_affix_span_via_retokenized_segments(self, question: str, affix_state, input_ids):
        question_header_ids = self._tokenize_chat_messages(
            [{
                "role": "user",
                "content": self._build_infix_user_content(
                    question=question,
                    include_examples=False,
                    include_answer_prompt=False,
                    affix_state=affix_state,
                ),
            }],
            add_generation_prompt=False,
        )
        question_plus_examples_ids = self._tokenize_chat_messages(
            [{
                "role": "user",
                "content": self._build_infix_user_content(
                    question=question,
                    include_examples=True,
                    include_answer_prompt=False,
                    affix_state=affix_state,
                ),
            }],
            add_generation_prompt=False,
        )
        affix_start = min(len(question_header_ids), len(input_ids))
        affix_end = min(len(question_plus_examples_ids), len(input_ids))
        if affix_end < affix_start:
            affix_end = affix_start
        return affix_start, affix_end

    def _resolve_infix_affix_span(self, question: str, affix_state, input_ids, prompt_text=None, encoded=None):
        if not affix_state["infix_plaintext_messages"]:
            return 0, 0

        if prompt_text is not None and encoded is not None:
            affix_span = self._resolve_infix_affix_span_from_prompt_text(
                input_ids,
                affix_state,
                prompt_text,
                encoded,
            )
            if affix_span is not None:
                return affix_span

        return self._resolve_infix_affix_span_via_retokenized_segments(
            question,
            affix_state,
            input_ids,
        )

    def _build_input_ids_with_affix(self, question: str, affix_state):
        affix_length = affix_state["affix_length"]
        if self.affix_type == "prefix":
            messages = self._build_prefix_fewshot_chat(question, affix_state)
            input_ids = self._tokenize_chat_messages(
                messages,
                add_generation_prompt=True,
            )
            affix_start = 0
            affix_end = min(affix_length, len(input_ids))
            generation_start, generation_end = self._build_generation_span(input_ids)
            return input_ids, affix_start, affix_end, generation_start, generation_end

        if self.affix_type == "infix":
            messages = self._build_infix_fewshot_chat(question, affix_state)
            if self._uses_chat_template_for_prompts():
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                encoded = self._encode_text_with_optional_offsets(prompt_text)
                input_ids = encoded["input_ids"]
            else:
                prompt_text = None
                encoded = None
                input_ids = self._tokenize_chat_messages(
                    messages,
                    add_generation_prompt=True,
                )
            affix_start, affix_end = self._resolve_infix_affix_span(
                question,
                affix_state,
                input_ids,
                prompt_text=prompt_text,
                encoded=encoded,
            )
            generation_start, generation_end = self._build_generation_span(input_ids)
            return input_ids, affix_start, affix_end, generation_start, generation_end

        if self.affix_type == "suffix":
            question_prompt_ids = self._tokenize_chat_messages(
                self._build_suffix_question_response_messages(question),
                add_generation_prompt=True,
            )
            generation_start, generation_end = self._build_generation_span(question_prompt_ids)
            suffix_affix_token_ids = affix_state["suffix_affix_token_ids"]
            suffix_affix_ids = list(suffix_affix_token_ids) if suffix_affix_token_ids else []

            if not suffix_affix_ids:
                return question_prompt_ids, 0, 0, generation_start, generation_end

            response_mask_ids = [self.mask_id] * self.gen_length
            input_ids = question_prompt_ids + response_mask_ids + suffix_affix_ids
            affix_start = generation_end
            affix_end = affix_start + len(suffix_affix_ids)
            return input_ids, affix_start, affix_end, generation_start, generation_end

        input_ids = self._tokenize_plain_prompt(question)
        generation_start, generation_end = self._build_generation_span(input_ids)
        return input_ids, 0, 0, generation_start, generation_end

    def _count_generated_tokens(self, token_ids: torch.Tensor) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            return int(token_ids.numel())
        return int((token_ids != pad_token_id).sum().item())

    def _decode_token_ids(self, token_ids) -> str:
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception as exc:
            return f"<decode failed: {exc}>"

    def _truncate_at_stop_tokens(self, text: str, stop_tokens) -> str:
        for stop_seq in stop_tokens or []:
            if stop_seq in text:
                text = text.split(stop_seq)[0]
        return text

    def _postprocess_generated_answer(self, generated_tokens, stop_tokens, req) -> str:
        if getattr(self, "model_label", None) == "dream_acache":
            generated_answer = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=False,
            )
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if isinstance(eos_token, str) and eos_token:
                generated_answer = generated_answer.split(eos_token)[0]
            generated_answer = self._truncate_at_stop_tokens(generated_answer, stop_tokens)
            return generated_answer

        generated_answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False,
        )
        generated_answer = self._truncate_at_stop_tokens(generated_answer, stop_tokens)

        if getattr(self, "model_label", None) == "llada_acache":
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids,
                skip_special_tokens=True,
            )
            return generated_answer

        return generated_answer

    def loglikelihood(self, requests):
        raise NotImplementedError("Use generate_until for ACache evaluation")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

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
                with open(save_path, "r", encoding="utf-8") as handle:
                    output = [json.loads(line) for line in handle]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")

        batched_requests = [[]]
        for i, req in enumerate(tqdm(requests, desc="Batching...")):
            if i < processed_count:
                continue
            batched_requests[-1].append(req)
            if len(batched_requests[-1]) == self.batch_size:
                batched_requests.append([])

        if len(batched_requests[-1]) == 0:
            batched_requests.pop()

        start_time = time.time()

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            batched_affix_info = []
            batched_generation_spans = []
            batched_affix_states = []
            batched_input_lengths = []
            max_len = 0

            for req in batch:
                affix_state = self._resolve_affix_state_for_request(req)
                if self.affix_type in {"prefix", "suffix", "infix"}:
                    question = self._extract_question_text(req)
                else:
                    question = req.args[0]
                input_ids, affix_start, affix_end, generation_start, generation_end = self._build_input_ids_with_affix(
                    question,
                    affix_state,
                )
                batched_input_ids.append(input_ids)
                batched_affix_info.append((affix_start, affix_end))
                batched_generation_spans.append((generation_start, generation_end))
                batched_affix_states.append(affix_state)
                batched_input_lengths.append(len(input_ids))
                max_len = max(max_len, len(input_ids))

            batched_input_ids = [
                torch.cat([
                    torch.full(
                        (1, max_len - len(input_ids)),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0),
                ], dim=1)
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)

            stop_tokens = batch[0].args[1]["until"]
            input_ids = batched_input_ids

            affix_start, affix_end = batched_affix_info[0]
            generation_start, generation_end = batched_generation_spans[0]
            affix_state = batched_affix_states[0]

            left_pad_first = max_len - batched_input_lengths[0]
            if left_pad_first > 0:
                affix_start += left_pad_first
                affix_end += left_pad_first
                generation_start += left_pad_first
                generation_end += left_pad_first

            has_affix = affix_end > affix_start
            suffix_infill_mode = self.affix_type == "suffix" and has_affix

            if has_affix:
                generated_answer, nfe = self._generate_with_affix_cache(
                    input_ids=input_ids,
                    affix_start=affix_start,
                    affix_end=affix_end,
                    generation_start=generation_start,
                    affix_state=affix_state,
                )
            else:
                generated_answer, nfe = self._generate_without_affix(input_ids)

            batched_generated_answer = []
            for i in range(len(generated_answer)):
                left_pad = max_len - batched_input_lengths[i]
                sample_generation_start, sample_generation_end = batched_generation_spans[i]
                sample_generation_start += left_pad
                sample_generation_end += left_pad
                if suffix_infill_mode:
                    generated_tokens = generated_answer[i][sample_generation_start:sample_generation_end]
                else:
                    generated_tokens = generated_answer[i][input_ids.shape[1]:]
                generated_answer_i = self._postprocess_generated_answer(
                    generated_tokens=generated_tokens,
                    stop_tokens=stop_tokens,
                    req=batch[i],
                )
                generated_answer_ids = torch.tensor(self.tokenizer(generated_answer_i)["input_ids"])
                if self.show_speed:
                    num_tokens += self._count_generated_tokens(generated_answer_ids)
                    num_nfe += nfe
                batched_generated_answer.append(generated_answer_i)

            output.extend(batched_generated_answer)

            if self.save_dir is not None:
                with open(save_path, "a", encoding="utf-8") as handle:
                    for generated_answer in batched_generated_answer:
                        handle.write(json.dumps(generated_answer, ensure_ascii=False) + "\n")

        end_time = time.time()

        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {end_time - start_time} seconds")
            print(f"Tokens per second: {num_tokens / (end_time - start_time) if end_time > start_time else 0}")
            print(f"Total NFE is {num_nfe}")

        return output

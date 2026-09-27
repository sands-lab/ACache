import importlib.util
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVAL_LLADA_PATH = ROOT / "eval_llada.py"


def _load_eval_llada_module(monkeypatch, dataset_rows=None):
    dataset_rows = [] if dataset_rows is None else dataset_rows
    accelerate = types.ModuleType("accelerate")

    class Accelerator:
        num_processes = 1

    accelerate.Accelerator = Accelerator

    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    torch.device = lambda name: name
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.manual_seed = lambda seed: None

    datasets = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        return list(dataset_rows)

    datasets.load_dataset = load_dataset

    lm_eval = types.ModuleType("lm_eval")
    lm_eval.__file__ = str(ROOT / "fake_lm_eval" / "__init__.py")
    lm_eval.__path__ = []

    lm_eval_main = types.ModuleType("lm_eval.__main__")
    lm_eval_main.cli_evaluate = lambda *args, **kwargs: None
    lm_eval_main.parse_eval_args = lambda parser: None
    lm_eval_main.setup_parser = lambda: None

    lm_eval_api = types.ModuleType("lm_eval.api")
    lm_eval_api.__path__ = []

    lm_eval_api_model = types.ModuleType("lm_eval.api.model")

    class LM:
        pass

    lm_eval_api_model.LM = LM

    lm_eval_api_registry = types.ModuleType("lm_eval.api.registry")
    lm_eval_api_registry.register_model = lambda name: (lambda cls: cls)

    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda iterable, *args, **kwargs: iterable

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = type("AutoConfig", (), {})
    transformers.AutoTokenizer = type("AutoTokenizer", (), {})

    config = types.ModuleType("config")
    config.Config = type("Config", (), {})

    model = types.ModuleType("model")
    model.__path__ = []

    model_modeling_llada = types.ModuleType("model.modeling_llada")
    model_modeling_llada.LLaDAModelLM = type("LLaDAModelLM", (), {})

    model_runner = types.ModuleType("model_runner")
    model_runner.ModelRunner = type("ModelRunner", (), {})

    sequence = types.ModuleType("sequence")
    sequence.Sequence = type("Sequence", (), {})

    utils = types.ModuleType("utils")
    utils.set_seed = lambda seed: None

    stubbed_modules = {
        "accelerate": accelerate,
        "torch": torch,
        "datasets": datasets,
        "lm_eval": lm_eval,
        "lm_eval.__main__": lm_eval_main,
        "lm_eval.api": lm_eval_api,
        "lm_eval.api.model": lm_eval_api_model,
        "lm_eval.api.registry": lm_eval_api_registry,
        "tqdm": tqdm,
        "transformers": transformers,
        "config": config,
        "model": model,
        "model.modeling_llada": model_modeling_llada,
        "model_runner": model_runner,
        "sequence": sequence,
        "utils": utils,
    }

    for name, module in stubbed_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = f"eval_llada_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, EVAL_LLADA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generic_fewshot_uses_hf_dataset_random_sample(monkeypatch):
    module = _load_eval_llada_module(
        monkeypatch,
        dataset_rows=[
            {"question": "Question 0", "answer": "Answer 0"},
            {"question": "Question 1", "answer": "Answer 1"},
            {"question": "Question 2", "answer": "Answer 2"},
        ],
    )
    monkeypatch.setattr(module.random, "sample", lambda population, count: [2, 0])

    harness = object.__new__(module.LLaDAEvalHarness)
    harness.fewshot_num_examples = 2
    harness.fewshot_dataset_path = "gsm8k"
    harness.fewshot_dataset_name = "main"
    harness.fewshot_split = "train"
    harness.fewshot_question_key = "question"
    harness.fewshot_answer_key = "answer"
    harness.prompt_style = module.GSM8K_PROMPT_STYLE
    harness.sampled_fewshot_examples = []

    messages = harness._build_prefix_fewshot_messages()

    assert harness.sampled_fewshot_examples == [
        {"index": 2, "question": "Question 2", "answer": "Answer 2"},
        {"index": 0, "question": "Question 0", "answer": "Answer 0"},
    ]
    assert messages == [
        {
            "role": "user",
            "content": "Question: Question 2\nLet's think step by step.\nAnswer:",
        },
        {"role": "assistant", "content": "Answer 2"},
        {
            "role": "user",
            "content": "Question: Question 0\nLet's think step by step.\nAnswer:",
        },
        {"role": "assistant", "content": "Answer 0"},
    ]


def test_gsm8k_infix_final_answer_prompt_marks_demonstrations(monkeypatch):
    module = _load_eval_llada_module(monkeypatch)
    harness = object.__new__(module.LLaDAEvalHarness)
    harness.prompt_style = module.GSM8K_PROMPT_STYLE

    assert harness._format_infix_final_answer_prompt() == (
        "The examples above are demonstrations only.\n"
        "Let's come back to the question at the beginning and think step by step.\n"
        "Answer:"
    )


def test_mbpp_fewshot_formats_code_prompt_and_done_marker(monkeypatch):
    dataset_rows = [
        {
            "task_id": 1,
            "text": "Write a function that returns 1.",
            "code": "def one():\n    return 1",
            "test_list": [
                "assert one() == 1",
                "assert one() + 1 == 2",
                "assert isinstance(one(), int)",
            ],
        },
        {
            "task_id": 2,
            "text": "Write a function that returns 2.",
            "code": "def two():\n    return 2",
            "test_list": [
                "assert two() == 2",
                "assert two() + 1 == 3",
                "assert isinstance(two(), int)",
            ],
        },
    ]
    module = _load_eval_llada_module(
        monkeypatch,
        dataset_rows=dataset_rows,
    )
    monkeypatch.setattr(module.random, "sample", lambda population, count: [1])

    harness = object.__new__(module.LLaDAEvalHarness)
    harness.fewshot_num_examples = 1
    harness.fewshot_dataset_path = "google-research-datasets/mbpp"
    harness.fewshot_dataset_name = "full"
    harness.fewshot_split = "prompt"
    harness.fewshot_question_key = "text"
    harness.fewshot_answer_key = "code"
    harness.prompt_style = harness._resolve_prompt_style()
    harness.sampled_fewshot_examples = []

    messages = harness._build_prefix_fewshot_messages()

    assert harness.sampled_fewshot_examples == [
        {
            "index": 1,
            "question": {
                "text": "Write a function that returns 2.",
                "test_list": [
                    "assert two() == 2",
                    "assert two() + 1 == 3",
                    "assert isinstance(two(), int)",
                ],
            },
            "answer": "def two():\n    return 2\n[DONE]",
            "task_id": 2,
        }
    ]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == (
        "You are an expert Python programmer, and here is your task: "
        "Write a function that returns 2. "
        "Your code should pass these tests:\n\n"
        "assert two() == 2\n"
        "assert two() + 1 == 3\n"
        "assert isinstance(two(), int)\n"
        "[BEGIN]\n"
    )
    assert messages[1] == {"role": "assistant", "content": "def two():\n    return 2\n[DONE]"}


def test_babilong_fewshot_preserves_story_and_location_answer(monkeypatch):
    module = _load_eval_llada_module(
        monkeypatch,
        dataset_rows=[
            {
                "input": "Mary went to the kitchen.",
                "question": "Where is Mary?",
                "target": "kitchen",
            }
        ],
    )
    monkeypatch.setattr(module.random, "sample", lambda population, count: [0])

    harness = object.__new__(module.LLaDAEvalHarness)
    harness.seed = 0
    harness.fewshot_num_examples = 1
    harness.fewshot_dataset_path = "RMT-team/babilong-1k-samples"
    harness.fewshot_dataset_name = "0k"
    harness.fewshot_split = "qa1"
    harness.fewshot_question_key = "question"
    harness.fewshot_answer_key = "target"
    harness.prompt_style = harness._resolve_prompt_style()
    harness.sampled_fewshot_examples = []

    messages = harness._build_prefix_fewshot_messages()

    assert harness.prompt_style == module.BABILONG_QA1_PROMPT_STYLE
    assert messages == [
        {
            "role": "user",
            "content": (
                "Story:\nMary went to the kitchen.\n"
                "Question: Where is Mary?\nAnswer with one location:"
            ),
        },
        {"role": "assistant", "content": "kitchen"},
    ]


def test_plain_shared_prefix_is_exact_cached_request_prefix(monkeypatch):
    module = _load_eval_llada_module(monkeypatch)

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": [ord(char) for char in text]}

    harness = object.__new__(module.LLaDAEvalHarness)
    harness.tokenizer = Tokenizer()
    harness.fewshot_num_examples = 0
    harness.shared_prefix_text = "SYSTEM"
    harness.shared_prefix_extra_text = "FORMAT"
    harness.shared_prefix_use_chat_template = False
    harness.prompt_style = module.RAW_PROMPT_STYLE
    harness.sampled_fewshot_examples = []

    harness.prefix_fewshot_messages = harness._build_prefix_fewshot_messages()
    harness.prefix_affix_token_ids = harness._build_prefix_affix_token_ids()
    harness.prefix_affix_len = len(harness.prefix_affix_token_ids)
    input_ids = harness._build_prefix_input_ids("Question and choices")

    prefix_text = "SYSTEM\n\nFORMAT\n\n"
    assert harness.prefix_affix_token_ids == [ord(char) for char in prefix_text]
    assert input_ids[:harness.prefix_affix_len] == harness.prefix_affix_token_ids
    assert input_ids == [ord(char) for char in f"{prefix_text}Question and choices"]


def test_raw_prompt_style_preserves_arc_request(monkeypatch):
    module = _load_eval_llada_module(monkeypatch)
    harness = object.__new__(module.LLaDAEvalHarness)
    harness.prompt_style = module.RAW_PROMPT_STYLE
    request = SimpleNamespace(
        args=("Question with A/B/C/D choices", {}),
        doc={"question": "Question without choices"},
    )

    assert harness._extract_question_text(request) == "Question with A/B/C/D choices"
    assert harness._format_query_prompt(request.args[0]) == request.args[0]


@pytest.mark.parametrize("slow_tokenizer", [False, True])
def test_infix_and_suffix_affix_spans_are_stable_across_questions(monkeypatch, slow_tokenizer):
    module = _load_eval_llada_module(monkeypatch)

    class CharacterTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize=False):
            text = "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in messages
            )
            if add_generation_prompt:
                text += "<assistant>"
            return text

        def __call__(self, text, return_offsets_mapping=False):
            if slow_tokenizer and return_offsets_mapping:
                raise NotImplementedError("Slow tokenizer has no offsets")
            result = {"input_ids": [ord(char) for char in text]}
            if return_offsets_mapping:
                result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
            return result

        def decode(self, token_ids, skip_special_tokens=False):
            return "".join(chr(token_id) for token_id in token_ids)

    harness = object.__new__(module.LLaDAEvalHarness)
    harness.tokenizer = CharacterTokenizer()
    harness.prompt_style = module.GSM8K_PROMPT_STYLE
    harness.mask_id = ord("#")
    harness.gen_length = 4
    harness.sampled_fewshot_examples = [
        {"question": "Two plus three?", "answer": "The answer is 5"},
        {"question": "Four plus six?", "answer": "The answer is 10"},
    ]
    harness.prefix_fewshot_messages = []
    for example in harness.sampled_fewshot_examples:
        harness.prefix_fewshot_messages.extend([
            {"role": "user", "content": harness._format_query_prompt(example["question"])},
            {"role": "assistant", "content": example["answer"]},
        ])

    harness.infix_examples_text = harness._build_infix_examples_text()
    harness.infix_affix_token_ids = harness._build_infix_affix_token_ids()
    harness.suffix_affix_token_ids = harness._build_suffix_affix_token_ids()

    for affix_type, expected in (
        ("infix", harness.infix_affix_token_ids),
        ("suffix", harness.suffix_affix_token_ids),
    ):
        harness.affix_type = affix_type
        for question in ("Short?", "A substantially longer question?"):
            input_ids, start, end, generation_start = harness._build_input_ids_with_affix(question)
            assert input_ids[start:end] == expected
            if affix_type == "suffix":
                assert input_ids[generation_start:generation_start + harness.gen_length] == [
                    harness.mask_id
                ] * harness.gen_length


def test_mbpp_infix_affix_probe_uses_task_shaped_question(monkeypatch):
    module = _load_eval_llada_module(monkeypatch)

    class CharacterTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize=False):
            text = "".join(message["content"] for message in messages)
            return text + ("<assistant>" if add_generation_prompt else "")

        def __call__(self, text, return_offsets_mapping=False):
            result = {"input_ids": [ord(char) for char in text]}
            if return_offsets_mapping:
                result["offset_mapping"] = [
                    (index, index + 1) for index in range(len(text))
                ]
            return result

    question = {
        "text": "Return one.",
        "test_list": [
            "assert one() == 1",
            "assert one() + 1 == 2",
            "assert isinstance(one(), int)",
        ],
    }
    harness = object.__new__(module.LLaDAEvalHarness)
    harness.tokenizer = CharacterTokenizer()
    harness.prompt_style = module.MBPP_PROMPT_STYLE
    harness.sampled_fewshot_examples = [
        {"question": question, "answer": "def one():\n    return 1\n[DONE]"}
    ]
    harness.infix_examples_text = harness._build_infix_examples_text()

    affix_ids = harness._build_infix_affix_token_ids()

    assert "".join(chr(token_id) for token_id in affix_ids) == harness.infix_examples_text


def test_profile_timing_mark_is_noop_when_disabled(monkeypatch):
    module = _load_eval_llada_module(monkeypatch)
    harness = object.__new__(module.LLaDAEvalHarness)
    harness.profile_timing = False
    harness.device = SimpleNamespace(type="cuda")

    monkeypatch.setattr(
        module.time,
        "perf_counter",
        lambda: pytest.fail("perf_counter should not run when profile_timing is disabled"),
    )

    assert harness._profile_timing_mark() is None


@pytest.mark.parametrize("model_args", ["", {}])
def test_prepare_cli_args_only_maps_num_fewshot_and_seed(monkeypatch, model_args):
    module = _load_eval_llada_module(monkeypatch)
    args = SimpleNamespace(
        tasks=["gsm8k"],
        num_fewshot=4,
        seed=[7, None, None, None],
        model_args=model_args,
    )

    monkeypatch.setattr(module, "setup_parser", lambda: object())
    monkeypatch.setattr(module, "parse_eval_args", lambda parser: args)

    parsed = module._prepare_cli_args_for_custom_fewshot()

    assert parsed.num_fewshot == 0
    assert parsed.seed == [None, None, None, None]
    if isinstance(parsed.model_args, dict):
        assert parsed.model_args["fewshot_num_examples"] == 4
        assert parsed.model_args["seed"] == 7
        assert "fewshot_dataset_path" not in parsed.model_args
        assert "fewshot_question_key" not in parsed.model_args
        assert "fewshot_answer_key" not in parsed.model_args
    else:
        assert "fewshot_num_examples=4" in parsed.model_args
        assert "seed=7" in parsed.model_args
        assert "fewshot_dataset_path=" not in parsed.model_args
        assert "fewshot_question_key=" not in parsed.model_args
        assert "fewshot_answer_key=" not in parsed.model_args


@pytest.mark.parametrize("model_args", ["", {}])
def test_prepare_cli_args_autoselects_mbpp_defaults(monkeypatch, model_args):
    module = _load_eval_llada_module(monkeypatch)
    args = SimpleNamespace(
        tasks=["mbpp"],
        num_fewshot=3,
        seed=[11, None, None, None],
        model_args=model_args,
    )

    monkeypatch.setattr(module, "setup_parser", lambda: object())
    monkeypatch.setattr(module, "parse_eval_args", lambda parser: args)

    parsed = module._prepare_cli_args_for_custom_fewshot()

    assert parsed.num_fewshot == 0
    assert parsed.seed == [None, None, None, None]
    if isinstance(parsed.model_args, dict):
        assert parsed.model_args["fewshot_dataset_path"] == "google-research-datasets/mbpp"
        assert parsed.model_args["fewshot_dataset_name"] == "full"
        assert parsed.model_args["fewshot_split"] == "prompt"
        assert parsed.model_args["fewshot_question_key"] == "text"
        assert parsed.model_args["fewshot_answer_key"] == "code"
    else:
        assert "fewshot_dataset_path=google-research-datasets/mbpp" in parsed.model_args
        assert "fewshot_dataset_name=full" in parsed.model_args
        assert "fewshot_split=prompt" in parsed.model_args
        assert "fewshot_question_key=text" in parsed.model_args
        assert "fewshot_answer_key=code" in parsed.model_args


@pytest.mark.parametrize("model_args", ["", {}])
def test_prepare_cli_args_autoselects_babilong_defaults(monkeypatch, model_args):
    module = _load_eval_llada_module(monkeypatch)
    args = SimpleNamespace(
        tasks=["babilong"],
        num_fewshot=2,
        seed=[5, None, None, None],
        model_args=model_args,
        include_path=None,
    )

    monkeypatch.setattr(module, "setup_parser", lambda: object())
    monkeypatch.setattr(module, "parse_eval_args", lambda parser: args)

    parsed = module._prepare_cli_args_for_custom_fewshot()

    assert str(parsed.include_path).endswith("lm_eval_tasks")
    if isinstance(parsed.model_args, dict):
        assert parsed.model_args["fewshot_dataset_path"] == "RMT-team/babilong-1k-samples"
        assert parsed.model_args["fewshot_dataset_name"] == "0k"
        assert parsed.model_args["fewshot_split"] == "qa1"
        assert parsed.model_args["fewshot_question_key"] == "question"
        assert parsed.model_args["fewshot_answer_key"] == "target"
    else:
        assert "fewshot_dataset_path=RMT-team/babilong-1k-samples" in parsed.model_args
        assert "fewshot_dataset_name=0k" in parsed.model_args
        assert "fewshot_split=qa1" in parsed.model_args
        assert "fewshot_question_key=question" in parsed.model_args
        assert "fewshot_answer_key=target" in parsed.model_args

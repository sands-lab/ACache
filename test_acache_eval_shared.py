import contextlib
import subprocess
import random
import sys
import types
import unittest
from types import SimpleNamespace

try:
    import lm_eval.__main__  # noqa: F401
except ModuleNotFoundError:
    lm_eval_module = types.ModuleType("lm_eval")
    lm_eval_module.__path__ = []
    lm_eval_main_module = types.ModuleType("lm_eval.__main__")
    lm_eval_main_module.parse_eval_args = lambda parser: None
    lm_eval_main_module.setup_parser = lambda: None
    lm_eval_module.__main__ = lm_eval_main_module
    sys.modules["lm_eval"] = lm_eval_module
    sys.modules["lm_eval.__main__"] = lm_eval_main_module

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = object
    torch_module.long = int
    torch_module.manual_seed = lambda seed: None
    torch_module.tensor = lambda *args, **kwargs: None
    torch_module.inference_mode = contextlib.nullcontext
    torch_module.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=False, benchmark=False)
    )
    sys.modules["torch"] = torch_module

try:
    import datasets  # noqa: F401
except ModuleNotFoundError:
    datasets_module = types.ModuleType("datasets")

    class DummyDataset(list):
        @staticmethod
        def from_file(path):
            raise NotImplementedError(path)

    datasets_module.Dataset = DummyDataset
    datasets_module.load_dataset = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError((args, kwargs)))
    sys.modules["datasets"] = datasets_module

try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

from acache_eval_shared import (
    ACacheEvalHarnessMixin,
    GSM8K_PROMPT_STYLE,
    MBPP_PROMPT_STYLE,
    BABILONG_QA1_PREFIX_QUERY_TEMPLATE,
    BABILONG_QA1_PROMPT_STYLE,
    prepare_cli_args_for_custom_fewshot,
    tasks_are_mbpp,
    tasks_are_babilong_qa1,
    tasks_include_humaneval,
)


MBPP_PROMPT_ROWS = [
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
    {
        "task_id": 3,
        "text": "Write a function that returns 3.",
        "code": "def three():\n    return 3",
        "test_list": [
            "assert three() == 3",
            "assert three() + 1 == 4",
            "assert isinstance(three(), int)",
        ],
    },
    {
        "task_id": 4,
        "text": "Write a function that returns 4.",
        "code": "def four():\n    return 4",
        "test_list": [
            "assert four() == 4",
            "assert four() + 1 == 5",
            "assert isinstance(four(), int)",
        ],
    },
]

BABILONG_QA1_ROWS = [
    {
        "question": "Where is John?",
        "target": "hallway",
        "input": "John travelled to the hallway.",
    },
]


class DummyMBPPHarness(ACacheEvalHarnessMixin):
    def __init__(self, fewshot_num_examples=3):
        self.fewshot_num_examples = fewshot_num_examples
        self.fewshot_dataset_path = "google-research-datasets/mbpp"
        self.fewshot_dataset_name = "full"
        self.fewshot_split = "prompt"
        self.fewshot_question_key = "text"
        self.fewshot_answer_key = "code"
        self.prompt_style = self._resolve_prompt_style()

    def _load_fewshot_rows(self):
        return MBPP_PROMPT_ROWS, "mbpp/full:prompt"


class DummyBabilongQAW1Harness(ACacheEvalHarnessMixin):
    def __init__(self, fewshot_num_examples=1):
        self.fewshot_num_examples = fewshot_num_examples
        self.fewshot_dataset_path = "RMT-team/babilong-1k-samples"
        self.fewshot_dataset_name = "0k"
        self.fewshot_split = "qa1"
        self.fewshot_question_key = "question"
        self.fewshot_answer_key = "target"
        self.prompt_style = self._resolve_prompt_style()

    def _load_fewshot_rows(self):
        return BABILONG_QA1_ROWS, "RMT-team/babilong-1k-samples/0k:qa1"


class DummyTokenizer:
    bos_token = "<bos>"

    def __call__(self, text):
        return {"input_ids": list(range(len(text)))}


class DummyChatTokenizer(DummyTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool):
        del tokenize
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered = f"{rendered}|assistant:"
        return rendered


class BoundarySensitiveChatTokenizer(DummyChatTokenizer):
    def __init__(self):
        self._token_to_id = {}

    def __call__(self, text, return_offsets_mapping=False):
        tokens = []
        offsets = []
        idx = 0
        while idx < len(text):
            start = idx
            if text.startswith("[DONE]\n\nLet's", idx):
                piece = "[DONE]\n\nLet's"
                idx += len(piece)
            else:
                piece = text[idx]
                idx += 1
            if piece not in self._token_to_id:
                self._token_to_id[piece] = len(self._token_to_id) + 1
            tokens.append(self._token_to_id[piece])
            offsets.append((start, idx))
        if return_offsets_mapping:
            return {"input_ids": tokens, "offset_mapping": offsets}
        return {"input_ids": tokens}


class BoundarySensitiveSlowChatTokenizer(BoundarySensitiveChatTokenizer):
    def __call__(self, text, return_offsets_mapping=False):
        if return_offsets_mapping:
            raise NotImplementedError("offset mapping is unavailable for this slow tokenizer")
        return super().__call__(text, return_offsets_mapping=False)


class DummyRawPromptHarness(ACacheEvalHarnessMixin):
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def _uses_chat_template_for_prompts(self) -> bool:
        return False

    def _prepare_plain_prompt_text(self, prompt_text: str) -> str:
        return f"{self.tokenizer.bos_token}{prompt_text}"


class DummyChatPromptHarness(ACacheEvalHarnessMixin):
    def __init__(self):
        self.tokenizer = DummyChatTokenizer()

    def _uses_chat_template_for_prompts(self) -> bool:
        return True


class DummyChatNoneHarness(DummyChatPromptHarness):
    def __init__(self):
        super().__init__()
        self.affix_type = "none"
        self.gen_length = 8

    def _build_input_ids_with_affix(self, question: str, affix_state):
        del affix_state
        input_ids = self._tokenize_chat_messages(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
        )
        generation_start = len(input_ids)
        generation_end = generation_start + self.gen_length
        return input_ids, 0, 0, generation_start, generation_end


class DummyAliasedMBPPHarness(ACacheEvalHarnessMixin):
    def __init__(self, fewshot_num_examples=3):
        self.fewshot_num_examples = fewshot_num_examples
        self.fewshot_dataset_path = "lm_eval:mbpp"
        self.fewshot_dataset_name = "full"
        self.fewshot_split = "prompt"
        self.fewshot_question_key = "text"
        self.fewshot_answer_key = "code"
        self.prompt_style = self._resolve_prompt_style()
        self.loaded_args = None

    def _load_mbpp_rows(self, dataset_path: str, dataset_name: str, split_name: str):
        self.loaded_args = (dataset_path, dataset_name, split_name)
        return MBPP_PROMPT_ROWS


class DummyBoundarySensitiveMBPPHarness(DummyMBPPHarness):
    def __init__(self, fewshot_num_examples=1):
        super().__init__(fewshot_num_examples=fewshot_num_examples)
        self.tokenizer = BoundarySensitiveChatTokenizer()
        self.affix_type = "infix"
        self.gen_length = 8


class DummyBoundarySensitiveSlowMBPPHarness(DummyMBPPHarness):
    def __init__(self, fewshot_num_examples=1):
        super().__init__(fewshot_num_examples=fewshot_num_examples)
        self.tokenizer = BoundarySensitiveSlowChatTokenizer()
        self.affix_type = "infix"
        self.gen_length = 8


class DummyPostprocessTokenizer:
    eos_token = "<eos>"

    def __call__(self, text):
        pieces = []
        idx = 0
        specials = ("<eos>", "<special>", "<pad>")
        while idx < len(text):
            matched = None
            for token in specials:
                if text.startswith(token, idx):
                    matched = token
                    break
            if matched is not None:
                pieces.append(matched)
                idx += len(matched)
                continue

            next_special_positions = [text.find(token, idx) for token in specials]
            next_special_positions = [pos for pos in next_special_positions if pos != -1]
            next_idx = min(next_special_positions) if next_special_positions else len(text)
            pieces.append(text[idx:next_idx])
            idx = next_idx

        return {"input_ids": [piece for piece in pieces if piece]}

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        del kwargs
        if isinstance(token_ids, str):
            tokens = self(token_ids)["input_ids"]
        elif isinstance(token_ids, (list, tuple)):
            tokens = list(token_ids)
        else:
            tokens = [token_ids]

        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {"<eos>", "<special>", "<pad>"}]
        return "".join(tokens)


class DummyPostprocessHarness(ACacheEvalHarnessMixin):
    def __init__(self, model_label, is_instruct=False, prompt_style=GSM8K_PROMPT_STYLE):
        self.model_label = model_label
        self.is_instruct = is_instruct
        self.prompt_style = prompt_style
        self.tokenizer = DummyPostprocessTokenizer()


class ACacheEvalSharedTests(unittest.TestCase):
    def _sample_prefix_messages_and_examples(self, harness):
        random_state = random.getstate()
        random.seed(0)
        try:
            return harness._build_prefix_fewshot_messages()
        finally:
            random.setstate(random_state)

    def _build_boundary_sensitive_infix_state(self, harness, cache_key):
        prefix_messages, sampled_examples = self._sample_prefix_messages_and_examples(harness)
        state = harness._make_empty_affix_state(cache_key)
        state["sampled_fewshot_examples"] = sampled_examples
        state["prefix_fewshot_messages"] = prefix_messages
        state["infix_plaintext_messages"] = harness._build_infix_plaintext_messages(sampled_examples)
        return state, sampled_examples

    def _old_probe_affix_ids(self, harness, state, probe_question):
        probe_header = [{
            "role": "user",
            "content": harness._build_infix_user_content(
                question=probe_question,
                include_examples=False,
                include_answer_prompt=False,
                affix_state=state,
            ),
        }]
        probe_header_ids = harness._tokenize_chat_messages(
            probe_header,
            add_generation_prompt=False,
        )
        probe_header_plus_examples = [{
            "role": "user",
            "content": harness._build_infix_user_content(
                question=probe_question,
                include_examples=True,
                include_answer_prompt=False,
                affix_state=state,
            ),
        }]
        probe_header_plus_examples_ids = harness._tokenize_chat_messages(
            probe_header_plus_examples,
            add_generation_prompt=False,
        )
        return probe_header_plus_examples_ids[len(probe_header_ids):]

    def _assert_boundary_sensitive_infix_probe_alignment(self, harness, cache_key):
        state, sampled_examples = self._build_boundary_sensitive_infix_state(harness, cache_key)
        probe_question = harness._build_affix_probe_question(sampled_examples)
        old_affix_ids = self._old_probe_affix_ids(harness, state, probe_question)

        probe_input_ids, probe_affix_start, probe_affix_end, _, _ = harness._build_input_ids_with_affix(
            probe_question,
            state,
        )
        new_affix_ids = probe_input_ids[probe_affix_start:probe_affix_end]

        question = {
            "text": "Write a function that returns 5.",
            "test_list": [
                "assert five() == 5",
                "assert five() + 1 == 6",
                "assert isinstance(five(), int)",
            ],
        }
        input_ids, affix_start, affix_end, _, _ = harness._build_input_ids_with_affix(
            question,
            state,
        )
        actual_affix_ids = input_ids[affix_start:affix_end]

        self.assertNotEqual(old_affix_ids, actual_affix_ids)
        self.assertEqual(new_affix_ids, actual_affix_ids)

    def test_tasks_are_mbpp(self):
        self.assertTrue(tasks_are_mbpp("mbpp"))
        self.assertTrue(tasks_are_mbpp(["mbpp"]))
        self.assertFalse(tasks_are_mbpp("mbpp_plus"))
        self.assertFalse(tasks_are_mbpp("gsm8k"))

    def test_tasks_are_babilong_qa1(self):
        self.assertTrue(tasks_are_babilong_qa1("babilong"))
        self.assertFalse(tasks_are_babilong_qa1("babilong_qa1"))

    def test_tasks_include_humaneval(self):
        self.assertTrue(tasks_include_humaneval("humaneval"))
        self.assertTrue(tasks_include_humaneval(["gsm8k", "humaneval_plus"]))
        self.assertFalse(tasks_include_humaneval("gsm8k"))
        self.assertFalse(tasks_include_humaneval(["mbpp"]))

    def test_shell_applies_babilong_qa1_generation_length(self):
        script = (
            "source ./acache_anchor_ratio_common.sh\n"
            "printf 'task=%s\\n' \"$(acache_normalize_tasks ' babilong ')\"\n"
            "printf 'task_args=%s\\n' \"$(acache_apply_generation_length_defaults "
            "'babilong' 'model_path=x,gen_length=256,steps=256,block_length=32')\"\n"
        )
        result = subprocess.run(
            ["bash", "-lc", script],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("task=babilong", result.stdout)
        self.assertIn("task_args=model_path=x,gen_length=8,steps=8,block_length=8", result.stdout)

    def test_mbpp_alias_loader_uses_prompt_split_rows(self):
        harness = DummyAliasedMBPPHarness()
        rows, desc = harness._load_fewshot_rows()
        self.assertEqual(desc, "lm_eval:mbpp/full:prompt")
        self.assertEqual(harness.loaded_args, ("google-research-datasets/mbpp", "full", "prompt"))
        self.assertEqual([row["task_id"] for row in rows], [1, 2, 3, 4])

    def test_mbpp_prompt_style_and_query_format(self):
        harness = DummyMBPPHarness()
        self.assertEqual(harness.prompt_style, MBPP_PROMPT_STYLE)

        prompt = harness._format_query_prompt(
            {
                "text": "Write a function that returns 1.",
                "test_list": [
                    "assert one() == 1",
                    "assert one() + 1 == 2",
                    "assert isinstance(one(), int)",
                ],
            }
        )
        expected = (
            "You are an expert Python programmer, and here is your task: "
            "Write a function that returns 1. "
            "Your code should pass these tests:\n\n"
            "assert one() == 1\n"
            "assert one() + 1 == 2\n"
            "assert isinstance(one(), int)\n"
            "[BEGIN]\n"
        )
        self.assertEqual(prompt, expected)
        self.assertIn("Example(s):", harness._format_infix_query_prompt(MBPP_PROMPT_ROWS[0]))
        self.assertEqual(
            harness._format_infix_final_answer_prompt(),
            "Let's come back to the task in the beginning and write the Python code.\n"
            "Do not include any explanation, comments outside the code, markdown fences, or test cases.\n[BEGIN]\n",
        )

    def test_mbpp_infix_final_answer_prompt_forbids_prose_and_markdown(self):
        harness = DummyMBPPHarness()

        prompt = harness._format_infix_final_answer_prompt()

        self.assertIn("write the Python code", prompt)
        self.assertIn("Do not include any explanation", prompt)
        self.assertIn("markdown fences", prompt)
        self.assertIn("test cases", prompt)
        self.assertTrue(prompt.endswith("[BEGIN]\n"))

    def test_mbpp_prompt_split_fewshot_uses_seeded_random_sample_and_done_marker(self):
        harness = DummyMBPPHarness(fewshot_num_examples=3)
        messages, sampled_examples = self._sample_prefix_messages_and_examples(harness)

        self.assertEqual(len(messages), 6)
        self.assertEqual([example["task_id"] for example in sampled_examples], [4, 2, 1])
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Write a function that returns 4.", messages[0]["content"])
        self.assertIn("assert four() == 4", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertTrue(messages[1]["content"].endswith("[DONE]"))

    def test_prepare_cli_args_autoselects_babilong_defaults(self):
        import acache_eval_shared

        original_setup_parser = acache_eval_shared.setup_parser
        original_parse_eval_args = acache_eval_shared.parse_eval_args
        args = SimpleNamespace(
            tasks="babilong",
            model_args="pretrained=facebook/opt-350m,steps=32,block_length=8",
            num_fewshot=2,
            seed=[0, None, None, None],
            output_path="auto",
        )
        acache_eval_shared.setup_parser = lambda: None
        acache_eval_shared.parse_eval_args = lambda parser: args
        try:
            prepared = prepare_cli_args_for_custom_fewshot()
        finally:
            acache_eval_shared.setup_parser = original_setup_parser
            acache_eval_shared.parse_eval_args = original_parse_eval_args

        self.assertEqual(prepared.tasks, "babilong")
        self.assertTrue(str(prepared.include_path).endswith("lm_eval_tasks"))
        self.assertIn("fewshot_dataset_path=RMT-team/babilong-1k-samples", prepared.model_args)
        self.assertIn("fewshot_dataset_name=0k", prepared.model_args)
        self.assertIn("fewshot_split=qa1", prepared.model_args)
        self.assertIn("fewshot_question_key=question", prepared.model_args)
        self.assertIn("fewshot_answer_key=target", prepared.model_args)

    def test_babilong_prompt_style_and_query_format(self):
        harness = DummyBabilongQAW1Harness()
        self.assertEqual(harness.prompt_style, BABILONG_QA1_PROMPT_STYLE)

        prompt = harness._format_query_prompt({
            "story": "John is in the hallway.",
            "question": "Where is John?",
        })
        expected = BABILONG_QA1_PREFIX_QUERY_TEMPLATE.format(
            story="John is in the hallway.",
            question="Where is John?",
        )
        self.assertEqual(prompt, expected)
        infix_prompt = harness._format_infix_query_prompt({
            "story": "John is in the hallway.",
            "question": "Where is John?",
        })
        self.assertIn("Story:\nJohn is in the hallway.", infix_prompt)

    def test_extract_question_text_for_babilong_qa1_uses_story_and_question(self):
        harness = DummyBabilongQAW1Harness()
        req = SimpleNamespace(
            doc=BABILONG_QA1_ROWS[0],
            args=["unused"],
            task_name="babilong",
        )

        question = harness._extract_question_text(req)
        self.assertEqual(question["story"], BABILONG_QA1_ROWS[0]["input"])
        self.assertEqual(question["question"], BABILONG_QA1_ROWS[0]["question"])

    def test_mbpp_infix_probe_question_keeps_test_list(self):
        harness = DummyMBPPHarness(fewshot_num_examples=3)
        _, sampled_examples = self._sample_prefix_messages_and_examples(harness)
        probe_question = harness._build_affix_probe_question(sampled_examples)

        self.assertEqual(probe_question["text"], "__acache_probe__")
        self.assertEqual(
            probe_question["test_list"],
            [
                "assert four() == 4",
                "assert four() + 1 == 5",
                "assert isinstance(four(), int)",
            ],
        )

    def test_mbpp_infix_probe_uses_full_prompt_slice_for_boundary_sensitive_tokenization(self):
        harness = DummyBoundarySensitiveMBPPHarness(fewshot_num_examples=1)
        self._assert_boundary_sensitive_infix_probe_alignment(harness, "boundary")

    def test_mbpp_infix_probe_uses_decoded_prefix_fallback_for_slow_tokenizer(self):
        harness = DummyBoundarySensitiveSlowMBPPHarness(fewshot_num_examples=1)
        self._assert_boundary_sensitive_infix_probe_alignment(harness, "boundary-slow")

    def test_extract_question_text_for_mbpp_uses_doc_tests(self):
        harness = DummyMBPPHarness()
        req = SimpleNamespace(
            doc={
                "text": "Write a function that returns 5.",
                "test_list": [
                    "assert five() == 5",
                    "assert five() + 1 == 6",
                    "assert isinstance(five(), int)",
                ],
            },
            args=["unused"],
            task_name="mbpp",
        )

        question = harness._extract_question_text(req)
        self.assertEqual(question["text"], "Write a function that returns 5.")
        self.assertEqual(
            question["test_list"],
            [
                "assert five() == 5",
                "assert five() + 1 == 6",
                "assert isinstance(five(), int)",
            ],
        )

    def test_dream_postprocess_matches_fast_dllm_eos_then_until_truncation(self):
        harness = DummyPostprocessHarness(model_label="dream_acache")
        req = SimpleNamespace(doc={}, task_name="mbpp")

        answer = harness._postprocess_generated_answer(
            generated_tokens=["def foo():\n    return 1", "<eos>", "[DONE]", "ignored"],
            stop_tokens=["[DONE]"],
            req=req,
        )

        self.assertEqual(answer, "def foo():\n    return 1")

    def test_llada_postprocess_matches_fast_dllm_until_then_special_cleanup(self):
        harness = DummyPostprocessHarness(model_label="llada_acache")
        req = SimpleNamespace(doc={}, task_name="mbpp")

        answer = harness._postprocess_generated_answer(
            generated_tokens=["def foo():\n    return 1", "<special>", "[DONE]", "<pad>"],
            stop_tokens=["[DONE]"],
            req=req,
        )

        self.assertEqual(answer, "def foo():\n    return 1")

    def test_raw_prompt_rendering_matches_plaintext_fewshot_layout(self):
        harness = DummyRawPromptHarness()
        messages = [
            {"role": "user", "content": "Q1\n"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2\n"},
            {"role": "assistant", "content": "A2"},
            {"role": "user", "content": "Q3\n"},
        ]

        prompt_text = harness._render_plain_prompt_from_messages(messages, add_generation_prompt=True)
        self.assertEqual(prompt_text, "Q1\nA1\n\nQ2\nA2\n\nQ3\n")
        token_ids = harness._tokenize_chat_messages(messages, add_generation_prompt=True)
        self.assertEqual(len(token_ids), len("<bos>Q1\nA1\n\nQ2\nA2\n\nQ3\n"))

    def test_chat_prompt_tokenization_uses_template_only(self):
        harness = DummyChatPromptHarness()
        messages = [{"role": "user", "content": "Q1"}]

        token_ids = harness._tokenize_chat_messages(messages, add_generation_prompt=True)
        expected_text = "user:Q1|assistant:"
        self.assertEqual(len(token_ids), len(expected_text))

    def test_chat_prompt_none_path_uses_generation_prompt(self):
        harness = DummyChatNoneHarness()

        input_ids, affix_start, affix_end, generation_start, generation_end = harness._build_input_ids_with_affix(
            "Q1",
            {},
        )
        self.assertEqual(len(input_ids), len("user:Q1|assistant:"))
        self.assertEqual((affix_start, affix_end), (0, 0))
        self.assertEqual((generation_start, generation_end), (len(input_ids), len(input_ids) + 8))

if __name__ == "__main__":
    unittest.main()

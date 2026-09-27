"""Dream dKV schedule and paged-engine parity with the existing HF reference."""
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch

from config import Config
from model.configuration_dream import DreamConfig
from model.modeling_dream import DreamModel
from model_runner import ModelRunner
from sequence import Sequence
from utils import reset_context


def test_dream_dkv_absolute_predecessors_and_refresh():
    runner = object.__new__(ModelRunner)
    runner.config = Config(model_type="dream", mask_id=63, block_length=4)
    seq = Sequence([2, 3], gen_length=4, block_length=4, mask_id=63)
    seq.token_ids[3] = 7
    assert runner._dream_dkv_query_positions(seq, 0, 8) == (list(range(6)), True)
    assert runner._dream_dkv_query_positions(seq, 1, 8) == ([1, 3, 4], False)
    assert runner._dream_dkv_query_positions(seq, 8, 8)[1]
    # Position 1 is a frozen affix token but still predicts masked token 2.
    assert runner._dream_dkv_query_positions(seq, 1, 8, [2, 3, 4, 5]) == ([1, 3, 4], False)


def _reference_modules():
    root = Path(__file__).resolve().parents[2] / "dream"
    package = types.ModuleType("hf_dream_reference")
    package.__path__ = [str(root / "model")]
    sys.modules[package.__name__] = package
    config = importlib.import_module("hf_dream_reference.configuration_dream")
    model = importlib.import_module("hf_dream_reference.modeling_dream")
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    old = sys.modules.get("generate_ACache")
    block_name = "model.generation_utils_block"
    old_block = sys.modules.get(block_name)
    try:
        sys.modules[block_name] = importlib.import_module("hf_dream_reference.generation_utils_block")
        sys.modules["generate_ACache"] = load("dream_reference_acache", "generate_ACache.py")
        generation = load("dream_reference_dkv", "generate_dKV_ACache.py")
    finally:
        if old_block is None:
            sys.modules.pop(block_name, None)
        else:
            sys.modules[block_name] = old_block
        if old is None:
            sys.modules.pop("generate_ACache", None)
        else:
            sys.modules["generate_ACache"] = old
    return config, model, generation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Paged attention requires CUDA")
@pytest.mark.parametrize("placement", ["prefix", "infix", "suffix"])
@pytest.mark.parametrize("ratio", [0.0, 1.0])
@pytest.mark.parametrize("interval", [1, 2])
@torch.inference_mode()
def test_dream_dkv_matches_hf_reference(placement, ratio, interval):
    torch.manual_seed(7)
    hf_config, hf_model, generation = _reference_modules()
    kwargs = dict(vocab_size=128, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  max_position_embeddings=64, mask_token_id=127, pad_token_id=0)
    reference = hf_model.DreamModel(hf_config.DreamConfig(**kwargs)).cuda().bfloat16().eval()
    model = DreamModel(DreamConfig(**kwargs)).cuda().bfloat16().eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    config = Config(hf_config=model.config, model_type="dream", mask_id=127,
                    gen_length=4, block_length=2, max_num_seqs=2, cache_block_size=8,
                    num_kvcache_blocks=16, enable_acache=True, anchor_ratio=ratio,
                    dkv_steps=4, dkv_cache_interval=interval)
    runner = ModelRunner(model, config)
    seqs, expected, traces = [], [], []
    for extra in ([], [9, 10]):
        if placement == "prefix":
            prompt, start, end, gen_start = [2, 3] + extra, 0, 2, None
        elif placement == "infix":
            prompt, start, end, gen_start = extra + [4, 2, 3, 5], len(extra) + 1, len(extra) + 3, None
        else:
            prompt = extra + [4, 5, 127, 127, 127, 127, 2, 3]
            start, end, gen_start = len(extra) + 6, len(extra) + 8, len(extra) + 2
        seqs.append(Sequence(prompt, gen_length=4, block_length=2, cache_block_size=8,
                             mask_id=127, prompt_affix_start=start, prompt_affix_end=end,
                             generation_start=gen_start))
        x = torch.tensor([prompt], device="cuda")
        captured = []
        def capture(module, args, kw, output):
            if kw.get("dual_cache"):
                captured.append((kw["position_ids"].clone(), output.logits.clone()))
        hook = reference.register_forward_hook(capture, with_kwargs=True)
        tokens, _ = generation.generate_with_dkv_anchor_attention(
            reference, x, steps=4, gen_length=4, block_length=2, temperature=0.,
            remasking="low_confidence", mask_id=127, affix_start=start, affix_end=end,
            generation_start=gen_start, anchor_ratio=ratio, selection_mode="top",
            drop_non_anchor=False, dkv_cache_interval=interval,
            precomputed_affix_cache=reference(x[:, start:end], use_cache=True).past_key_values)
        hook.remove()
        offset = len(prompt) if gen_start is None else gen_start
        replay = list(prompt) if gen_start is not None else list(prompt) + [127] * 4
        trace = []
        for step, (pos, logits) in enumerate(captured):
            block_start = offset + (step // 2) * 2
            masks = [p for p in range(block_start, block_start + 2) if replay[p] == 127]
            indices = [pos[0].tolist().index(max(p - 1, 0)) for p in masks]
            masked = logits[:, indices]
            proposals = masked.argmax(-1)
            confidence = masked.double().softmax(-1).gather(-1, proposals.unsqueeze(-1)).squeeze(-1)
            selected = confidence.topk(1, dim=-1).indices
            trace.append((masked[0], proposals[0], selected[0]))
            for idx in selected[0].tolist():
                replay[masks[idx]] = int(proposals[0, idx])
        assert replay[offset:offset + 4] == tokens[0, offset:offset + 4].tolist()
        traces.append(trace)
        expected.append(tokens[0, offset:offset + 4].tolist())
    calls = []
    def checked_select(logits, *args):
        step = len(calls)
        ref_logits, proposals, selected = [torch.stack([trace[step][i] for trace in traces]) for i in range(3)]
        torch.testing.assert_close(logits, ref_logits, atol=0.01, rtol=0.02)
        calls.append(True)
        # BF16 attention can change a near-tied token commitment. Follow the
        # HF trajectory to compare every refresh/compact forward's logits.
        return proposals, selected
    runner._dkv_select_masked_tokens = checked_select
    actual, nfe = runner.generate_with_dkv_acache(seqs)
    reset_context()
    assert nfe == 8
    assert actual == expected
    if ratio == 1.0:
        config.enable_acache = False
        baseline = ModelRunner(model, config)
        prompts = []
        for seq in seqs:
            tokens = list(seq.token_ids)
            tokens[seq.generation_start:seq.generation_end] = [127] * 4
            prompts.append(Sequence(tokens, gen_length=4, block_length=2, cache_block_size=8,
                                    mask_id=127, generation_start=seq.generation_start))
        vanilla, vanilla_nfe = baseline.generate_with_dkv_cache(prompts)
        reset_context()
        assert vanilla_nfe == nfe
        assert vanilla == expected

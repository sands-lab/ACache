from types import SimpleNamespace

import pytest
import torch

from config import Config
from model.configuration_llada import LLaDAConfig
from model.modeling_llada import (
    LLaDAModelLM,
    ModelConfig,
    create_model_config_from_pretrained_config,
)
from model_runner import ModelRunner
from sequence import Sequence


CUDA_AVAILABLE = torch.cuda.is_available()


def test_create_model_config_falls_back_to_defaults():
    partial_config = SimpleNamespace(
        d_model=128,
        n_heads=8,
        n_layers=4,
        vocab_size=1024,
    )

    model_config = create_model_config_from_pretrained_config(partial_config)

    assert model_config.d_model == 128
    assert model_config.n_heads == 8
    assert model_config.n_layers == 4
    assert model_config.vocab_size == 1024
    assert model_config.train_max_sequence_length == ModelConfig().train_max_sequence_length


def test_profile_timing_helpers_are_noop_when_disabled(monkeypatch):
    runner = object.__new__(ModelRunner)
    runner.profile_timing = False
    runner.device = SimpleNamespace(type="cuda")

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *args, **kwargs: pytest.fail("CUDA sync should not run when profile_timing is disabled"),
    )
    monkeypatch.setattr(
        "model_runner.time.perf_counter",
        lambda: pytest.fail("perf_counter should not run when profile_timing is disabled"),
    )

    assert runner._new_timing("dual_cache", 1) is None
    assert runner._start_timing() is None
    runner._finish_timing(None, "total", None)
    runner._add_timing_count(None, "calls")


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is required for ModelRunner ACache tests",
)
def test_prepare_acache_tiny_model_runs_on_cuda():
    torch.manual_seed(0)
    device = torch.device("cuda")

    config = LLaDAConfig(
        vocab_size=64,
        embedding_size=64,
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=64,
        activation_type="relu",
        block_type="llama",
        block_group_size=1,
        rope=True,
        max_sequence_length=64,
        train_max_sequence_length=64,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
        mask_token_id=63,
    )
    model = LLaDAModelLM(config, init_params=True).to(device=device, dtype=torch.bfloat16).eval()
    serve_config = Config(
        hf_config=config,
        mask_id=63,
        recompute_batch_size=2,
        anchor_selection_batch_size=2,
        gen_length=4,
        block_length=4,
        cache_block_size=8,
        num_kvcache_blocks=8,
        temperature=0.0,
        threshold=0.0,
        enable_acache=True,
        anchor_ratio=0.25,
        selection_mode="top",
        use_reference_attention=False,
    )
    runner = ModelRunner(model, serve_config)

    # Advance the global sequence counter so result placement does not rely on seq_id == prompt index.
    Sequence([1], gen_length=1, cache_block_size=8, block_length=1, mask_id=63)
    Sequence([1], gen_length=1, cache_block_size=8, block_length=1, mask_id=63)

    seqs = [
        Sequence([2, 3, 4, 5], gen_length=4, cache_block_size=8, block_length=4, mask_id=63, prompt_affix_len=2),
        Sequence([2, 3, 4, 6], gen_length=4, cache_block_size=8, block_length=4, mask_id=63, prompt_affix_len=2),
    ]

    assert runner.kv_cache.shape[2] == 8
    prefix_token_ids = runner._prepare_acache(seqs)
    assert prefix_token_ids == (2, 3)
    runner._prepare_admitted_sequences_for_acache(seqs, prefix_token_ids, allow_kv_cache_resize=True)
    assert runner.shared_prefix_len == 2
    assert runner.shared_prefix_blocks == 1
    assert runner.shared_prefix_token_ids == (2, 3)
    assert all(len(seq.anchor_positions) == 1 for seq in seqs)
    assert all(0 <= seq.anchor_positions[0] < 2 for seq in seqs)


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is required for ModelRunner ACache tests",
)
def test_batched_anchor_selection_matches_single_sequence_selection():
    torch.manual_seed(0)
    device = torch.device("cuda")

    config = LLaDAConfig(
        vocab_size=64,
        embedding_size=64,
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=64,
        activation_type="relu",
        block_type="llama",
        block_group_size=1,
        rope=True,
        max_sequence_length=64,
        train_max_sequence_length=64,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
        mask_token_id=63,
    )
    model = LLaDAModelLM(config, init_params=True).to(device=device, dtype=torch.bfloat16).eval()
    serve_config = Config(
        hf_config=config,
        mask_id=63,
        recompute_batch_size=2,
        anchor_selection_batch_size=2,
        gen_length=4,
        block_length=4,
        cache_block_size=8,
        num_kvcache_blocks=8,
        temperature=0.0,
        threshold=0.0,
        enable_acache=True,
        anchor_ratio=0.5,
        selection_mode="top",
        use_reference_attention=False,
    )
    runner = ModelRunner(model, serve_config)

    seqs = [
        Sequence([2, 3, 4, 5], gen_length=4, cache_block_size=8, block_length=4, mask_id=63, prompt_affix_len=2),
        Sequence([2, 3, 7, 8], gen_length=4, cache_block_size=8, block_length=4, mask_id=63, prompt_affix_len=2),
    ]
    runner._precompute_shared_prefix([2, 3])
    for seq in seqs:
        seq.enable_acache(2, serve_config.anchor_ratio)

    single_anchor_positions = [runner._compute_anchor_positions(seq) for seq in seqs]
    batched_anchor_positions = runner._compute_anchor_positions_batch(seqs)

    assert batched_anchor_positions == single_anchor_positions


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is required for ModelRunner ACache tests",
)
def test_prepare_acache_keeps_current_kv_cache_when_selection_fits(monkeypatch):
    torch.manual_seed(0)
    device = torch.device("cuda")

    config = LLaDAConfig(
        vocab_size=64,
        embedding_size=64,
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=64,
        activation_type="relu",
        block_type="llama",
        block_group_size=1,
        rope=True,
        max_sequence_length=64,
        train_max_sequence_length=64,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
        mask_token_id=63,
    )
    model = LLaDAModelLM(config, init_params=True).to(device=device, dtype=torch.bfloat16).eval()
    serve_config = Config(
        hf_config=config,
        mask_id=63,
        recompute_batch_size=2,
        anchor_selection_batch_size=1,
        gen_length=4,
        block_length=4,
        cache_block_size=8,
        num_kvcache_blocks=8,
        temperature=0.0,
        threshold=0.0,
        enable_acache=True,
        anchor_ratio=0.25,
        selection_mode="top",
        use_reference_attention=False,
    )
    runner = ModelRunner(model, serve_config)

    seqs = [
        Sequence([2, 3, 4, 5], gen_length=4, cache_block_size=8, block_length=4, mask_id=63, prompt_affix_len=2),
        Sequence([2, 3, 4, 6], gen_length=4, cache_block_size=8, block_length=4, mask_id=63, prompt_affix_len=2),
    ]

    original_compute = runner._compute_anchor_positions_batch
    selection_cache_blocks = []

    def wrapped_compute(batch):
        selection_cache_blocks.append(runner.kv_cache.shape[2])
        return original_compute(batch)

    monkeypatch.setattr(runner, "_compute_anchor_positions_batch", wrapped_compute)

    assert runner.kv_cache.shape[2] == 8
    prefix_token_ids = runner._prepare_acache(seqs)
    assert prefix_token_ids == (2, 3)
    runner._prepare_admitted_sequences_for_acache(seqs, prefix_token_ids, allow_kv_cache_resize=True)
    assert selection_cache_blocks == [8, 8]
    assert runner.kv_cache.shape[2] == 8
    assert runner.config.num_kvcache_blocks == 8
    assert runner.shared_prefix_token_ids == (2, 3)
    assert all(len(seq.anchor_positions) == 1 for seq in seqs)


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is required for ModelRunner ACache tests",
)
def test_generate_with_acache_selects_only_newly_admitted_sequences(monkeypatch):
    torch.manual_seed(0)
    device = torch.device("cuda")

    config = LLaDAConfig(
        vocab_size=64,
        embedding_size=64,
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=64,
        activation_type="relu",
        block_type="llama",
        block_group_size=1,
        rope=True,
        max_sequence_length=64,
        train_max_sequence_length=64,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
        mask_token_id=63,
    )
    model = LLaDAModelLM(config, init_params=True).to(device=device, dtype=torch.bfloat16).eval()
    serve_config = Config(
        hf_config=config,
        mask_id=63,
        recompute_batch_size=1,
        anchor_selection_batch_size=2,
        max_num_seqs=1,
        gen_length=4,
        block_length=4,
        cache_block_size=8,
        num_kvcache_blocks=8,
        temperature=0.0,
        threshold=0.0,
        enable_acache=True,
        anchor_ratio=0.5,
        selection_mode="top",
        use_reference_attention=False,
    )
    runner = ModelRunner(model, serve_config)

    seqs = [
        Sequence([2, 3, 4, 5], gen_length=4, cache_block_size=8, block_length=4, mask_id=63, prompt_affix_len=2),
        Sequence([2, 3, 4, 6], gen_length=4, cache_block_size=8, block_length=4, mask_id=63, prompt_affix_len=2),
    ]

    selection_batch_sizes = []

    def fake_compute(batch):
        selection_batch_sizes.append(len(batch))
        return [[0] for _ in batch]

    monkeypatch.setattr(runner, "_compute_anchor_positions_batch", fake_compute)

    generated, _ = runner.generate_with_acache(seqs)

    assert selection_batch_sizes == [1, 1]
    assert len(generated) == 2
    assert all(tokens is not None for tokens in generated)

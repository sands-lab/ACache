import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import types

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
from utils import reset_context


CUDA_AVAILABLE = torch.cuda.is_available()


def test_attention_read_layout_packs_shared_slots_for_query_relocation():
    runner = object.__new__(ModelRunner)
    runner.shared_prefix_len = 3
    seq = SimpleNamespace(
        affix_start=2,
        shared_slot_offset=0,
        read_slot_map=[8, 9, 0, 10, 2, 11],
    )

    assert runner._attention_read_layout(seq, True) == ([0, 2, 8, 9, 10, 11], 2)


def test_attention_read_layout_keeps_logical_order_for_key_relocation():
    runner = object.__new__(ModelRunner)
    runner.shared_prefix_len = 3
    seq = SimpleNamespace(
        affix_start=2,
        shared_slot_offset=0,
        read_slot_map=[8, 9, 0, 10, 2, 11],
    )

    assert runner._attention_read_layout(seq, False) == ([8, 9, 0, 10, 2, 11], 0)


@pytest.mark.parametrize(
    ("max_num_seqs", "relocation_materialized", "expected"),
    [(0, False, True), (16, False, True), (1, False, False), (0, True, False)],
)
def test_query_side_relocation_treats_zero_as_unlimited_batching(
    max_num_seqs, relocation_materialized, expected
):
    runner = object.__new__(ModelRunner)
    runner._affix_relocation_enabled = True
    runner.config = SimpleNamespace(max_num_seqs=max_num_seqs)

    assert runner._use_query_side_affix_relocation(relocation_materialized) is expected


def _load_hf_llada_acache_modules():
    root = Path(__file__).resolve().parents[2]
    package_name = "acache_hf_llada_model"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root / "llada" / "model")]
        sys.modules[package_name] = package
    config_module = importlib.import_module(f"{package_name}.configuration_llada")
    model_module = importlib.import_module(f"{package_name}.modeling_llada")

    llada_root = str(root / "llada")
    if llada_root not in sys.path:
        sys.path.insert(0, llada_root)
    module_name = "acache_hf_generate"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name,
            root / "llada" / "generate_ACache.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return config_module, model_module, sys.modules[module_name]


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


def test_dkv_transfer_schedule_matches_llada_full_step_setting():
    assert ModelRunner._dkv_transfer_schedule(32, 32) == [1] * 32
    assert ModelRunner._dkv_transfer_schedule(8, 3) == [3, 3, 2]


def test_dkv_query_positions_apply_one_step_delay_and_refresh():
    seq = Sequence([2, 3], gen_length=4, cache_block_size=8, block_length=4, mask_id=63)
    delayed_unresolved = [3, 4, 5]

    assert ModelRunner._dkv_query_positions(seq, None, 0, 8) == (list(range(6)), True)
    assert ModelRunner._dkv_query_positions(seq, None, 1, 8) == (list(range(6)), True)
    assert ModelRunner._dkv_query_positions(seq, delayed_unresolved, 2, 8) == (
        delayed_unresolved,
        False,
    )
    assert ModelRunner._dkv_query_positions(seq, delayed_unresolved, 8, 8) == (
        list(range(6)),
        True,
    )


def test_dkv_acache_refresh_uses_recompute_positions_without_changing_compact_steps():
    seq = Sequence([2, 3], gen_length=4, cache_block_size=8, block_length=4, mask_id=63)
    refresh_positions = [0, 2, 3, 4, 5]
    delayed_unresolved = [3, 4, 5]

    assert ModelRunner._dkv_query_positions(
        seq, None, 0, 8, refresh_positions
    ) == (refresh_positions, True)
    assert ModelRunner._dkv_query_positions(
        seq, delayed_unresolved, 2, 8, refresh_positions
    ) == (delayed_unresolved, False)


def test_dkv_low_confidence_selection_uses_fixed_quota():
    runner = object.__new__(ModelRunner)
    logits = torch.tensor(
        [[
            [5.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0, 1.0],
        ]]
    )

    x0, selected = runner._dkv_select_masked_tokens(
        logits,
        temperature=0.0,
        remasking="low_confidence",
        num_transfer_tokens=2,
    )

    assert x0.tolist() == [[0, 0, 0]]
    assert set(selected[0].tolist()) == {0, 2}


def test_dkv_selective_output_head_matches_full_logits_and_restores_forward():
    torch.manual_seed(0)
    output_head = torch.nn.Linear(5, 7, bias=False)
    runner = object.__new__(ModelRunner)
    runner.base_model = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(weight_tying=False),
            transformer=SimpleNamespace(ff_out=output_head),
        )
    )
    hidden_states = torch.randn(4, 5)
    full_logits = output_head(hidden_states)
    row_indices = torch.tensor([2, 0], dtype=torch.int64)

    with runner._temporary_llada_logit_rows(row_indices):
        selected_logits = output_head(hidden_states)

    torch.testing.assert_close(selected_logits, full_logits[row_indices])
    torch.testing.assert_close(output_head(hidden_states), full_logits)


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is required for ModelRunner dKV+ACache tests",
)
@pytest.mark.parametrize("placement", ["prefix", "infix", "suffix"])
def test_dkv_acache_full_anchor_ratio_matches_dkv_tokens(placement):
    torch.manual_seed(0)
    device = torch.device("cuda")
    model_config = LLaDAConfig(
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
        weight_tying=False,
    )
    baseline_model = LLaDAModelLM(
        model_config, init_params=True
    ).to(device=device, dtype=torch.bfloat16).eval()
    combined_model = LLaDAModelLM(
        model_config, init_params=True
    ).to(device=device, dtype=torch.bfloat16).eval()
    combined_model.load_state_dict(baseline_model.state_dict())

    common = dict(
        hf_config=model_config,
        mask_id=63,
        max_num_seqs=1,
        gen_length=4,
        block_length=4,
        cache_block_size=8,
        num_kvcache_blocks=8,
        temperature=0.0,
        remasking="low_confidence",
        enable_dkv_cache=True,
        dkv_steps=4,
        dkv_cache_interval=2,
        use_reference_attention=False,
    )
    baseline_runner = ModelRunner(baseline_model, Config(**common))
    combined_runner = ModelRunner(
        combined_model,
        Config(
            **common,
            enable_acache=True,
            anchor_ratio=1.0,
            selection_mode="top",
        ),
    )
    if placement == "prefix":
        prompt = [2, 3, 4, 5]
        affix_start, affix_end = 0, 2
        generation_start = None
    elif placement == "infix":
        prompt = [4, 2, 3, 5]
        affix_start, affix_end = 1, 3
        generation_start = None
    else:
        prompt = [4, 5, 63, 63, 63, 63, 2, 3]
        affix_start, affix_end = 6, 8
        generation_start = 2
    baseline_seq = Sequence(
        prompt,
        gen_length=4,
        cache_block_size=8,
        block_length=4,
        mask_id=63,
        generation_start=generation_start,
    )
    combined_seq = Sequence(
        prompt,
        gen_length=4,
        cache_block_size=8,
        block_length=4,
        mask_id=63,
        prompt_affix_start=affix_start,
        prompt_affix_end=affix_end,
        generation_start=generation_start,
    )

    baseline_tokens, baseline_nfe = baseline_runner.generate_with_dkv_cache([baseline_seq])
    combined_tokens, combined_nfe = combined_runner.generate_with_dkv_acache([combined_seq])
    reset_context()

    assert combined_nfe == baseline_nfe == 4
    assert combined_tokens == baseline_tokens


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
        Sequence(
            [4, 10, 11, 12, 13, 2, 3],
            gen_length=4,
            generation_start=1,
            cache_block_size=8,
            block_length=4,
            mask_id=63,
            prompt_affix_start=5,
            prompt_affix_end=7,
        ),
    ]

    selection_batch_sizes = []

    def fake_compute(batch):
        selection_batch_sizes.append(len(batch))
        return [[seq.affix_start] for seq in batch]

    monkeypatch.setattr(runner, "_compute_anchor_positions_batch", fake_compute)

    original_transfer = runner.get_transfer_index

    def non_mask_transfer(*args, **kwargs):
        x0, transfer_index = original_transfer(*args, **kwargs)
        x0 = torch.where(x0 == 63, torch.zeros_like(x0), x0)
        return x0, transfer_index

    monkeypatch.setattr(runner, "get_transfer_index", non_mask_transfer)

    generated, _ = runner.generate_with_acache(seqs)

    assert selection_batch_sizes == [1, 1, 1]
    assert len(generated) == 3
    assert all(tokens is not None for tokens in generated)
    assert seqs[2].token_ids[5:] == [2, 3]


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is required for ModelRunner ACache tests",
)
@pytest.mark.parametrize(
    ("tokens", "affix_start", "affix_end", "generation_start"),
    [
        ([2, 3, 4, 5], 0, 2, None),
        ([4, 2, 3, 5], 1, 3, None),
        ([4, 63, 63, 2, 3], 3, 5, 1),
    ],
)
def test_full_recomputation_matches_baseline_for_all_affix_layouts(
    tokens,
    affix_start,
    affix_end,
    generation_start,
):
    torch.manual_seed(7)
    device = torch.device("cuda")
    model_config = LLaDAConfig(
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
    model = LLaDAModelLM(model_config, init_params=True).to(
        device=device, dtype=torch.bfloat16
    ).eval()

    def make_serve_config(enable_acache):
        return Config(
            hf_config=model_config,
            mask_id=63,
            recompute_batch_size=1,
            gen_length=2,
            block_length=2,
            cache_block_size=8,
            num_kvcache_blocks=8,
            max_num_seqs=4,
            temperature=0.0,
            threshold=0.0,
            enable_acache=enable_acache,
            anchor_ratio=1.0,
            selection_mode="top",
            use_reference_attention=False,
        )

    def make_sequence():
        return Sequence(
            tokens,
            gen_length=2,
            block_length=2,
            cache_block_size=8,
            mask_id=63,
            prompt_affix_start=affix_start,
            prompt_affix_end=affix_end,
            generation_start=generation_start,
        )

    baseline_runner = ModelRunner(model, make_serve_config(False))
    baseline_seq = make_sequence()
    baseline_runner.block_manager.allocate(baseline_seq)
    baseline_seq.finalize_slot_mapping()
    input_ids, positions, _, _ = baseline_runner.prepare_caching([baseline_seq])
    baseline_logits = model(input_ids, position_ids=positions).logits.detach().float()
    reset_context()

    acache_runner = ModelRunner(model, make_serve_config(True))
    acache_seq = make_sequence()
    affix_tokens = acache_runner._prepare_acache([acache_seq])
    acache_runner._precompute_shared_prefix(list(affix_tokens))
    acache_seq.set_anchor_positions(list(range(affix_start, affix_end)))
    acache_runner.block_manager.allocate(acache_seq)
    acache_seq.finalize_slot_mapping()
    input_ids, positions, _ = acache_runner.prepare_caching_acache([acache_seq])
    acache_logits = model(input_ids, position_ids=positions).logits.detach().float()
    reset_context()

    torch.testing.assert_close(acache_logits, baseline_logits, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is required for Hugging Face semantic parity tests",
)
def test_infix_suffix_anchor_and_logits_match_hugging_face_semantics():
    hf_config_module, hf_model_module, hf_generate = _load_hf_llada_acache_modules()
    device = torch.device("cuda")
    common_config = dict(
        vocab_size=128,
        embedding_size=128,
        d_model=64,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        mlp_hidden_size=64,
        activation_type="silu",
        block_type="llama",
        rope=True,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        eos_token_id=0,
        pad_token_id=0,
        mask_token_id=127,
    )
    hf_config = hf_config_module.LLaDAConfig(
        **common_config,
        flash_attention=False,
        init_device="cpu",
    )
    nano_config = LLaDAConfig(
        **common_config,
        block_group_size=1,
        max_sequence_length=64,
        train_max_sequence_length=64,
    )
    torch.manual_seed(19)
    hf_model = hf_model_module.LLaDAModelLM(hf_config).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    nano_model = LLaDAModelLM(nano_config, init_params=True)
    nano_model.load_state_dict(hf_model.state_dict(), strict=True)
    nano_model = nano_model.to(device=device, dtype=torch.bfloat16).eval()

    layouts = [
        ([7, 8, 20, 21], 0, 2, None),
        ([20, 7, 8, 21], 1, 3, None),
        ([20, 30, 31, 7, 8], 3, 5, 1),
    ]
    for tokens, affix_start, affix_end, generation_start in layouts:
        seq = Sequence(
            tokens,
            gen_length=2,
            block_length=2,
            cache_block_size=8,
            mask_id=127,
            prompt_affix_start=affix_start,
            prompt_affix_end=affix_end,
            generation_start=generation_start,
        )
        full_input = torch.tensor(seq.token_ids, device=device, dtype=torch.long).unsqueeze(0)
        affix_ids = full_input[:, affix_start:affix_end]

        with torch.inference_mode():
            affix_cache = hf_model(affix_ids, use_cache=True).past_key_values
            importance = hf_generate.compute_attention_importance_cross_affix(
                hf_model,
                full_input,
                affix_cache,
                mask_id=127,
                affix_start=affix_start,
            )
            expected_anchor = hf_generate.select_anchor_tokens(
                importance,
                1,
                "top",
            )[0].add(affix_start).tolist()

        serve_config = Config(
            hf_config=nano_config,
            mask_id=127,
            recompute_batch_size=1,
            max_num_seqs=4,
            gen_length=2,
            block_length=2,
            cache_block_size=8,
            num_kvcache_blocks=8,
            temperature=0.0,
            threshold=0.0,
            enable_acache=True,
            anchor_ratio=0.5,
            selection_mode="top",
            use_reference_attention=False,
        )
        runner = ModelRunner(nano_model, serve_config)
        affix_tokens = runner._prepare_acache([seq])
        runner._precompute_shared_prefix(list(affix_tokens))
        actual_anchor = runner._compute_anchor_positions(seq)
        assert actual_anchor == expected_anchor

        recompute_mask = torch.ones(
            1, len(seq), dtype=torch.bool, device=device
        )
        recompute_mask[:, affix_start:affix_end] = False
        recompute_mask[:, expected_anchor] = True
        recompute_positions = recompute_mask[0].nonzero(as_tuple=True)[0]
        past_key_values = []
        for affix_k, affix_v in affix_cache:
            layer_k = torch.zeros(
                1,
                affix_k.shape[1],
                len(seq),
                affix_k.shape[-1],
                dtype=affix_k.dtype,
                device=device,
            )
            layer_v = torch.zeros_like(layer_k)
            layer_k[:, :, affix_start:affix_end] = affix_k
            layer_v[:, :, affix_start:affix_end] = affix_v
            past_key_values.append((layer_k, layer_v))

        with torch.inference_mode():
            hf_output = hf_model(
                full_input[:, recompute_positions],
                past_key_values=tuple(past_key_values),
                use_cache=True,
                replace_position=recompute_mask,
                position_ids=recompute_positions,
            )

        seq.set_anchor_positions(expected_anchor)
        runner.block_manager.allocate(seq)
        seq.finalize_slot_mapping()
        input_ids, position_ids, _ = runner.prepare_caching_acache([seq])
        with torch.inference_mode():
            nano_output = nano_model(input_ids, position_ids=position_ids)
        reset_context()

        assert position_ids.tolist() == recompute_positions.tolist()
        torch.testing.assert_close(
            nano_output.logits.float(),
            hf_output.logits.squeeze(0).float(),
            rtol=7e-2,
            atol=7e-2,
        )

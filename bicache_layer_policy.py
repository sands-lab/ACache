"""BiCache-style matched-budget layers on the unchanged ACache schedule.

The boundary O tensor is captured once during the affix-only preparation.
Only request tokens traverse shallow layers; O is spliced at original positions
before the first deep layer. Block-only refinement uses the normal ACache path.
"""
import json
import math
import torch


def layers(model, family):
    return model.model.transformer.blocks if family == "llada" else model.model.layers


class BiCacheEvalMixin:
    def _build_affix_state(self, state_name="default"):
        blocks = layers(self.model, self.bicache_family)
        depth = math.ceil(float(self.anchor_ratio) * len(blocks))
        assert 0 < depth < len(blocks)
        boundary = len(blocks) - depth
        captured = []

        def capture(module, args):
            captured.append(args[0].detach().clone())

        handle = blocks[boundary].register_forward_pre_hook(capture)
        try:
            state = super()._build_affix_state(state_name)
        finally:
            handle.remove()
        assert len(captured) == 1, "Expected one shared affix preparation forward"
        hidden = captured[0]
        assert hidden.shape[1] == state["affix_length"]
        state["bicache_boundary_hidden"] = hidden
        state["bicache_boundary"] = boundary
        kv_bytes = sum(t.numel() * t.element_size() for pair in state["precomputed_affix_cache"] for t in pair)
        state["bicache_metadata"] = dict(
            family=self.bicache_family, ratio=float(self.anchor_ratio), layers=len(blocks),
            shallow_layers=boundary, deep_layers=depth, affix_tokens=hidden.shape[1],
            boundary_shape=list(hidden.shape), boundary_dtype=str(hidden.dtype),
            boundary_bytes=hidden.numel() * hidden.element_size(), shared_kv_bytes=kv_bytes,
            effective_layer_fraction=depth / len(blocks), shared_preparation_forwards=1,
        )
        print("BICACHE_SHARED " + json.dumps(state["bicache_metadata"]), flush=True)
        return state

    def _generate_with_affix_cache(self, input_ids, affix_start, affix_end, generation_start, affix_state):
        import generate_ACache as acache
        assert not self.drop_non_anchor
        proxy = LayerPolicyModel(self.model, self.bicache_family, affix_state, affix_start, affix_end)
        result = acache.generate_with_anchor_attention(
            proxy, input_ids, steps=self.steps, gen_length=self.gen_length,
            block_length=self.block_length, temperature=0.0, remasking=self.remasking,
            mask_id=self.mask_id, threshold=self.threshold, factor=self.factor,
            affix_start=affix_start, affix_end=affix_end,
            generation_start=generation_start if self.affix_type == "suffix" else None,
            anchor_ratio=0.0, selection_mode="top", drop_non_anchor=False,
            precomputed_affix_cache=affix_state["precomputed_affix_cache"],
        )
        assert proxy.events > 0
        print("BICACHE_REQUEST " + json.dumps(dict(
            **affix_state["bicache_metadata"], affix_start=affix_start, affix_end=affix_end,
            recomputation_events=proxy.events, refinement_events=proxy.refinements,
            boundary_reused=True, shallow_kv_verified=proxy.shallow_verified,
            layers_verified=True, positions_verified=True,
        )), flush=True)
        return result


class LayerPolicyModel:
    def __init__(self, model, family, state, start, end):
        self.model, self.family, self.state = model, family, state
        self.start, self.end = start, end
        self.events = self.refinements = 0
        self.shallow_verified = False

    def __getattr__(self, name):
        return getattr(self.model, name)

    @torch.no_grad()
    def __call__(self, input_ids, **kwargs):
        mask = kwargs.get("replace_position")
        assert mask is not None and kwargs.get("past_key_values") is not None
        total = mask.shape[1]
        expected = torch.ones_like(mask)
        expected[:, self.start:self.end] = False
        if not torch.equal(mask, expected):
            assert not mask[:, self.start:self.end].any()
            self.refinements += 1
            return self.model(input_ids, **kwargs)
        positions = expected[0].nonzero(as_tuple=True)[0]
        assert input_ids.shape[1] == positions.numel()
        assert torch.equal(kwargs["position_ids"].reshape(-1), positions)
        hidden = self.state["bicache_boundary_hidden"]
        boundary = self.state["bicache_boundary"]
        blocks = layers(self.model, self.family)
        full_positions = torch.arange(total, device=input_ids.device)
        full_mask = torch.ones_like(mask)
        seen = []

        def pre(index):
            def hook(module, args, kw):
                x = args[0]
                if index == boundary:
                    assert x.shape[1] == positions.numel()
                    expanded = x.new_empty(x.shape[0], total, x.shape[2])
                    expanded[:, positions] = x
                    expanded[:, self.start:self.end] = hidden
                    x = expanded
                if index >= boundary:
                    kw["replace_position"] = full_mask
                    kw["position_ids"] = full_positions if self.family == "llada" else full_positions[None, :]
                    if self.family == "dream":
                        assert kw["dual_cache"] and kw["position_embeddings"] is None
                required = positions.numel() if index < boundary else total
                assert x.shape[1] == required
                assert torch.equal(kw["position_ids"].reshape(-1), positions if index < boundary else full_positions)
                seen.append(index)
                return (x, *args[1:]), kw
            return hook

        def compact_output(module, args, output):
            return (output[0][:, positions], *output[1:])

        handles = [block.register_forward_pre_hook(pre(i), with_kwargs=True) for i, block in enumerate(blocks)]
        handles.append(blocks[-1].register_forward_hook(compact_output))
        try:
            out = self.model(input_ids, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        assert seen == list(range(len(blocks)))
        if not self.shallow_verified:
            axis = 2 if self.family == "llada" else 1
            for index in range(boundary):
                for actual, cached in zip(out.past_key_values[index], self.state["precomputed_affix_cache"][index]):
                    affix = actual.narrow(axis, self.start, self.end - self.start)
                    assert torch.equal(affix, cached), "Shallow affix KV changed"
            self.shallow_verified = True
        self.events += 1
        return out

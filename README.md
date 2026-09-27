# ACache

Official code for [Affix Cache for Diffusion Large Language Models](https://arxiv.org/abs/2608.26140).

ACache enables affix-cache acceleration for diffusion large language models
(DLLMs). Prompts often share text spans such as system prompts, few-shot
examples, or instructions, and these spans can appear before, between, or after
the request-specific content. With bidirectional attention, reusing their KV
cache directly becomes stale, while recomputing it for every request is
expensive. ACache identifies a small request-specific set of **Anchor tokens**
inside the shared affix, recomputes only their KV states, and reuses the rest
of the affix cache.

<p align="center">
  <img src="assets/ARvsDLLM.png" alt="ACache comparison with AR LLMs and DLLMs" width="100%">
</p>

The code supports LLaDA (`GSAI-ML/LLaDA-8B-Instruct`) and Dream
(`Dream-org/Dream-v0-Instruct-7B`) on GSM8K, MBPP, and BABILong.

## Installation

The experiments used Python 3.12, PyTorch 2.5.1 with CUDA 12.1, and NVIDIA
A100-SXM4-40GB GPUs. Install a PyTorch build that matches your CUDA runtime
first, then the remaining dependencies and FlashAttention:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation --no-cache-dir
```

Models and datasets are downloaded from the Hugging Face Hub on first use.

> **Security note.** The evaluators load model code with `trust_remote_code`,
> and MBPP scoring executes generated programs. Review the model and task code
> before running them. MBPP runs must be confirmed explicitly: set
> `HF_ALLOW_CODE_EVAL=1` and pass `--confirm_run_unsafe_code` to the Python
> evaluators or `--confirm-run-unsafe-code` to the shell scripts.

## Quick start

Evaluate ACache on LLaDA with a 1-shot GSM8K prompt as a shared prefix,
recomputing 20% of the affix tokens as Anchors. `--limit 1` runs a single
question as a smoke test; remove it to evaluate the full test split.

```bash
cd llada
python eval_ACache.py --seed 0 --tasks gsm8k --num_fewshot 1 --batch_size 1 --trust_remote_code --limit 1 \
  --model llada_acache \
  --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,threshold=0.9,affix_type=prefix,anchor_ratio=0.2,selection_mode=top" \
  --output_path ../outputs/quickstart
```

For Dream, run `dream/eval_ACache.py` with `--model dream_acache` and replace
`model_path=GSAI-ML/LLaDA-8B-Instruct` with
`pretrained=Dream-org/Dream-v0-Instruct-7B`.

### Key options

These are passed in `--model_args`:

| Option | Values | Meaning |
| --- | --- | --- |
| `affix_type` | `prefix`, `infix`, `suffix` | Where the shared few-shot affix is placed relative to the request |
| `anchor_ratio` | `0` to `1` | Fraction of affix tokens recomputed as Anchors. `0` reuses the affix cache directly; `1` recomputes the whole affix |
| `selection_mode` | `top`, `bottom`, `random` | How Anchors are chosen: highest influence (ACache), lowest influence, or random |
| `gen_length`, `steps`, `block_length` | integers | Generation length, denoising steps, and semi-autoregressive block size. We use `256/256/32` for GSM8K and MBPP and `8/8/8` for BABILong |
| `threshold` | float | Fast-dLLM parallel-decoding confidence threshold (we use `0.9`) |

`--num_fewshot N` sets the number of shared examples in the affix. If
`--output_path` is omitted, results are written under `evals_results/` in the
working directory.

## Running experiments

Run commands from the repository root unless a command changes directory. Use a
distinct output path for each configuration.

### Accuracy vs. Anchor ratio

`eval_ACache_anchor_ratio.sh` sweeps Anchor ratios `0, 0.1, 0.2, 0.3, 0.5, 1.0`
for one model, dataset, shot count, placement, and seed:

```bash
bash llada/eval_ACache_anchor_ratio.sh --seed 0 --dataset gsm8k --num-fewshot 1 --prefix
bash dream/eval_ACache_anchor_ratio.sh --seed 0 --dataset babilong --num-fewshot 2 --suffix
```

`--dataset` accepts `gsm8k`, `mbpp`, or `babilong`. Placement is `--prefix`,
`--infix`, or `--suffix`. Generation lengths are set per dataset automatically.
Results go to `llada/evals_results/` or `dream/evals_results/`.
`eval_CacheBlend_ACache_anchor_ratio.sh` runs the same sweep with CacheBlend's
selector.

### System performance

`nano-vdllm/run_eval.sh` measures throughput, recomputation latency, and KV-cache
memory. Compare the Fast-dLLM baseline without cross-request reuse against
ACache:

```bash
cd nano-vdllm
bash run_eval.sh --model llada --dataset gsm8k --baseline --seed 0 --num-fewshot 4 --batch-size 16 --no-profile
bash run_eval.sh --model llada --dataset gsm8k --acache --anchor-ratio 0.2 --seed 0 --num-fewshot 4 --batch-size 16 --no-profile
```

`--model` is `llada` or `dream`, and `--dataset` is `gsm8k` or `mbpp`.
`--no-profile` reports throughput and peak KV allocation; `--profile` enables
timing instrumentation for recomputation latency. `--dkv-cache` and
`--dkv-acache` run dKV-Cache without and with ACache. Arguments after `--` go
to the evaluator, for example `-- --limit 1`. See
[`nano-vdllm/README.md`](nano-vdllm/README.md) for engine details.

### Baselines

**Alternative Anchor selectors.** CacheBlend, ProphetKV, and matched BiCache
layers replace ACache's selection criterion. Evaluator arguments, the Anchor
budget, and cache reuse stay the same. To run one, take the Quick start command
and change only the script and `--model`:

| Selector | Script in `llada/` or `dream/` | Model name |
| --- | --- | --- |
| ACache | `eval_ACache.py` | `llada_acache` / `dream_acache` |
| CacheBlend | `eval_CacheBlend_ACache.py` | `llada_cacheblend_acache` / `dream_cacheblend_acache` |
| ProphetKV | `eval_ProphetKV_ACache.py` | `llada_prophetkv_acache` / `dream_prophetkv_acache` |
| BiCache (matched layers) | `eval_BiCache_ACache.py` | `llada_bicache_acache` / `dream_bicache_acache` |

**dKV-Cache.** ACache on top of dKV-Cache's intra-request caching. Use
`eval_dKV_ACache.py` with `--model llada_dkv_acache_hf` or
`dream_dkv_acache_hf`, and add the dKV options to `--model_args`:

```bash
cd llada
python eval_dKV_ACache.py --seed 0 --tasks gsm8k --num_fewshot 1 --batch_size 1 --trust_remote_code \
  --model llada_dkv_acache_hf \
  --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,threshold=0.9,affix_type=prefix,anchor_ratio=0.2,selection_mode=top,dkv_steps=256,dkv_cache_interval=8" \
  --output_path ../outputs/llada_dkv_gsm8k_prefix_1shot_r0.2_seed0
```

`dkv_steps` equals `steps`, so it is `8` for BABILong.

**Official BiCache.** The official engine is bundled under
[`third_party/BiCache/`](third_party/BiCache/), with profiled policies in
`policies/`. It is evaluated with prefix affixes. For LLaDA:

```bash
# Accuracy
python bicache_fast_dllm_accuracy.py --policy-path policies/bicache_llada.json --dataset gsm8k --shots 1 --seed 0 \
  --output outputs/bicache_llada_gsm8k_1shot_seed0.json --requests-output outputs/bicache_llada_gsm8k_1shot_seed0.jsonl
# Throughput with Fast-dLLM, batch size 1
python bicache_fast_dllm_system.py --mode run --policy-path policies/bicache_llada.json --dataset gsm8k --shots 1 --seed 0 \
  --output outputs/bicache_llada_gsm8k_1shot_seed0_system.json
```

For Dream, use `dream_bicache_accuracy.py` and `dream_bicache_system.py` with
`policies/bicache_dream.json`. The engine uses a 16-step intra-request refresh
interval and cache budget 5000. The policies were profiled on
`allenai/WildChat-4.8M`. `bicache_fast_dllm_system.py --mode profile` and
`dream_bicache_profile.py` regenerate them, which takes substantial GPU time.

## Reproducing the paper results

Figure and table numbers follow the current version of the paper. Each
reported number is the mean over seeds 0 and 1 on the complete benchmark split.

| Result | Configurations |
| --- | --- |
| Accuracy vs. Anchor ratio (Figure 5) | LLaDA and Dream; GSM8K, MBPP, BABILong; prefix, infix, suffix; 1- and 2-shot; Fast-dLLM and dKV-Cache |
| Throughput, recomputation latency, KV memory (Tables 2–4) | Baseline vs. ACache at ratio 0.2; GSM8K, MBPP; 1-, 2-, 4-shot; batch 1, 4, 16. Latency from LLaDA runs with `--profile` |
| Anchor selectors (Table 5) | Ratios 0.2 and 0.3; 1-shot; all datasets and placements (official BiCache: prefix only) |
| Comparison with official BiCache (Table 6) | Prefix; batch 1; GSM8K, MBPP; 1-, 2-, 4-shot; compared with ACache batch-1 runs from `run_eval.sh` |

Figure 5's average recovery excludes BABILong suffix, although those points are
still evaluated. System timings and memory depend on the GPU type and software
versions.

## Repository layout

| Path | Contents |
| --- | --- |
| `acache_eval_shared.py` | Shared prompt, affix, and few-shot construction for all evaluators |
| `llada/`, `dream/` | ACache generation and `lm-eval` evaluators, dKV-Cache and selector baselines, sweep scripts |
| `nano-vdllm/` | Batched Fast-dLLM/ACache engine and system benchmark entry point |
| `lm_eval_tasks/` | Bundled BABILong task definition |
| `bicache_*.py`, `dream_bicache_*.py` | Adapters that run the official BiCache engine on the paper workloads |
| `third_party/BiCache/` | Official BiCache source (AGPL-3.0) |
| `policies/` | Profiled BiCache policies and the Dream ratio index file |
| `test_acache_eval_shared.py`, `nano-vdllm/tests/` | Unit tests (`pytest`) |

## License

This project is released under the Apache License 2.0; see [`LICENSE`](LICENSE).
It adapts code from Fast-dLLM, LLaDA, Dream, and other projects listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The bundled BiCache source
keeps its own AGPL-3.0 license in `third_party/BiCache/LICENSE`.

Parts of this codebase were developed with assistance from AI coding tools; the
authors reviewed and take responsibility for this code.

## Citation

```bibtex
@article{liang2026affix,
  title={Affix Cache for Diffusion Large Language Models},
  author={Liang, Kaihua and Zhong, An and Tan, Xin and Qazi, Zafar Ayyub and Xu, Hong and Weng, Jian and Canini, Marco},
  journal={arXiv preprint arXiv:2608.26140},
  year={2026}
}
```

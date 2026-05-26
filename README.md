# ACache

ACache enables affix-cache acceleration for diffusion large language models.
This repository contains two evaluation paths:

- `llada/` and `dream/`: ACache evaluation scripts for LLaDA and Dream.
- `nano-vdllm/`: a compact batched inference/evaluation implementation with ACache support.

## Environment

The experiments were run in a conda environment named `ACache`.

```bash
conda create -n ACache python=3.12 -y
conda activate ACache

# Install the PyTorch build that matches your CUDA or CPU runtime first.
# For example, the experiments here used torch 2.5.1 with CUDA 12.1 wheels.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install flash-attn --no-build-isolation --no-cache-dir
```

Choose the PyTorch command that matches your CUDA or CPU runtime from the
PyTorch installation selector:
https://pytorch.org/get-started/locally/.

## ACache Evaluation

The ACache scripts sweep anchor ratios `0.0, 0.1, 0.2, 0.3, 0.5, 1.0` for a
fixed model, dataset, seed, number of few-shot examples, and placement.

Run LLaDA on MBPP with infix placement:

```bash
cd llada
./eval_ACache_anchor_ratio.sh --seed 0 --dataset mbpp --num-fewshot 2 --infix
```

Run LLaDA on BABILong with default prefix placement:

```bash
cd llada
./eval_ACache_anchor_ratio.sh --seed 1 --dataset babilong --num-fewshot 1
```

Run the drop-non-anchor ablation for Dream:

```bash
cd dream
./eval_ACache_anchor_ratio.sh --seed 1 --dataset gsm8k --num-fewshot 1 --suffix --drop-non-anchor
```

Run the CacheBlend-style anchor-selection ablation for LLaDA with default
prefix placement:

```bash
cd llada
./eval_CacheBlend_ACache_anchor_ratio.sh --seed 0 --dataset mbpp --num-fewshot 2
```

Run the CacheBlend-style ablation for Dream with suffix placement:

```bash
cd dream
./eval_CacheBlend_ACache_anchor_ratio.sh --seed 1 --dataset babilong --num-fewshot 1 --suffix
```

The supported placement flags are `--prefix`, `--infix`, and `--suffix`;
the default is `--prefix`.
Use `--drop-non-anchor` to enable the drop-non-anchor ablation.
The CacheBlend-style scripts default to L2 scoring over both keys and values.

The evaluation entry points are:

- `llada/eval_ACache.py`
- `llada/eval_CacheBlend_ACache.py`
- `dream/eval_ACache.py`
- `dream/eval_CacheBlend_ACache.py`

## nano-vdllm Evaluation

`nano-vdllm` provides a compact evaluator for batched LLaDA/Dream inference
with and without ACache.

```bash
cd nano-vdllm
./run_eval.sh --model llada --dataset mbpp --acache --seed 0 --num-fewshot 2 --batch-size 16 --anchor-ratio 0.2
```

Baseline evaluation uses the same script:

```bash
cd nano-vdllm
./run_eval.sh --model llada --dataset gsm8k --baseline --seed 1 --num-fewshot 1 --batch-size 16
```

Dream is selected with `--model dream`:

```bash
cd nano-vdllm
./run_eval.sh --model dream --dataset mbpp --acache --seed 1 --num-fewshot 1 --batch-size 16 --anchor-ratio 0.2
```

`nano-vdllm/run_eval.sh` accepts `--dataset {mbpp|gsm8k}`,
`--model {llada|dream}`, `--acache` or `--baseline`, `--seed`,
`--num-fewshot`, `--batch-size`, `--anchor-ratio`, and optional profiling flags
`--profile` / `--no-profile`.

## AI Assistance Disclosure

Parts of this codebase were developed with assistance from AI coding tools.
The authors reviewed, tested, and take responsibility for all submitted code.

## Third-Party Code

Some source files retain upstream copyright and license headers from public
implementations. See `THIRD_PARTY_NOTICES.md` for the corresponding sources
and license notices.

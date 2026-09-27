# Prerequisite

```bash
conda create -n ACache python=3.12 -y
conda activate ACache

# Install the PyTorch build that matches your CUDA or CPU runtime first.
# For example, the experiments here used torch 2.5.1 with CUDA 12.1 wheels.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r ../requirements.txt
pip install flash-attn --no-build-isolation --no-cache-dir
```

Choose the PyTorch command that matches your CUDA or CPU runtime from the
PyTorch installation selector: https://pytorch.org/get-started/locally/.

# Evaluation

```bash
python eval_llada.py --seed 1 \
  --tasks gsm8k \
  --num_fewshot 1 \
  --batch_size 16 \
  --model llada_dist \
  --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,recompute_batch_size=4,show_speed=True"
```

`eval_llada.py` maps `--num_fewshot` to model-side `fewshot_num_examples` and rebuilds prefix few-shot prompts with the same logic as `../llada/eval_ACache.py`. `lm_eval`'s own few-shot prompt construction is disabled for this path.

For MBPP, use `--tasks mbpp`; lm-eval already provides the task, and the script auto-selects `google-research-datasets/mbpp`, `full:prompt`, and `text`/`code` few-shot keys. MBPP may execute generated Python code during evaluation, so pass `--confirm_run_unsafe_code` only after reviewing and trusting the task and model code.

# Affix placement

The engine accepts shared prefixes, infixes, and suffixes through explicit
logical spans on `Sequence`. The shared affix must not overlap the generation
span.

```python
from sequence import Sequence

# Completion with a shared infix at positions [2, 5).
infix_request = Sequence(
    token_ids,
    gen_length=256,
    prompt_affix_start=2,
    prompt_affix_end=5,
)

# In-place generation at [4, 260), conditioned on a shared suffix [260, 300).
suffix_request = Sequence(
    token_ids,
    gen_length=256,
    generation_start=4,
    prompt_affix_start=260,
    prompt_affix_end=300,
)
```

Shared keys remain post-RoPE in the paged cache. For an infix or suffix, the
attention kernel applies the exact RoPE delta between the affix's precompute
position and its logical request position. Prefix requests use the original
kernel specialization without relocation work.

Relocated batch-size-1 requests materialize a request-positioned copy of shared
K/V once and select it through the read-slot map, preserving the reference
numerical path without repeating relocation during decoding. The workspace is
used only when sufficient KV blocks remain; otherwise the kernel falls back to
key-side relocation. Larger configured batches pack shared slots into a
separate attention partition and apply the inverse RoPE shift to each query
tile once, amortizing relocation without per-request KV copies.

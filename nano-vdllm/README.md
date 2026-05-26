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

`eval_llada.py` maps `--num_fewshot` to model-side `fewshot_num_examples` and rebuilds prefix few-shot prompts with the same logic as `../ACache/llada/eval_ACache.py`. `lm_eval`'s own few-shot prompt construction is disabled for this path.

For MBPP, use `--tasks mbpp`; lm-eval already provides the task, and the script auto-selects `google-research-datasets/mbpp`, `full:prompt`, and `text`/`code` few-shot keys.

# Third-Party Notices

This repository includes code adapted from public third-party implementations.
The notices below identify upstream sources and retained license headers only.
They are not author or affiliation information for this submission.

## NVIDIA Fast-dLLM

- Upstream: https://github.com/NVlabs/Fast-dLLM
- License: Apache License 2.0
- License notice retained in source files: Apache License 2.0
- Files retaining NVIDIA/Fast-dLLM copyright and Apache-2.0 headers:
  - `llada/generate.py`
  - `llada/model/__init__.py`
  - `llada/model/configuration_llada.py`
  - `llada/model/modeling_llada.py`
  - `dream/model/__init__.py`
  - `dream/model/configuration_dream.py`
  - `dream/model/generation_utils.py`
  - `dream/model/generation_utils_block.py`
  - `dream/model/modeling_dream.py`
  - `dream/model/tokenization_dream.py`
  - `nano-vdllm/model/configuration_dream.py`
  - `nano-vdllm/model/configuration_llada.py`
  - `nano-vdllm/model/modeling_dream.py`

## LLaDA

- Upstream: https://github.com/ML-GSAI/LLaDA
- License: MIT License
- Files whose in-file comments mark them as modified from LLaDA code:
  - `llada/generate.py`
  - `llada/model/__init__.py`
  - `llada/model/configuration_llada.py`
  - `llada/model/modeling_llada.py`
  - `nano-vdllm/model/configuration_llada.py`
  - `nano-vdllm/model/modeling_llada.py`

## Dream

- Upstream: https://github.com/HKUNLP/Dream
- License: Apache License 2.0
- Files whose in-file comments mark them as modified from Dream code:
  - `dream/model/__init__.py`
  - `dream/model/configuration_dream.py`
  - `dream/model/generation_utils.py`
  - `dream/model/generation_utils_block.py`
  - `dream/model/modeling_dream.py`
  - `dream/model/tokenization_dream.py`
  - `nano-vdllm/model/configuration_dream.py`
  - `nano-vdllm/model/modeling_dream.py`

## nano-vllm

- Upstream: https://github.com/GeeeekExplorer/nano-vllm
- License: MIT License
- Copyright notice: Copyright (c) 2025 Xingkai Yu
- Files adapted from the nano-vllm runtime, scheduler, and paged-attention
  implementation:
  - `nano-vdllm/config.py`
  - `nano-vdllm/block_manager.py`
  - `nano-vdllm/scheduler.py`
  - `nano-vdllm/sequence.py`
  - `nano-vdllm/model_runner.py`
  - `nano-vdllm/model/attention.py`
  - `nano-vdllm/utils.py`

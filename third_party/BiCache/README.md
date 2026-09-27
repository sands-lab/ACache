# BiCache baseline source

This directory contains the BiCache source used by the official baseline in the paper. The upstream LLaDA implementation is from commit `82a8d0122807daf58ceacd6a121201b8355f46c1` of `OSSS-KU/BiCache`. The Dream model and Fast-dLLM adapter are local extensions used for the Dream comparison. ACache's workload adapters in the repository root preserve BiCache's cache and policy logic while matching the paper's prompt construction and threshold decoding. See `THIRD_PARTY_NOTICES.md` and this directory's `LICENSE` for attribution and terms.

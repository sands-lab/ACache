from typing import (
    Dict,
    List,
    Tuple,
    Optional
)
from itertools import takewhile

from datasets import load_dataset
import torch
from torch.nn.functional import cosine_similarity
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm


class ProfilerBase:
    def __init__(
        self,
        device: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    
    def tokenize(self, sequence: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        raise NotImplementedError


    def generate_kvs(self, prefix_ids: List[int], user_prompt_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


    def calculate_cosine_similarity(self, kv1, kv2) -> torch.Tensor:
        res = []
        for (k1, v1), (k2, v2) in zip(kv1, kv2):
            sim_k = cosine_similarity(k1, k2, dim=1)
            sim_v = cosine_similarity(v1, v2, dim=1)
            res.append(torch.mean(torch.stack((sim_k, sim_v))))
        return torch.tensor(res).to(self.device)
    

    def set_number_of_inter_request_caching_layer(self, similarity: List[torch.Tensor], threshold: float) -> Dict[int, int]:
        number_of_inter_request_caching_layer = {}

        previous = 0
        for prefix_ratio, s in enumerate(similarity):
            n = sum(1 for _ in takewhile(lambda x: x > threshold, s))
            if previous < n:
                previous = n
                number_of_inter_request_caching_layer[prefix_ratio] = n
        return number_of_inter_request_caching_layer
    

    def profile(
        self,
        dataset_name: str,
        ids: List[List[Tuple[str, int]]],
        max_sequence_length: Optional[int] = None,
        num_profiling_data_per_ratio: int = 100,
        threshold: float = 0.95
    ) -> Dict[int, int]:
        ds = load_dataset(dataset_name, split="train")
        datas = []

        for r, l in enumerate(tqdm(ids, desc="profiling")):
            if len(l) < num_profiling_data_per_ratio:
                raise ValueError("Not enough data exist.")
            
            d = []
            for id, n_t in l:
                if len(d) == num_profiling_data_per_ratio:
                    break
                conversation = ds['conversation'][id][:n_t+1]
                sequence = [{"role": c["role"], "content": c['content']} for c in conversation]

                prefix_ids, user_prompt_ids = self.tokenize(sequence)
                prefix_len = len(prefix_ids)
                seq_len = prefix_len + len(user_prompt_ids)
                prefix_ratio = round(prefix_len / seq_len * 100)
                assert prefix_ratio == r

                if max_sequence_length is not None and seq_len > max_sequence_length:
                    continue

                kv1, kv2 = self.generate_kvs(prefix_ids, user_prompt_ids) # [(k1, v1), ...], [(k2, v2), ...]
                res = self.calculate_cosine_similarity(kv1, kv2)
                d.append(res)
            datas.append(d)

        for d in datas:
            if len(d) < num_profiling_data_per_ratio:
                raise ValueError("Not enough data exist.")
            
        return self.set_number_of_inter_request_caching_layer([torch.mean(torch.stack(d), dim=0) for d in datas], threshold)
    

class LLaDAProfiler(ProfilerBase):
    def tokenize(self, sequence: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        prefix_ids = self.tokenizer.apply_chat_template(sequence[:-1], add_generation_prompt=True, tokenize=True)[:-6]
        user_prompt_ids = self.tokenizer.apply_chat_template([sequence[-1]], add_generation_prompt=True, tokenize=True)[1:]

        return prefix_ids, user_prompt_ids


    def generate_kvs(self, prefix_ids: List[int], user_prompt_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        prefix_len = len(prefix_ids)
        prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=self.device)
        sequence = torch.concat([prefix_ids, torch.tensor(user_prompt_ids, dtype=torch.long, device=self.device)])

        with torch.inference_mode():
            _, prefix_cache = self.model(prefix_ids.unsqueeze(0))
            _, cache = self.model(sequence.unsqueeze(0))

        return (
            [
            (k.squeeze(0).permute(1, 0, 2).contiguous().view(-1, 32 * 128), 
             v.squeeze(0).permute(1, 0, 2).contiguous().view(-1, 32 * 128)
            ) for k, v, _ in prefix_cache
            ], 
            [
            (k[:, :, :prefix_len, :].squeeze(0).permute(1, 0, 2).contiguous().view(-1, 32 * 128), 
             v[:, :, :prefix_len, :].squeeze(0).permute(1, 0, 2).contiguous().view(-1, 32 * 128)
            ) for k, v, _ in cache
            ], 
        )


class DreamProfiler(ProfilerBase):
    """Official BiCache profiler adapted to Dream's HF cache layout.

    Dream returns one ``(key, value)`` pair per layer with shape
    ``(batch, sequence, kv_heads * head_dim)``.  The official profiler's
    cosine rule is unchanged: compare the prefix-only KV with the same token
    slice from the full request and count the contiguous shallow layers whose
    mean K/V cosine is above the profiler threshold.
    """

    _ASSISTANT_SUFFIX = "<|im_start|>assistant\n"

    def _assistant_suffix_ids(self) -> List[int]:
        return list(
            self.tokenizer(
                self._ASSISTANT_SUFFIX,
                add_special_tokens=False,
            )["input_ids"]
        )

    def tokenize(self, sequence: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        if not sequence or sequence[-1]["role"] != "user":
            raise ValueError("DreamProfiler expects a conversation ending in a user message.")

        full_ids = list(
            self.tokenizer.apply_chat_template(
                sequence,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        if len(sequence) == 1:
            return [], full_ids

        prefix_with_generation_prompt = list(
            self.tokenizer.apply_chat_template(
                sequence[:-1],
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        suffix_ids = self._assistant_suffix_ids()
        if suffix_ids and prefix_with_generation_prompt[-len(suffix_ids) :] == suffix_ids:
            prefix_ids = prefix_with_generation_prompt[: -len(suffix_ids)]
        else:
            raise ValueError(
                "Dream chat template did not end in the expected assistant generation prompt; "
                f"suffix={suffix_ids!r}, prefix_tail={prefix_with_generation_prompt[-8:]!r}"
            )

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError("Dream profiler prefix tokenization is not a true full-conversation prefix.")
        user_prompt_ids = full_ids[len(prefix_ids) :]
        if prefix_ids + user_prompt_ids != full_ids:
            raise AssertionError("Dream profiler tokenization failed to reconstruct the full request.")
        return prefix_ids, user_prompt_ids

    def generate_kvs(
        self,
        prefix_ids: List[int],
        user_prompt_ids: List[int],
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        prefix_len = len(prefix_ids)
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        full_tensor = torch.cat(
            (
                prefix_tensor,
                torch.tensor(user_prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0),
            ),
            dim=1,
        )

        with torch.inference_mode():
            prefix_output = self.model(
                prefix_tensor,
                use_cache=True,
                num_logits_to_keep=1,
                return_dict=True,
            )
            full_output = self.model(
                full_tensor,
                use_cache=True,
                num_logits_to_keep=1,
                return_dict=True,
            )

        prefix_kv = []
        full_prefix_kv = []
        for (prefix_k, prefix_v), (full_k, full_v) in zip(
            prefix_output.past_key_values,
            full_output.past_key_values,
        ):
            prefix_k = prefix_k.squeeze(0).float()
            prefix_v = prefix_v.squeeze(0).float()
            full_k = full_k[:, :prefix_len, :].squeeze(0).float()
            full_v = full_v[:, :prefix_len, :].squeeze(0).float()
            prefix_kv.append((prefix_k, prefix_v))
            full_prefix_kv.append((full_k, full_v))

        del prefix_output, full_output, prefix_tensor, full_tensor
        return prefix_kv, full_prefix_kv

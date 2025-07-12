from __future__ import annotations
import collections
from math import sqrt
from itertools import chain, tee
from functools import lru_cache

import scipy.stats
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from normalizers import normalization_strategy_lookup
from alternative_prf_schemes import prf_lookup, seeding_scheme_lookup


class WatermarkBaseBatched:
    def __init__(
            self,
            vocab: list[int] = None,
            gamma: float = 0.25,
            delta: float = 2.0,
            seeding_scheme: str = "selfhash",  # simple default, find more schemes in alternative_prf_schemes.py
            select_green_tokens: bool = True,  # should always be the default if not running in legacy mode
    ):
        # patch now that None could now maybe be passed as seeding_scheme
        if seeding_scheme is None:
            seeding_scheme = "selfhash"

        # Vocabulary setup
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rngs = None
        self._initialize_seeding_scheme(seeding_scheme)
        # Legacy behavior:
        self.select_green_tokens = select_green_tokens

    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)

    def _seed_rng_in_batch(self, input_ids: torch.LongTensor) -> None:
        batch_size, seq_len = input_ids.shape

        if seq_len < self.context_width:
            raise ValueError(f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG.")

        contexts = input_ids[:, -self.context_width:]  # [batch_size, context_width]
        prf_func = prf_lookup[self.prf_type]
        self.rngs = ((prf_func(contexts, self.hash_key)) % (2 ** 64 - 1)).tolist()   # [batch_size]

    def _get_greenlist_list(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng_in_batch(input_ids)  # 调用方法设定随机种子成员变量
        greenlist_size = int(self.vocab_size * self.gamma)

        generators = [torch.Generator(device=input_ids.device).manual_seed(seed) for seed in self.rngs]
        perms = torch.stack([
            torch.randperm(self.vocab_size, generator=gen, device=input_ids.device)
            for gen in generators
        ])

        greenlist_list = perms[:, :greenlist_size].tolist()
        return greenlist_list


class WatermarkLogitsProcessorBatched(WatermarkBaseBatched, LogitsProcessor):

    def __init__(self, *args, store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None
        if self.store_spike_ents:
            self._init_spike_entropies()

    # batched! scores和greenlist_token_ids都是批量的
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx, greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    # batched
    # greedy！scores降序排列，测试其前k个token是否满足
    def _score_rejection_sampling(self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
                                  k=50) -> list[int]:
        # 得分和token id，针对批量scores依然成立
        sorted_scores, greedy_predictions = scores[:, :k].sort(dim=-1, descending=True)
        batch_size, top_k = greedy_predictions.shape

        # 储存所有的k种拼接方式，input+prediction（在最后一个维度相加）
        expanded_inputs = torch.cat([
            # 扩充第二维为[B,1,L]，也就是给每一个seq再套上一层
            # expand -1表示保留原有维度，top_k表示把每个seq复制top_k次
            input_ids.unsqueeze(1).expand(-1, top_k, -1),  # [B, K, L]
            # greedy_predictions本来是[B,K]，-1表示最后增加一个维度
            greedy_predictions.unsqueeze(-1)  # [B, K, 1]
        ], dim=-1).reshape(batch_size * top_k, -1)  # [B*K, L+1]

        # 针对每一种拼接得到一个对应的green list，得到一个包含B*K个green list的list
        batched_greenlists = self._get_greenlist_list(expanded_inputs)  # [B*K, fixed_len]
        flat_candidates = greedy_predictions.flatten()  # [B*K]
        is_green = torch.tensor([
            candidate.item() in greenlist
            for candidate, greenlist in zip(flat_candidates, batched_greenlists)
        ], device=input_ids.device).reshape(batch_size, top_k)  # [B, K]

        # 得到长度不一的green list
        final_greenlists = [[] for _ in range(batch_size)]
        for batch_index in range(batch_size):
            curr_predictions = greedy_predictions[batch_index]
            curr_bool = is_green[batch_index]
            for i in range(k):
                if curr_bool[i]:
                    final_greenlists[batch_index].append(curr_predictions[i])

        # return torch.as_tensor(final_greenlist, device=input_ids.device)
        return final_greenlists

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""

        # this is lazy to allow us to co-locate on the watermarked model's device
        self.rngs = torch.Generator(device=input_ids.device) if self.rngs is None else self.rngs

        if self.self_salt:  # 只有seeding_scheme == "algorithm-3" or "selfhash"的盐值是True
            list_of_greenlist_ids = self._score_rejection_sampling(input_ids, scores)
        else:
            list_of_greenlist_ids = self._get_greenlist_list(input_ids)

        # logic for computing and storing spike entropies for analysis
        # 尖峰熵在这里只是计算、存储，以作分析，没有对其进行什么筛选判断
        # if self.store_spike_ents:

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=list_of_greenlist_ids)
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)

        return scores

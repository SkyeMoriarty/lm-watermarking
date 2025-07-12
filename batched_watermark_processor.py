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
        self.rngs = []
        self._initialize_seeding_scheme(seeding_scheme)
        # Legacy behavior:
        self.select_green_tokens = select_green_tokens

    # 设置四个内部关键属性：
    # 1. 伪随机函数prf：决定如何生成哈希值
    # 2. 上下文宽度，表示依赖的前缀token的数量
    # 3. 盐值，用于增强哈希的唯一性（是一个随机生成的额外数据，以防止攻击），这里是一个布尔值，如果true，hash key就是盐值
    # 4. 哈希密钥，用于初始化伪随机数生成器（PRNG）——全局唯一，长期有效
    # 区别于1.prf_key：临时生成的随机密钥，由hash key+上下文通过哈希函数计算得出
    # 2. rng：伪随机数生成器，通过prf_key初始化，用于决定green_list的划分
    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)

    def _seed_rng_in_batch(self, input_ids: torch.LongTensor) -> None:
        batch_size, seq_len = input_ids.shape
        if seq_len < self.context_width:
            raise ValueError(f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG.")

        prf_func = prf_lookup[self.prf_type]
        prf_keys = [prf_func(input_ids[i][-self.context_width:], salt_key=self.hash_key) for i in range(batch_size)]

        for prf_key in prf_keys:
            self.rngs.append(prf_key % (2 ** 64 - 1))

    def _get_greenlist_list(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng_in_batch(input_ids)  # 调用方法设定随机种子成员变量

        greenlist_size = int(self.vocab_size * self.gamma)
        greenlist_list = []
        for rng in self.rngs:
            vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=rng)
            if self.select_green_tokens:  # directly
                greenlist_ids = vocab_permutation[:greenlist_size]  # new
            else:  # select green via red
                greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size):]  # legacy behavior
            greenlist_list.append(greenlist_ids)
        return greenlist_list


class WatermarkLogitsProcessorBatched(WatermarkBaseBatched, LogitsProcessor):

    def __init__(self, *args, store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None
        if self.store_spike_ents:
            self._init_spike_entropies()

    def _init_spike_entropies(self):
        alpha = torch.exp(torch.tensor(self.delta)).item()  # 将logit偏置delta转换为【概率空间的乘数】=>改变原来的token概率分布
        gamma = self.gamma

        # 分子表示非绿色token概率被压缩的程度
        self.z_value = ((1 - gamma) * (alpha - 1)) / (
                    1 - gamma + (alpha * gamma))  # 熵的模数-决定熵的敏感度，用于量化【水印对token分布的扰动强度】=>作用于全局，后面用此计算文本熵
        self.expected_gl_coef = (gamma * alpha) / (
                    1 - gamma + (alpha * gamma))  # 水印文本中绿色列表token的期望数量（无水印文本中绿色token的比例是gamma）

        # catch for overflow when bias is "infinite"
        if alpha == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0

    def _get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]  # 初始化一个二维数组
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):  # 按批次遍历
            for ent_tensor in ent_tensor_list:  # 遍历张量list
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    # 获取当前存储的所有尖峰熵值，并清空存储
    def _get_and_clear_stored_spike_ents(self):
        spike_ents = self._get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    # 公式：S(p,z) = \sum{p/(1+zpk)}
    def _compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)  # 将logits转换为probs：[batch_size, vocab_size]
        denoms = 1 + (self.z_value * probs)  # 分母
        renormed_probs = probs / denoms  # 归一化概率
        sum_renormed_probs = renormed_probs.sum()  # 求和得到尖峰熵值
        return sum_renormed_probs

    # batched! scores和greenlist_token_ids都是批量的
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
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

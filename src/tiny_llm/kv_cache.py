from abc import ABC, abstractmethod
from typing import Optional

from .attention import causal_mask
import mlx.core as mx


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[TinyKvCache] = [None] * max_active_requests
        self.HD = None

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None, # mask_length 表示 本次查询（query）序列的长度，也就是本步要生成的 token 数（或要计算注意力的步长）。
        mask: mx.array | str | None = None, # attention mask, 注意力遮罩
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        # keys 和 values 的形状为 (B, H, S, D)
        # B: 批量大小，表示当前同时处理的请求数
        # H: 注意力头数
        # S: 序列长度（当前输入序列的长度）
        # D: 每个注意力头的维度大小
        B, H, S, D = keys.shape
        assert keys.shape == values.shape
        # 确保当前序列长度不超过最大序列长度限制
        assert S <= self.max_seq_len
        # 如果尚未设置HD维度，则初始化，否则检查维度一致性
        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D), f"expect {self.HD} but got {H, D}"
        # 确保批量大小与最大活跃请求数一致
        assert B == self.max_active_requests

        # Step 1: 逐个请求更新对应的kv缓存
        # 对于批量中的每个请求，调用其对应的kv_cache更新缓存并获取最新kv和mask信息
        data = []
        # 表示在遍历批量（batch）中的每一个独立请求。
        # 每个 b 对应的其实是一个 独立的上下文缓存 (TinyKvFullCache)。
        for b in range(B): 
            # 如果该请求没有对应缓存，跳过
            if self.kv_caches[b] is None:
                data.append(None)
                continue
            # 取出单个请求的key和value，形状为 (1, H, S, D)
            key, value = keys[b : b + 1], values[b : b + 1]
            # 调用对应的kv_cache更新缓存，返回更新后的key, value, 当前序列长度和mask
            new_key, new_value, seq_len, mask = self.kv_caches[b].update_and_fetch(
                key, value
            )
            # 将结果保存，注意这里解包成二维，方便后续拼接
            data.append((new_key[0], new_value[0], seq_len, mask))

        # Step 2: 计算整个批次中所有请求的最大序列长度seq_len
        # 这里的seq_len是所有请求缓存的最大长度，用于统一后续的拼接操作
        def get_seq_len(data):
            if data is None:
                return 0
            _, _, seq_len, _ = data
            return seq_len

        seq_len = max(map(get_seq_len, data))

        # Step 3: 生成统一大小的keys, values和mask张量
        # keys和values的形状为 (B, H, seq_len, D)，seq_len为最大序列长度
        # 这里通过右对齐拷贝的方式将每个请求的缓存数据对齐到seq_len的尾部
        # 这样做的目的是保证不同长度缓存的数据在时间维度上对齐，方便后续批量计算
        keys = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=key.dtype)
        values = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=value.dtype)
        # masks的形状为 (B, mask_length, seq_len)，初始化为负无穷，表示默认不允许注意
        masks = mx.full(
            (self.max_active_requests, mask_length, seq_len), -mx.inf, dtype=key.dtype
        )
        # 逐个请求填充keys, values和mask
        for b in range(B):
            # 如果该请求无缓存，直接生成因果掩码
            if data[b] is None:
                masks[b, :, :] = causal_mask(mask_length, seq_len, dtype=key.dtype)
                continue
            key, value, S, mask = data[b]
            # 右对齐拷贝：将长度为S的缓存数据拷贝到序列尾部 [seq_len - S : seq_len]
            # 这样保证了不同长度的缓存数据在时间维度上右对齐，方便mask和计算
            keys[b, :, seq_len - S : seq_len, :] = key
            values[b, :, seq_len - S : seq_len, :] = value
            # 根据mask类型选择生成对应的mask
            if mask is None or mask == "causal":
                # 生成因果掩码，只允许当前位置及之前位置的注意力
                masks[b, :, seq_len - S : seq_len] = causal_mask(
                    mask_length, S, dtype=key.dtype
                )
            elif isinstance(mask, mx.array):
                # 如果mask是mx.array，直接使用给定mask
                masks[b, :, seq_len - S : seq_len] = mask
            else:
                raise NotImplementedError
        # 将mask形状调整为 (B, 1, mask_length, seq_len)，方便后续广播计算
        return keys, values, None, masks.reshape(B, 1, mask_length, seq_len)

    def add_request(self, prefilled: TinyKvCache, id: int):
        # 添加一个新的请求缓存到批量缓存中，id表示请求索引
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        # 如果传入的缓存中包含key_values，提取其形状信息，用于维度一致性检查
        if getattr(prefilled, "key_values", None) is not None:
            keys, _ = prefilled.key_values
            B, H, _, D = keys.shape
            assert B == 1
            # 如果尚未设置HD维度，则初始化，否则检查维度一致性
            if self.HD is None:
                self.HD = (H, D)
            else:
                assert self.HD == (H, D)
        # 将该请求缓存加入缓存列表
        self.kv_caches[id] = prefilled
        
    def remove_request(self, id: int):
        # 移除指定id的请求缓存，释放资源
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if self.key_values is None:
            self.key_values = (key, value)
            B, H, S, D = key.shape
            self.offset = S # initialize the offset to the sequence length of the new key-value pair
            return key, value, self.offset, mask
        else:
            # Step 0: check the shape of the new key-value pair
            B, H, S, D = key.shape
            assert key.shape == value.shape
            # Step 1: concat the new key-value pair with the previous key-value pairs
            prev_keys, prev_values = self.key_values
            assert prev_keys.shape == (B, H, self.offset, D)
            assert prev_values.shape == (B, H, self.offset, D)
            # Step 2: concat the new key-value pair with the previous key-value pairs
            new_keys = mx.concat([prev_keys, key], axis=2)
            new_values = mx.concat([prev_values, value], axis=2)
            # Step 3: update the key-value cache and the offset
            self.key_values = (new_keys, new_values)
            self.offset += S
            # Step 4: return the updated key-value cache and the offset
            return new_keys, new_values, self.offset, mask

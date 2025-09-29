import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # step 1: calculate the scale
    # shape of query
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    # step 2: calculate the QK^T   
    # reverse the last two dimensions of key, for calculation of QK^T
    # 用于转置张量的最后两个维度，将 key 从 [B, H, S, D] 转成 [B, H, D, S]，以便和 query 点乘。
    attention_scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor

    # step 3: add the mask
    if mask is not None:
        attention_scores = attention_scores + mask

    # step 4: apply the softmax
    # # axis=-1 means the last dimension, 作用是将最后一个维度上的数值归一化成概率分布。
    attention_scores = softmax(attention_scores,axis=-1) 

    # step 5: calculate the output
    output = mx.matmul(attention_scores, value)
    return output



class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        pass

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        pass


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    # mx.tril is a function that creates a lower triangular matrix.
    # L, S is the shape of the mask
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    # mx.where is a function that replaces the values in the mask with 0 if the value is 1, otherwise replace with -mx.inf.
    # astype is a function that converts the mask to the dtype.
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    # step 1: calculate the scale
    # shape of query: B x H_q x L x D, H_q is the number of query heads, L is the sequence length of query, D is the dimension of the query
    # shape of key: B x H x S x D, H < H_q, S is the sequence length of key and value
    # shape of value: B x H x S x D, H of value is the same as key
    factor = mx.rsqrt(query.shape[-1]) if scale is None else scale
    factor = factor.astype(query.dtype)
    # shape of expected_shape: B x H_q x L x D
    expected_shape = query.shape

    # step 2: extract the shape of query and key and value
    H_q, L, D = query.shape[-3:]
    B = query.shape[:-3]
    # extract the shape of key and value
    H, S, _ = key.shape[-3:]
    # assert the number of heads of query and key are divisible
    assert H_q % H == 0
    n_repeats = H_q // H

    # step 3: reshape the query, key, and value
    # n_repeats is the number of times to repeat the key and value for each head
    # For queeryH is number of heads of query, n_repeats is the number of times to repeat the key and value for each head
    # L is the sequence length of query
    # *B is the batch size, why use *B? because the batch size is not fixed, it can be 1, 2, 3, etc.
    # -1 means the rest of the dimensions will be filled with the remaining dimensions of the query
    query = query.reshape(*B, -1, H, n_repeats, L, D) 
    # For key and value, H is number of heads of key and value, 1 is the number of times to repeat the key and value for each head
    # S is the sequence length of key and value
    key = key.reshape(*B, -1, H, 1, S, D) 
    value = value.reshape(*B, -1, H, 1, S, D)

    # step 4: calculate the attention scores
    attention_scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if mask == "causal":
            # handle the causal mask
            # 下三角矩阵，防止看到未来信息。
            mask = causal_mask(L, S, attention_scores.dtype)
            attention_scores = attention_scores + mask
        else:
            # handle the self-defined mask
            # could be any shape of mask
            mask = mx.broadcast_to(mask, (*B, H_q, L, S)) # broadcast the mask to the shape of query
            mask = mask.reshape(*B, 1, H, n_repeats, L, S) # reshape the mask to the shape of query
            attention_scores = attention_scores + mask
    # step 5: apply the softmax
    result = mx.matmul(softmax(attention_scores, axis=-1), value)
    # step 6: reshape the result to the expected shape
    result = result.reshape(expected_shape)
    return result


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass

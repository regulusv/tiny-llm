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
        hidden_size: int, # d_model
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        """
        一个标准的多头注意力（Multi-Head Attention, 多头注意力）实现：
        - 输入/输出形状: (N, L, d_model)
        - 头数: num_heads = h
        - 每头维度: head_dim = d_model // h
        - 权重形状:
        wq, wk, wv: (d_model, h * head_dim)
        wo:         (h * head_dim, d_model)
        """
        # Step 1: Store basic parameters
        # Store the hidden size and number of attention heads
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Step 2: Calculate head dimension and validate
        # Calculate head_dim = hidden_size // num_heads
        # Assert that hidden_size is divisible by num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0
        
        # Step 3: Calculate scale factor for attention
        # Calculate scale = 1 / sqrt(head_dim) for attention scaling
        self.scale = 1 / mx.rsqrt(self.head_dim)
        
        # Step 4: Validate weight matrix shapes
        # Assert that all weight matrices have correct shapes:
        # - wq: (hidden_size, num_heads * head_dim)
        # - wk: (hidden_size, num_heads * head_dim) 
        # - wv: (hidden_size, num_heads * head_dim)
        # - wo: (num_heads * head_dim, hidden_size)
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        
        # Step 5: Store weight matrices
        # Store all weight matrices as instance variables
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # Step 1: Extract input dimensions
        # Extract batch size (N) and sequence length (L) from query shape
        # Assert that query, key, value have the same shape
        pass
        
        # Step 2: Project Q, K, V to multi-head space
        # Apply linear transformation to Q, K, V using wq, wk, wv
        # Reshape from [N, L, hidden_size] to [N, L, num_heads, head_dim]
        # Transpose to [N, num_heads, L, head_dim] for attention computation
        pass
        
        # Step 3: Apply scaled dot product attention
        # Call scaled_dot_product_attention_simple with projected Q, K, V
        # Use the precomputed scale factor and optional mask
        pass
        
        # Step 4: Reshape and transpose back
        # Transpose from [N, num_heads, L, head_dim] to [N, L, num_heads, head_dim]
        # Reshape back to [N, L, hidden_size]
        pass
        
        # Step 5: Apply output projection
        # Apply linear transformation using wo to get final output
        # Return the final result with shape [N, L, hidden_size]
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
    # Step 1: Calculate scale factor
    # Calculate scale = 1 / sqrt(query.shape[-1]) if not provided
    # Convert scale to the same dtype as query
    pass
    
    # Step 2: Extract dimensions and validate
    # Extract batch dimensions (*B), query heads (H_q), sequence lengths (L, S), head dimension (E)
    # Assert that query heads are divisible by key/value heads for grouped query attention
    pass
    
    # Step 3: Reshape for grouped query attention
    # Reshape query, key, value to support grouped query attention
    # Flatten batch dimensions for efficient computation
    pass
    
    # Step 4: Prepare mask
    # Handle mask preparation for flash attention
    # If no mask provided, create zero mask
    # If mask provided, reshape to match the attention computation
    pass
    
    # Step 5: Call optimized flash attention kernel
    # Use the C++/Metal optimized flash attention implementation
    # This is typically implemented in extensions for performance
    pass
    
    # Step 6: Reshape output back to original shape
    # Reshape the result back to the expected output shape
    # Return the final attention output
    pass

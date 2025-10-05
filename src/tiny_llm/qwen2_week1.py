import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    """
    Multi-Head Attention implementation for Qwen2 model.
    
    This class implements Grouped Query Attention (GQA) where:
    - Query heads: num_heads (full attention heads)
    - Key/Value heads: num_kv_heads (shared across query heads)
    - Each KV head serves multiple Q heads (num_heads / num_kv_heads ratio)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        """
        Initialize Multi-Head Attention layer.
        init() 用于 接收模型结构的超参数（如 hidden_size, num_heads) 和预训练权重 (如 wq, wk, wv, wo) 并保存下来, 为后续推理做准备。
        
        Args:
            hidden_size: Dimension of the hidden state
            num_heads: Number of attention heads (query heads)
            num_kv_heads: Number of key-value heads (shared)
            wq, wk, wv, wo: Weight matrices for query, key, value, output projections
            bq, bk, bv: Bias vectors for query, key, value projections
                Output = XW + b, W is weight matrix, b is bias vector, X is input matrix
            max_seq_len: Maximum sequence length for RoPE
            theta: RoPE theta parameter
        """
        # Step 1 - Store basic parameters
        # Store the input parameters as instance variables
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        # Step 2 - Add validation checks
        # Check that hidden_size is divisible by num_heads
        # Check that num_heads is divisible by num_kv_heads
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        
        # Step 3 - Calculate derived parameters
        # Calculate head_dim = hidden_size // num_heads
        # Calculate scale = mx.rsqrt(head_dim) for attention scaling
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)

        
        # Step 4 - Store weight matrices and biases
        # Store all weight matrices (wq, wk, wv, wo) and biases (bq, bk, bv)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        
        # Step 5 - Initialize RoPE
        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Attention mask (can be "causal" string or tensor)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # Step 1 - Get input dimensions
        # Extract batch_size (B), sequence_length (L) from input shape
        B, L, _ = x.shape
        
        # Step 2 - Linear projections for Q, K, V
        # 我们必须先从输入向量 x 中投影出 Q/K/V 三个向量
        # Apply linear transformations with bias:
        projection_q = linear(x, self.wq, bias=self.bq)
        projection_k = linear(x, self.wk, bias=self.bk)  
        projection_v = linear(x, self.wv, bias=self.bv)
        
        # Step 3 - Reshape for multi-head attention
        # Reshape projections to separate heads:
        # projection_q: (B, L, num_heads, head_dim)
        # projection_k: (B, L, num_kv_heads, head_dim)
        # projection_v: (B, L, num_kv_heads, head_dim)
        projection_q = projection_q.reshape(B, L, self.num_heads, self.head_dim)
        projection_k = projection_k.reshape(B, L, self.num_kv_heads, self.head_dim)
        projection_v = projection_v.reshape(B, L, self.num_kv_heads, self.head_dim)
        
        # Step 4 - Apply RoPE positional encoding
        # Apply RoPE to query and key projections:
        projection_q = self.rope(projection_q, offset=slice(0, L))
        projection_k = self.rope(projection_k, offset=slice(0, L))
        
        # Step 5 - Transpose for attention computation
        # Transpose from (B, L, num_heads, head_dim) to (batch_size(B), num_heads, seq_len(L), head_dim):
        projection_q = projection_q.transpose(0, 2, 1, 3)
        projection_k = projection_k.transpose(0, 2, 1, 3)
        projection_v = projection_v.transpose(0, 2, 1, 3)
        
        # Step 6 - Apply scaled dot-product attention
        # Call scaled_dot_product_attention_grouped with:
        # - Convert to float32 for computation: .astype(mx.float32)
        # - Use self.scale for attention scaling
        # - Pass the mask parameter
        # - Convert result back to original dtype: .astype(x.dtype)
        x = scaled_dot_product_attention_grouped(
            projection_q.astype(mx.float32),
            projection_k.astype(mx.float32),
            projection_v.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(x.dtype) # float16, 统一类型，节省内存，便于残差连接等后续操作
        
        # Step 7 - Transpose back and reshape
        # Transpose back to (batch_size, seq_len, num_heads, head_dim)
        # Reshape to (batch_size, seq_len, hidden_size)
        # 把多个 head 的输出合并成原始的 hidden_size 维度, 恢复成了跟输入一样的维度（hidden_size 是 num_heads × head_dim），可以送入后续层。
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        
        # Step 8 - Apply output projection
        # Apply final linear transformation: linear(x, self.wo)
        # 
        return linear(x, self.wo)


#
# -----------------------------------------------------------------------------
# 原始 Transformer 中的 MLP 与 Qwen2 MLP 的区别说明：
#
# 【原始 Transformer 中的 MLP 结构】：
#     MLP(x) = W2(ReLU(W1 x)) 或 MLP(x) = W2(GELU(W1 x))
#
#   - W1: Linear projection from hidden_size → intermediate_size (通常是 4x 扩展)
#   - Activation: ReLU / GELU 非线性激活
#   - W2: Linear projection from intermediate_size → hidden_size（降维）
#
# 【Qwen2 中的 MLP 结构（使用 SwiGLU）】：
#     MLP(x) = W_down(SiLU(W_gate x) ⊙ W_up x)
#
#   - 使用三路投影：gate_proj、up_proj、down_proj
#   - SiLU 激活用于门控（gate）路径：SiLU(x) = x * sigmoid(x)
#   - ⊙ 表示逐元素乘法：SiLU(gate_proj(x)) * up_proj(x)
#   - 最后通过 down_proj 线性变换降维
#
# 【优势】：
#   - SwiGLU 比单一激活更强表达力，能建模更复杂的关系
#   - 实现细节上虽然增加了参数，但效果和收敛性更优
# -----------------------------------------------------------------------------
class Qwen2MLP:
    """
    Multi-Layer Perceptron (MLP) implementation for Qwen2 model.
    
    This implements a SwiGLU-style MLP with:
    - Gate projection: Linear transformation for gating
    - Up projection: Linear transformation for up-scaling  
    - Down projection: Linear transformation for down-scaling
    - SiLU activation function applied to gate projection
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        """
        Initialize MLP layer.
        
        Args:
            dim: Input/output dimension (hidden_size)
            hidden_dim: Hidden dimension (intermediate_size, typically 4x dim)
            w_gate: Weight matrix for gate projection (dim -> hidden_dim)
            w_up: Weight matrix for up projection (dim -> hidden_dim)
            w_down: Weight matrix for down projection (hidden_dim -> dim)
        """
        # Step 1 - Store dimensions
        # Store input/output dimension and hidden dimension
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Step 2 - Store weight matrices
        # Store the three weight matrices for gate, up, and down projections
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of MLP.
        
        Implements: down_proj(silu(gate_proj(x)) * up_proj(x))
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # Step 1 - Gate projection
        # Apply gate projection: gate = linear(x, self.w_gate)
        gate = linear(x, self.w_gate)
        
        # Step 2 - Up projection  
        # Apply up projection: up = linear(x, self.w_up)
        up = linear(x, self.w_up)
        
        # Step 3 - Apply SiLU activation
        activated_gate = silu(gate)
        
        # Step 4 - Element-wise multiplication
        # Multiply activated gate with up projection: intermediate = activated_gate * up
        intermediate = activated_gate * up
        
        # Step 5 - Down projection
        # Apply down projection: output = linear(intermediate, self.w_down)
        output = linear(intermediate, self.w_down)
        
        # Step 6 - Return result
        return output


class Qwen2TransformerBlock:
    """
    Transformer Block implementation for Qwen2 model.
    
    This block implements the standard transformer architecture with:
    - Pre-norm attention (RMSNorm before attention)
    - Post-norm MLP (RMSNorm before MLP)
    - Residual connections around both attention and MLP
    - Causal attention mask
    """
    
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        """
        Initialize Transformer Block.
        
        Args:
            num_attention_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (for GQA)
            hidden_size: Hidden dimension size
            intermediate_size: MLP intermediate dimension (typically 4x hidden_size)
            rms_norm_eps: Epsilon for RMS normalization
            wq, wk, wv, wo: Attention weight matrices
            bq, bk, bv: Attention bias vectors
            w_gate, w_up, w_down: MLP weight matrices
            w_input_layernorm: Input layer norm weight
            w_post_attention_layernorm: Post-attention layer norm weight
            max_seq_len: Maximum sequence length for RoPE
            theta: RoPE theta parameter
        """
        # TODO: Step 1 - Store attention parameters
        # Store num_attention_heads and hidden_size for later use
        # self.num_attention_heads = num_attention_heads
        # self.hidden_size = hidden_size
        
        # TODO: Step 2 - Initialize MLP
        # Create MLP instance: self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        
        # TODO: Step 3 - Initialize Layer Normalizations
        # Create input layer norm: self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        # Create post-attention layer norm: self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, eps=rms_norm_eps)
        
        self.self_attn = Qwen2MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            bq=bq,
            bk=bk,
            bv=bv,
            max_seq_len=max_seq_len,
            theta=theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """
        Forward pass of Transformer Block.
        
        Implements: x + MLP(RMSNorm(x + Attention(RMSNorm(x))))
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Attention mask (typically "causal")
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        # TODO: Step 1 - Pre-attention normalization and attention
        # Apply input layer norm: normed_x = self.input_layernorm(x)
        # Apply attention: attention_output = self.self_attn(normed_x, mask)
        # Add residual connection: h = x + attention_output
        
        # TODO: Step 2 - Pre-MLP normalization and MLP
        # Apply post-attention layer norm: normed_h = self.post_attention_layernorm(h)
        # Apply MLP: mlp_output = self.mlp(normed_h)
        # Add residual connection: out = h + mlp_output
        
        # TODO: Step 3 - Return result
        # Return the final output
        pass


class Qwen2ModelWeek1:
    """
    Complete Qwen2 Model implementation for Week 1.
    
    This class assembles all components into a complete transformer model:
    - Token embedding layer
    - Multiple transformer blocks (layers)
    - Final layer normalization
    - Language modeling head (output projection)
    """
    
    def __init__(self, mlx_model: Any):
        """
        Initialize the complete Qwen2 model from MLX model.
        
        Args:
            mlx_model: Pre-loaded MLX model containing weights and configuration
        """
        # TODO: Step 1 - Extract model configuration
        # Extract key parameters from mlx_model.args:
        # self.num_hidden_layers = mlx_model.args.num_hidden_layers
        # self.hidden_size = mlx_model.args.hidden_size
        # self.vocab_size = mlx_model.args.vocab_size
        
        # TODO: Step 2 - Set precision
        # Set precision for computations (typically float16):
        # precision = mx.float16
        # self.precision = precision
        
        # TODO: Step 3 - Initialize embedding layer
        # Create embedding layer with dequantized weights:
        # self.embedding = Embedding(
        #     vocab_size=self.vocab_size,
        #     embedding_dim=self.hidden_size,
        #     weight=dequantize_linear(mlx_model.model.embed_tokens).astype(precision)
        # )
        
        # TODO: Step 4 - Initialize transformer layers
        # Create list to store transformer blocks: self.layers_inner = []
        # Loop through each layer (range(mlx_model.args.num_hidden_layers)):
        #   - Extract and dequantize attention weights (wq, wk, wv, wo)
        #   - Extract and dequantize MLP weights (w_gate, w_up, w_down)
        #   - Extract attention biases (bq, bk, bv)
        #   - Extract layer norm weights (input_layernorm, post_attention_layernorm)
        #   - Create Qwen2TransformerBlock with all parameters
        #   - Append to self.layers_inner
        
        # TODO: Step 5 - Initialize final layer normalization
        # Create final RMSNorm layer:
        # self.norm = RMSNorm(
        #     mlx_model.args.hidden_size,
        #     weight=mlx_model.model.norm.weight.astype(precision),
        #     eps=mlx_model.args.rms_norm_eps
        # )
        
        # TODO: Step 6 - Initialize language modeling head
        # Check if word embeddings are tied:
        # if not mlx_model.args.tie_word_embeddings:
        #     self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        # else:
        #     self.w_lm_head = None
        
        # TODO: Step 7 - Store reference to original model
        # Store mlx_model for potential future use: self.mlx_model = mlx_model
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        """
        Forward pass of the complete Qwen2 model.
        
        Args:
            inputs: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # TODO: Step 1 - Token embedding
        # Convert token IDs to embeddings: h = self.embedding(inputs)
        
        # TODO: Step 2 - Pass through transformer layers
        # Loop through each transformer layer:
        # for layer in range(self.num_hidden_layers):
        #     h = self.layers_inner[layer](h, mask="causal")
        
        # TODO: Step 3 - Final layer normalization
        # Apply final normalization: h = self.norm(h)
        
        # TODO: Step 4 - Language modeling head
        # Apply output projection:
        # if self.w_lm_head is not None:
        #     return linear(h, self.w_lm_head)
        # else:
        #     return self.embedding.as_linear(h)
        
        # TODO: Step 5 - Return logits
        # Return the final logits
        pass

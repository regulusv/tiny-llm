import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        # Step 1: Validate inputs
        # - Ensure `dims` is even (required by rotary embeddings)
        # - Store constructor arguments on `self`
        assert dims % 2 == 0, "dims must be even"
        self.dims = dims
        self.seq_len = seq_len

        # Step 2: Precompute frequency matrix
        # - Compute half dimension: half_dims = dims // 2
        # - Create `inner` as mx.arange(0, half_dims, dtype=mx.float32) / half_dims
        # - Compute `freqs` as base ** (-inner)
        # - Create positions `t = mx.arange(seq_len)`
        # - Build outer product: freqs = mx.outer(t, freqs)
        half_dims = dims // 2

        # 下面构造用于 RoPE（Rotary Positional Encoding，旋转位置编码）的角度矩阵。
        # 目标：为每个位置 pos 和通道索引 i 计算角度
        #   \( \theta_{pos,i} = pos \times base^{-2i/d} \)
        # 其中 d = dims，half_dims = d/2，i ∈ [0, half_dims-1]。
        # inner[i] = i / half_dims = 2i/d，用来构造指数 -2i/d（用于 base 的幂运算）。
        inner = mx.arange(0, half_dims, dtype=mx.float32) / half_dims
        
        # - mx.arange(0, half_dims) 产生 [0,1,...,half_dims-1]（索引向量 index vector）。
        # - 除以 half_dims 后，相当于得到 [0, 1/half_dims, ..., (half_dims-1)/half_dims]，
        #   即数学上的 2i/d（因为 half_dims = d/2）。

        # freqs[i] = base ** (-inner[i]) = base^{-2i/d}, base = 10000
        # 这里得到每个通道的“逆频率” inverse frequency（逆频率）。
        freqs = mx.power(base, -inner)
        # - 这对应经典公式 inv_freq_i = 1 / base^{2i/d}（当 base=10000 时即 Transformer 的做法）。

        # t 是位置索引向量 t = [0, 1, ..., seq_len-1]
        t = mx.arange(seq_len)
        # - 每一行代表序列中的某个位置 pos（position index）。

        # 外积得到形状 (seq_len, half_dims):
        # freqs_matrix[pos, i] = pos * base^{-2i/d}
        # 这正是用于后续取 cos/sin 的角度矩阵 θ_{pos,i}
        freqs = mx.outer(t, freqs)

        # - mx.outer(t, freqs) 计算外积（outer product，外积矩阵），得到每个位置 pos
        #   和每个通道 i 对应的角度值 pos * inv_freq_i。
        # - 最终结果形状为 (seq_len, half_dims)，方便后面做 self.cos_freqs = cos(freqs)、self.sin_freqs = sin(freqs)。
        # Step 3: Cache sin/cos tables for all sequence positions
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)

        # Step 4: Store aux attributes
        self.base = base
        self.half_dims = half_dims
        self.traditional = traditional

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # Expected input shape: x is (batch, seq, num_heads, head_dim)
        # Step 1: Unpack shape
        N, S, H, D = x.shape

        # Step 2: Handle offset (for cached decoding)
        # - If offset is a slice: assert length == S
        # - If offset is a list of slices (per batch item):
        #   * assert len == N and each slice length == S
        #   * convert to an array of position indices with mx.array([...])
        if offset is not None:
            if isinstance(offset, slice):
                assert offset.stop - offset.start == S, f"offset must be of length {S}"
            elif isinstance(offset, list):
                assert len(offset) == N, f"offsets must have the same length as batch size {N}"
                for o in offset:
                    assert o.stop - o.start == S, f"offset must be of length {S}"

        # Step 3: Select rotary bases for the current positions
        cos_basis = self.cos_freqs[:S, :] if offset is None else self.cos_freqs[offset, :]
        sin_basis = self.sin_freqs[:S, :] if offset is None else self.sin_freqs[offset, :]

        # Step 4: Split the last dimension into two halves
        #   If self.traditional:
        #     - reshape x to (N, S, H, half_dims, 2) and take x1=x[...,0], x2=x[...,1]
        #   Else:
        #     - x1 = x[..., :half_dims]; x2 = x[..., half_dims:]
        if self.traditional:
            x = x.reshape(N, S, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1 = x[..., 0 : self.half_dims]
            x2 = x[..., self.half_dims : self.dims]

        # Step 5: Reshape bases for broadcasting
        cos_basis = cos_basis.reshape(-1, S, 1, self.half_dims)
        sin_basis = sin_basis.reshape(-1, S, 1, self.half_dims)

        # Step 6: Apply complex rotation (manual real/imag multiply-add)
        real = x1 * cos_basis - x2 * sin_basis
        imag = x2 * cos_basis + x1 * sin_basis

        # Step 7: Recombine halves and restore original shape
        if self.traditional:
            y = mx.stack([real, imag], axis=-1).reshape(N, S, H, D)
        else:
            y = mx.concat([real, imag], axis=-1).reshape(N, S, H, D)

        # Step 8: Cast back to input dtype and return
        return y.astype(x.dtype)
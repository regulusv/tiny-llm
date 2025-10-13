import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        """
        初始化嵌入层
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            weight: 嵌入权重矩阵，形状为 (vocab_size, embedding_dim), 对应每个token的embedding向量
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight 

    def __call__(self, x: mx.array) -> mx.array:
        """
        将token ID转换为嵌入向量
        
        Args:
            x: token ID张量，形状为 (batch_size, seq_len)
            
        Returns:
            嵌入向量张量，形状为 (batch_size, seq_len, embedding_dim)
        """
        # 使用索引操作从权重矩阵中获取对应的嵌入向量
        # x中的每个token ID对应weight矩阵中的一行
        return self.weight[x]

    def as_linear(self, x: mx.array) -> mx.array:
        """
        将嵌入向量转换为logits（用于语言建模头）
        
        Args:
            x: 嵌入向量张量，形状为 (batch_size, seq_len, embedding_dim)
            
        Returns:
            logits张量，形状为 (batch_size, seq_len, vocab_size)
        """
        # 使用矩阵乘法计算logits
        # x @ weight.T 将嵌入向量映射回词汇表空间
        return mx.matmul(x, self.weight.T)

import mlx.core as mx


class RMSNorm:
    """
    Root Mean Square Normalization (RMSNorm) implementation.
    
    RMSNorm是一种比LayerNorm更简单、更高效的归一化方法：
    
    理论背景：
    1. LayerNorm: 对每个样本的最后一维进行归一化，公式为：
       y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
       
    2. RMSNorm: 只使用均方根进行归一化，不需要计算均值，公式为：
       y = x / sqrt(mean(x^2) + eps) * weight
       
    优势：
    - 计算更简单（不需要计算均值）
    - 训练更稳定
    - 在某些任务上性能更好
    - 减少计算开销
    
    应用场景：
    - Transformer模型中的层归一化
    - 替代LayerNorm的轻量级选择
    - 特别是在大模型训练中广泛使用
    """
    
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        """
        初始化RMSNorm层。
        
        Args:
            dim: 输入特征的维度
            weight: 可学习的缩放权重，形状为 (dim,)
            eps: 防止除零的小常数，默认1e-5
        """
        # Step 1: 存储基本参数
        # 保存输入维度和epsilon值
        self.dim = dim          # 特征维度，如4096
        self.eps = eps          # 防止除零的小常数
        
        # Step 2: 处理权重
        # 将权重转换为float32以确保计算精度
        # 权重形状: (dim,) - 每个特征维度有一个缩放因子
        self.weight = weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        """
        RMSNorm前向传播。
        
        实现公式: y = x / sqrt(mean(x^2) + eps) * weight
        
        Args:
            x: 输入张量，形状为 (..., dim)
            
        Returns:
            归一化后的张量，形状与输入相同
        """
        # Step 1: 保存原始数据类型
        # 记录输入的数据类型，最后需要转换回来
        orig_dtype = x.dtype
        
        # Step 2: 转换为float32进行计算
        # 使用float32确保数值稳定性，避免精度损失
        x = x.astype(mx.float32)
        
        # Step 3: 计算均方根 (Root Mean Square)
        # mean(x^2) 计算最后一个维度的平方均值
        # 形状: (..., 1) - 保持除最后一维外的所有维度
        mean_squared = mx.mean(mx.square(x), axis=-1, keepdims=True)
        
        # Step 4: 计算归一化因子
        # rsqrt = 1/sqrt，即平方根的倒数
        # 加上eps防止除零
        norm_factor = mx.rsqrt(mean_squared + self.eps)
        
        # Step 5: 应用归一化
        # x * norm_factor 将输入归一化到单位方差
        normalized_x = x * norm_factor
        
        # Step 6: 应用可学习权重
        # weight * normalized_x 允许模型学习每个特征的重要性
        # 权重广播到输入的所有维度
        output = self.weight * normalized_x
        
        # Step 7: 转换回原始数据类型
        # 保持与输入相同的数据类型（如float16）
        return output.astype(orig_dtype)

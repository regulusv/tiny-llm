import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    # 实现仿射变换: y = x @ W^T + b
    # 其中 x 形状为 [*, in_features]
    #     w 形状为 [out_features, in_features]
    # 返回形状与 x 的前缀维度一致，最后维度为 out_features
    y = mx.matmul(x, w.T)
    if bias is not None:
        y = y + bias
    return y


def silu(x: mx.array) -> mx.array:
    # SiLU 激活: x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1 + mx.exp(-x))

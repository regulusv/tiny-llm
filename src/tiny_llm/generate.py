import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        """
        单步生成：根据当前token序列预测下一个token
        
        参数:
            model: 语言模型
            y: 当前已生成的所有token序列 [seq_len] (包括prompt + 已生成的tokens)
        
        返回:
            y: 下一个token [1]
        """
        
        # 1. 模型推理：输入token序列，输出每个位置的下一个token预测分数
        # y是什么：当前已生成的所有token序列（包括初始prompt + 已生成的tokens）
        # 例如：prompt="今天天气" -> y=[101, 102, 103, 104] (4个token)
        # 为什么需要添加batch维度：
        #   - 神经网络模型期望输入格式：[batch_size, sequence_length]
        #   - MLX框架要求张量至少有2个维度
        #   - 保持与训练时数据格式的一致性
        # 维度变化：
        #   y: [4]           - 1维数组，包含4个token ID
        #   y[None]: [1,4]   - 添加batch维度，变成2维
        #   logits: [1,4,50000] - 模型输出，每个位置对50000个词汇的预测分数
        logits = model(y[None])
        
        # 2. 取最后一个位置的预测：我们只关心下一个token
        # 为什么只取最后一个位置：
        #   - 自回归生成：每次只预测一个token
        #   - 基于完整历史序列预测下一个token
        # 维度变化：logits[1,4,50000] -> logits[1,50000]
        logits = logits[:, -1, :]
        
        # 3. 转换为对数概率：数值稳定的softmax
        # 为什么用log概率：
        #   - 数值稳定性：避免exp(x)溢出
        #   - 计算效率：log空间中的运算更稳定
        # 维度变化：logits[1,50000] -> logprobs[1,50000]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        
        # 4. 采样：选择下一个token
        # 采样策略：
        #   - 贪心采样(sampler=None)：选择概率最高的token，确定性输出
        #   - 自定义采样(sampler函数)：根据概率分布随机采样，创造性输出
        if sampler is None:
            y = mx.argmax(logprobs, axis=-1)        
        else:
            y = sampler(logprobs)
        return y

    # 1. 编码：将文本转换为token序列
    # 将用户输入的prompt文本转换为模型可以理解的token ID序列
    # 例如："今天天气" -> [101, 102, 103, 104]
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    
    # 2. 初始化：准备实时显示生成的文本
    # detokenizer用于将token ID转换回文本，并实时显示生成过程
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    
    # 3. 生成循环：自回归生成文本
    # 自回归：每次基于完整历史序列预测下一个token，然后将其加入序列
    while True:
        # 预测下一个token
        # 基于当前所有tokens预测下一个token
        token = _step(model, tokens)
        
        # 执行计算（MLX需要显式eval）
        # MLX使用延迟计算，需要显式调用eval()来执行计算
        mx.eval(token)
        
        # 将新token添加到序列
        # 将预测的token添加到现有序列中，用于下次预测
        tokens = mx.concat([tokens, token])
        
        # 检查是否结束
        # 如果遇到结束token(EOS)，停止生成
        if token.item() == tokenizer.eos_token_id:
            break
        
        # 实时显示生成的文本
        # 将新token转换为文本并立即显示，让用户看到生成过程
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)
    
    # 4. 返回完整生成文本
    # 返回完整的生成结果（包括prompt + 生成的文本）
    return detokenizer.text


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass

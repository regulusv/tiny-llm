import mlx.core as mx
import copy

# 这个文件要做的是构建一个“采样器”（sampler）：
# 给定模型输出的每个词的对数概率（log-probabilities），决定下一步选哪个词。
#
# make_sampler 的输入参数：
# - temp（温度）：控制随机性。0 表示不随机（总是选概率最高的那个）。
# - top_p（核采样）：只保留累计概率刚好达到 top_p 的最小一组高概率词。
# - top_k：只保留概率最高的前 top_k 个词（例如 40 就是只留 40 个）。
#
# 采样流程（带有随机性的情况）分为以下步骤：
# Step 1：先复制一份 logprobs，避免直接改动输入数据。
# Step 2：如果设置了 top_k，就把除前 top_k 以外的词都“屏蔽掉”。
# Step 3：如果设置了 top_p，就只保留累计概率达到 top_p 的那一小部分高概率词。
# Step 4：用温度进行缩放（logprobs / temp）。温度越高，越随机；越低，越保守。
# Step 5：按照调整后的分布，随机抽取一个词（类别采样）。
#
# 说明：logprobs 的形状是 [batch, vocab_size]，表示一批样本，每个样本都有一个词表大小的分布。

def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        # Step 1：复制 logprobs，安全修改
        # Step 2：如果用了 top_k，只保留前 top_k 个词
        # Step 3：如果用了 top_p，只保留累计概率达到 top_p 的高概率词
        # Step 4：用温度缩放（logprobs / temp）
        # Step 5：按调整后的分布做类别采样
        pass

    return sample

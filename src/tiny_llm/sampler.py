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
        logprobs = copy.copy(logprobs)
        # Step 2：如果用了 top_k，只保留前 top_k 个词
        if top_k is not None and top_k > 0:
            # 通过对负的 logprobs 做部分排序，找出每行前 top_k 的位置
            mask_elements = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[:, top_k:]
            # 把不在前 top_k 的位置设为 -inf，表示被屏蔽
            logprobs[:, mask_elements] = -mx.inf
        # Step 3：如果用了 top_p，只保留累计概率达到 top_p 的高概率词
        if top_p is not None and top_p > 0:
            # 先按概率降序排序，计算 softmax 概率的累计和
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            # 先 exp 把 log 概率还原为概率，再沿着最后一维做“前缀和”（从左到右累加），得到累计概率。第 j 个位置的值表示“前 j+1 个最高概率词”的概率总和，用于 top-p（核采样）判断该保留到哪里。
            # 形状: 与输入相同 [batch, vocab]。
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            # 保留累计概率小于 top_p 的位置；并确保至少保留第一个（最高概率）
            # 按元素逐个比较，得到一个布尔数组。位置 i 为 True 表示“累计概率 cumsum[i] 仍然小于 top_p，需要保留”；为 False 表示“超过或等于 top_p，可以屏蔽”。
            mask_elements = cumsum < top_p
            # “省略号”... 表示前面所有维度不变，仅把最后一维的第 0 列（每一行的第一个元素）设为 True。 因为即使 top_p 很小，也至少保留每一行中概率最高的那个 token，避免把最高概率项也屏蔽掉。
            mask_elements[..., 0] = True
            # 其余位置设置为 -inf
            # 把“按降序重排后的每行值”或 -inf（根据掩码）写回到原来的 logprobs 中，但写回的位置是“每行的降序索引位置”。
            # logprobs[:, sorted_idx] 会“按每一行自己的索引顺序sorted_idx重排列”。
            # a[:, x] 表示“取 a 的所有行（:），在列维度用索引 x 取指定的列”。如果 x 是标量/一维整数数组，就等价于“选出这些列”,(如果在MLX/NumPy 里，这属于“高级索引”)每一行都按照自己那一行的索引顺序重排。 
            # where() 函数用于根据条件选择元素，根据 mask_elements 中的布尔值选择 sorted_logprobs 或 -mx.inf
            logprobs[:, sorted_idx] = mx.where(mask_elements, sorted_logprobs, -mx.inf)
        # Step 4：用温度缩放（logprobs / temp）
        logprobs = logprobs / temp
        # Step 5：按调整后的分布做类别采样
        return mx.random.categorical(logprobs, axis=-1) # categroical() 函数用于从给定的概率分布中随机采样一个类别。

    return sample

import torch
from model import TransformerEncoder

# 1. 配置参数
vocab_size = 1000   # 词汇表大小
embed_dim = 16      # 每个词转换成16维向量
num_heads = 4       # 注意力头数（16能被4整除）
num_layers = 2      # 堆叠2层结构
seq_len = 3         # 句子长度

# 2. 实例化模型
model = TransformerEncoder(vocab_size, embed_dim, num_heads, num_layers)

# 3. 准备模拟输入
# 使用一个 Batch，里面有一句话，包含3个词 (ID 分别为 5, 10, 2)
input_indices = torch.tensor([[5, 10, 2]], dtype=torch.long)

# 4. 运行模型 (前向传播)
output = model(input_indices)

# 5. 查看结果
print("模型输出形状:", output.shape) 
# 预期输出: torch.Size([1, 3, 1000])

# 6. 简单的分析：模型为这3个词分别在词表中预测出的“倾向”
# 每一行代表一个位置的预测分布
print("第一个词预测出的原始数值(Logits):", output[0, 0, :5].detach().numpy())
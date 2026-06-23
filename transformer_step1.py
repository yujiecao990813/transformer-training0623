import torch
import torch.nn as nn

# 假设词表里有 1000 个不同的词
# 每个词用 16 维的向量来表示 (Transformer 实际模型通常是 512 或更多)
vocab_size = 1000
embed_dim = 16

# 定义嵌入层
embedding_layer = nn.Embedding(vocab_size, embed_dim)

# 假设这是输入的句子，用词在字典里的 ID 表示（比如第 5 号词，第 10 号词）
input_indices = torch.tensor([5, 10, 2])

# 将词转换为向量
output_vectors = embedding_layer(input_indices)
import torch
import torch.nn as nn
import torch.nn.functional as F

# 维度设置
embed_dim = 16  # 每个词的向量维度
seq_len = 3     # 刚才输入的词数

# 1. 模拟刚才嵌入层输出的向量 (X)
X = torch.rand(seq_len, embed_dim)

# 2. 定义三个线性变换矩阵 (Wq, Wk, Wv)
# 这就是模型在训练过程中要学习的“参数”
Wq = nn.Linear(embed_dim, embed_dim, bias=False)
Wk = nn.Linear(embed_dim, embed_dim, bias=False)
Wv = nn.Linear(embed_dim, embed_dim, bias=False)

# 3. 计算 Q, K, V
Q = Wq(X)
K = Wk(X)
V = Wv(X)

# 4. 计算注意力分数 (Attention Scores)
# Q * K转置，计算词与词之间的关联度
scores = torch.matmul(Q, K.transpose(-2, -1)) / (embed_dim ** 0.5)

# 5. 用 Softmax 归一化 (让关联度总和为 1)
attention_weights = F.softmax(scores, dim=-1)

# 6. 计算最终输出
output = torch.matmul(attention_weights, V)

print("注意力机制输出形状:", output.shape) # 应该是 [3, 16]

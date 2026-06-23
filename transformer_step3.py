import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 将所有头的 Wq, Wk, Wv 放在一个线性层里计算，效率更高
        self.qkv_layer = nn.Linear(embed_dim, embed_dim * 3)
        self.out_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        seq_len, embed_dim = x.shape
        qkv = self.qkv_layer(x) # 得到 [seq_len, embed_dim * 3]
        
        # 拆分出 Q, K, V，并改变形状以便多头计算
        qkv = qkv.reshape(seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1) # 拆分成 3 份
        
        # 注意力计算 (简化版单头计算逻辑)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # 合并所有的头
        out = out.reshape(seq_len, embed_dim)
        return self.out_layer(out)

# 测试代码
model = MultiHeadAttention(embed_dim=16, num_heads=4)
x = torch.rand(3, 16)
output = model(x)
print("多头注意力输出形状:", output.shape)
# 这就是 Transformer 里的那个“前馈网络”
feed_forward = nn.Sequential(
    nn.Linear(16, 64), # 扩张维度
    nn.ReLU(),          # 非线性激活
    nn.Linear(64, 16)  # 回归原维度
)
import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerEncoder

# 1. 超参数设置
vocab_size = 100
embed_dim = 16
num_heads = 4
num_layers = 2
learning_rate = 0.01

# 2. 实例化模型
model = TransformerEncoder(vocab_size, embed_dim, num_heads, num_layers)

# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 模拟训练数据
# 输入：[1, 2, 3] -> 目标：[2, 3, 4]
input_seq = torch.tensor([[1, 2, 3]], dtype=torch.long)
target_seq = torch.tensor([[2, 3, 4]], dtype=torch.long)

# 5. 训练循环
print("开始训练...")
for epoch in range(100): # 训练 100 次
    optimizer.zero_grad()            # 清空之前的梯度
    output = model(input_seq)        # 前向传播：预测
    
    # 计算损失：将模型输出展平，与目标序列对比
    loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
    
    loss.backward()                  # 反向传播：计算梯度
    optimizer.step()                 # 更新参数
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("训练完成！模型已经学会了简单的数字加法逻辑。")
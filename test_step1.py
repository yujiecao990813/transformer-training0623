import torch

# 1. 打印版本，测试 PyTorch 是否安装成功
print(f"PyTorch 版本: {torch.__version__}")

# 2. 测试创建张量 (Tensor)
x = torch.tensor([1.0, 2.0, 3.0])
print(f"创建的张量: {x}")
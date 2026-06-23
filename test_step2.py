import torch

# 1. 定义一个需要求导的变量 x
# 我们设置 requires_grad=True，相当于告诉 PyTorch：请帮我记录所有关于 x 的运算
x = torch.tensor([2.0], requires_grad=True)

# 2. 定义函数 y = x^3
# (这意味着 y = 2^3 = 8)
y = x**3

# 3. 反向传播：告诉 PyTorch 开始算导数
y.backward()

# 4. 查看 x 的梯度 (导数)
# y = x^3 的导数是 3*x^2。当 x=2 时，导数应该是 3*(2^2) = 12
print(f"y 的值: {y.item()}")
print(f"x 的导数 (梯度): {x.grad.item()}")

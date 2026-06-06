# ---
# chapter: 11
# topic: 深度学习与PyTorch
# section: 11.2.2 Autograd 自动求导
# difficulty: ⭐⭐⭐⭐⭐
# tier: core
# deps: torch
# run: python 02_autograd.py
# expected_runtime: <5s
# expected_output: 标量/向量/矩阵梯度, requires_grad 状态
# ---
# See: ../tutorial/11_深度学习与PyTorch.md#11.2.2-autograd-自动求导
#
# Interview hooks:
#  1. requires_grad=True 的 Tensor 调用 backward() 后, grad 是否会累加? 如何清零?
#  2. torch.no_grad() 与 torch.inference_mode() 的区别? 性能/使用场景差异?
#  3. 解释 PyTorch 动态计算图的构建与回溯机制 (topological order)。
import torch

# ========== 基础求导 ==========
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x

# 反向传播
y.backward()
print(f"x = {x.item()}, y = {y.item()}")
print(f"dy/dx = {x.grad.item()}")  # 3*x^2 + 2 = 3*4 + 2 = 14

# ========== 多元函数梯度 ==========
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sum() * 2  # y = 2*(x1+x2+x3)
y.backward()
print(f"梯度: {x.grad}")  # 每个分量都是 2

# ========== 矩阵梯度 ==========
X = torch.randn(3, 4, requires_grad=True)
W = torch.randn(4, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# 前向: Y = XW + b
Y = torch.matmul(X, W) + b
loss = Y.sum()
loss.backward()

print(f"∂L/∂X shape: {X.grad.shape}")  # (3, 4)
print(f"∂L/∂W shape: {W.grad.shape}")  # (4, 2)
print(f"∂L/∂b shape: {b.grad.shape}")  # (2,)

# ========== 关闭梯度追踪 ==========
# 推理时不需要计算梯度，节省内存和计算
with torch.no_grad():
    pred = torch.matmul(X, W) + b
    # pred.backward()  # 报错！在 no_grad 上下文中无法求导

# 或使用 torch.inference_mode()（PyTorch 1.9+ 推荐）
with torch.inference_mode():
    pred = torch.matmul(X, W) + b


if __name__ == "__main__":
    print("OK")

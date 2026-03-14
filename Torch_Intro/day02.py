import torch

# 1. 定义参数 w
w = torch.tensor(data=10, requires_grad=True, dtype=torch.float32)

# 2. 定义损失函数
loss = w ** 2 + 20

# 3. 梯度下降迭代 30 次
for i in range(1, 31):

    # 3.1 前向计算
    loss = w ** 2 + 20

    # 3.2 梯度清零（PyTorch 默认梯度会累加）
    if w.grad is not None:
        w.grad.zero_()

    # 3.3 反向传播
    loss.sum().backward()

    # 3.4 梯度更新
    w.data = w.data - 0.01 * w.grad

    # 3.5 打印结果
    print(f'第 {i} 次, 权重值: {w}, (0.01 * w.grad): {0.01 * w.grad}, loss: {loss}')
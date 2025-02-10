import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据
x = torch.linspace(-np.pi, np.pi, 100)
y = torch.sin(x)

# 2. 定义模型 (线性回归 + 基函数)
class FourierRegression(nn.Module):
    def __init__(self, n_basis=10):
        super(FourierRegression, self).__init__()
        self.a = nn.Parameter(torch.randn(n_basis))
        self.b = nn.Parameter(torch.randn(n_basis))
        self.n_basis = n_basis

    def forward(self, x):
        output = torch.zeros_like(x)
        for i in range(self.n_basis):
            output += self.a[i] * torch.sin((i+1) * x) + self.b[i] * torch.cos((i+1) * x)
        return output

# 3. 训练模型
model = FourierRegression(n_basis=10)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

n_epochs = 1000
for epoch in range(n_epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

# # 4. 可视化结果
# with torch.no_grad():
#     y_pred = model(x)

# plt.plot(x.numpy(), y.numpy(), label='sin(x)')
# plt.plot(x.numpy(), y_pred.numpy(), label='Fourier Approximation')
# plt.legend()
# plt.show()

for i in range(100):
    print(f"sin(${i}*pi)= {model(torch.tensor([i*np.pi])).item()}")
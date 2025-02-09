import torch
import numpy as np

# 生成数据
x = torch.linspace(1, 10, 100)
y = 2 * x ** 3 + torch.randn(x.size()) * 5  # 假设真实关系为 y = 2x^3 + 噪声

# 定义模型
class PowerModel(torch.nn.Module):
    def __init__(self):
        super(PowerModel, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.a * x ** self.b

# 初始化模型和优化器
model = PowerModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# 训练模型
for epoch in range(3000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print(f'Learned parameters: a={model.a.item()}, b={model.b.item()}')

import torch

class MultilevelLinearModel(torch.nn.Module):
    def f(x: float) -> float:
        return x ** 0.5
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.m(x)
    
# 生成数据
X = torch.tensor([[i] for i in range(10)], dtype=torch.float32)
Y = torch.tensor([[MultilevelLinearModel.f(i)] for i in range(10)], dtype=torch.float32)

# 初始化模型和优化器
model = MultilevelLinearModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# 训练模型
for epoch in range(3000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print(f'Learned parameters: {model.state_dict()}')
print(model(torch.tensor([[1], [4], [9]], dtype=torch.float32)))
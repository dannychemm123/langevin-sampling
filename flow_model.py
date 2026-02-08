import torch
import torch.nn as nn

# 1. Neural network = velocity field
class VelocityNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim)
        )

    def forward(self, x):
        return self.net(x)

# 2. Simple ODE solver (Euler)
def flow_step(x, v_net, dt):
    return x + dt * v_net(x)

# 3. Initialize
dim = 2
v_net = VelocityNet(dim)

# Initial distribution (Gaussian noise)
x = torch.randn(1000, dim)

# 4. Flow particles forward in time
T = 1.0
n_steps = 100
dt = T / n_steps

for _ in range(n_steps):
    x = flow_step(x, v_net, dt)

# x is now X_T
print(x.shape)

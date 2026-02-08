import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# 1. Toy data
# -----------------------------
torch.manual_seed(0)
N = 300

# Initial distribution: Gaussian
x0 = torch.randn(N, 2)

# Target distribution: circle
theta = 2 * np.pi * torch.rand(N)
r = 2.0
x_target = torch.stack([r * torch.cos(theta),
                         r * torch.sin(theta)], dim=1)

# -----------------------------
# 2. Neural velocity field
# dx/dt = v_theta(x)
# -----------------------------
class VelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

vnet = VelocityNet()

# -----------------------------
# 3. ODE solver (Euler)
# -----------------------------
T = 1.0
n_steps = 25
dt = T / n_steps

def flow(x):
    trajectory = [x]
    for _ in range(n_steps):
        x = x + dt * vnet(x)
        trajectory.append(x)
    return trajectory

# -----------------------------
# 4. Training
# -----------------------------
optimizer = optim.Adam(vnet.parameters(), lr=1e-3)

for epoch in range(800):
    traj = flow(x0)
    x_final = traj[-1]
    loss = ((x_final - x_target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

# -----------------------------
# 5. Generate trajectories
# -----------------------------
with torch.no_grad():
    traj = flow(x0)

traj_np = [x.numpy() for x in traj]

# -----------------------------
# 6. Visualization
# -----------------------------
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")

scat = ax.scatter(traj_np[0][:, 0], traj_np[0][:, 1], s=15)

def update(frame):
    scat.set_offsets(traj_np[frame])
    ax.set_title(f"Flow time step {frame}")
    return scat,

ani = FuncAnimation(fig, update, frames=len(traj_np), interval=200)
plt.show()

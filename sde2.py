from ode1 import BrownianMotion, EulerMaruyamaSimulator, SDE, ODE, plot_trajectories_1d, Simulator
import torch
import math
from tqdm import tqdm   
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.axes._axes import Axes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OUProcess(SDE):
    def __init__(self,theta:float, sigma:float):
        self.theta = theta
        self.sigma = sigma

    def drift_coefficent(self,xt:torch.Tensor, t:torch.Tensor)->torch.Tensor:
        """ Returns the drift coefficient at time t and position xt.
        Args:
            xt: state tensor of shape (batch_size, dim) at time t
            t: time tensor of shape () or 
        returns: 
            drift_coeff: Tensor of shape (batch_size, dim)"""
        return -self.theta * xt
    
    def diffusion_coefficent(self, xt:torch.Tensor, t:torch.Tensor)->torch.Tensor:
        """ Returns the diffusion coefficient at time t and position xt.
        Args:
            xt: state tensor of shape (batch_size, dim) at time t
            t: time tensor of shape () or 
        returns: 
            diffusion_coeff: Tensor of shape (batch_size, dim, dim)"""
        batch_size, dim = xt.shape
        diffusion_matrix = self.sigma * torch.eye(dim).to(xt.device)
        diffusion_coeff = diffusion_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        return diffusion_coeff

# thetas_and_sigmas = [(0.5, 0.1), (1.0, 0.3), (1.5, 0.5)]
# simulation_time = 20.0

# num_plot_plots = len(thetas_and_sigmas)
# fig, axes = plt.subplots(1,num_plot_plots, figsize=(8*num_plot_plots,7))

# for idx, (theta,sigma) in enumerate(thetas_and_sigmas):
#     ou_process = OUProcess(theta=theta, sigma=sigma)
#     simulator = EulerMaruyamaSimulator(sde=ou_process)
#     x0 = torch.linspace(-10.0,10.0,10).view(-1,1).to(device) # Initial values
#     ts = torch.linspace(0.0,simulation_time,1000).to(device) # simulation timesteps

#     ax = axes[idx]
#     ax.set_title(f'OU: theta={theta}, sigma={sigma}', fontsize=16)
#     ax.set_xlabel('Time', fontsize=14)
#     ax.set_ylabel('X(t)', fontsize=14)
#     plot_trajectories_1d(x0, simulator, ts, ax)
# plt.show()


def plot_scaled_trajectories_1d(x0: torch.Tensor, simulator: Simulator, timesteps: torch.Tensor, time_scale: float, label: str, ax: Optional[Axes] = None):
        """
        Graphs the trajectories of a one-dimensional SDE with given initial values (x0) and simulation timesteps (timesteps).
        Args:
            - x0: state at time t, shape (num_trajectories, 1)
            - simulator: Simulator object used to simulate
            - t: timesteps to simulate along, shape (num_timesteps,)
            - time_scale: scalar by which to scale time
            - label: self-explanatory
            - ax: pyplot Axes object to plot on
        """
        if ax is None:
            ax = plt.gca()
        trajectories = simulator.generate_trajectory(x0, timesteps) # (num_trajectories, num_timesteps, ...)
        for trajectory_idx in range(trajectories.shape[0]):
            trajectory = trajectories[trajectory_idx, :, 0] # (num_timesteps,)
            ax.plot(timesteps.cpu() * time_scale, trajectory.cpu(), label=label)

# Let's try rescaling with time
sigmas = [1.0, 2.0, 3.0]
ds = [0.25, 1.0,1.5] # sigma**2 / 2t
simulation_time = 10.0

fig, axes = plt.subplots(len(ds), len(sigmas), figsize=(8 * len(sigmas), 8 * len(ds)))
axes = axes.reshape((len(ds), len(sigmas)))
for d_idx, d in enumerate(ds):
    for s_idx, sigma in enumerate(sigmas):
        theta = sigma**2 / 2 / d
        ou_process = OUProcess(theta, sigma)
        simulator = EulerMaruyamaSimulator(sde=ou_process)
        x0 = torch.linspace(-20.0,20.0,20).view(-1,1).to(device)
        time_scale = sigma**2
        ts = torch.linspace(0.0,simulation_time / time_scale,1000).to(device) # simulation timesteps
        ax = axes[d_idx, s_idx]
        plot_scaled_trajectories_1d(x0=x0, simulator=simulator, timesteps=ts, time_scale=time_scale, label=f'Sigma = {sigma}', ax=ax)
        ax.set_title(rf'OU with $\sigma={sigma}$, $\theta={theta}$, D={d}')
        ax.set_xlabel(r't / $\sigma^2$')
        ax.set_ylabel(r'$X_t$')
plt.show()
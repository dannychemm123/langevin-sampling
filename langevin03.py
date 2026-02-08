from ode1 import BrownianMotion, EulerMaruyamaSimulator, SDE, ODE, plot_trajectories_1d, Simulator
from langevin02 import Density,imshow_density, Sampleable, GaussianMixture, Gaussian
import torch
import math
from tqdm import tqdm   
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.axes._axes import Axes
from abc import ABC, abstractmethod
import numpy as np
import torch.distributions as D
from torch.func import vmap, jacrev
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LangevinSDE(SDE):
    def __init__(self, sigma: float, density: Density):
        """
        Langevin SDE: dX_t = 0.5 * sigma^2 * score(X_t) dt + sigma dW_t
        where score(x) = grad_x log p(x)
        
        Args:
            sigma: noise scale
            density: instance of Density class providing log_density and score methods
        """
        self.sigma = sigma
        self.density = density

    def drift_coefficent(self, xt, t):
        """ Returns the drift coefficient at position xt and time t.
        Args:
            xt: Tensor of shape (batch_size, dim)
            t: float, time
        returns:
            drift: Tensor of shape (batch_size, dim)"""
        return 0.5 * self.sigma**2 * self.density.score(xt)
        
    def diffusion_coefficent(self, xt:torch.Tensor, t:torch.Tensor)-> torch.Tensor:
        """ Returns the diffusion coefficient at position xt and time t.
        Args:
            xt: Tensor of shape (batch_size, dim)
            t: float, time
        returns:
            diffusion: Tensor of shape (batch_size, dim)"""
        return self.sigma * torch.ones_like(xt)

# First, let's define two utility functions...
def every_nth_index(num_timesteps: int, n: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory
    """
    if n == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, n),
            torch.tensor([num_timesteps - 1]),
        ]
    )

def graph_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator, 
    density: Density,
    timesteps: torch.Tensor, 
    plot_every: int,
    bins: int,
    scale: float
):
    """
    Plot the evolution of samples from source under the simulation scheme given by simulator (itself a discretization of an ODE or SDE).
    Args:
        - num_samples: the number of samples to simulate
        - source_distribution: distribution from which we draw initial samples at t=0
        - simulator: the discertized simulation scheme used to simulate the dynamics
        - density: the target density
        - timesteps: the timesteps used by the simulator
        - plot_every: number of timesteps between consecutive plots
        - bins: number of bins for imshow
        - scale: scale for imshow
    """
    # Simulate
    x0 = source_distribution.sample(num_samples)
    xts = simulator.generate_trajectory(x0, timesteps)
    indices_to_plot = every_nth_index(len(timesteps), plot_every)
    plot_timesteps = timesteps[indices_to_plot]
    plot_xts = xts[:,indices_to_plot]

    # Graph
    fig, axes = plt.subplots(2, len(plot_timesteps), figsize=(8*len(plot_timesteps), 16))
    axes = axes.reshape((2,len(plot_timesteps)))
    for t_idx in range(len(plot_timesteps)):
        t = plot_timesteps[t_idx].item()
        xt = plot_xts[:,t_idx]
        # Scatter axes
        scatter_ax = axes[0, t_idx]
        imshow_density(density, bins, scale, scatter_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
        scatter_ax.scatter(xt[:,0].cpu(), xt[:,1].cpu(), marker='x', color='black', alpha=0.75, s=15)
        scatter_ax.set_title(f'Samples at t={t:.1f}', fontsize=15)
        scatter_ax.set_xticks([])
        scatter_ax.set_yticks([])

        # Kdeplot axes
        kdeplot_ax = axes[1, t_idx]
        imshow_density(density, bins, scale, kdeplot_ax, vmin=-15, alpha=0.5, cmap=plt.get_cmap('Blues'))
        sns.kdeplot(x=xt[:,0].cpu(), y=xt[:,1].cpu(), alpha=0.5, ax=kdeplot_ax,color='grey')
        kdeplot_ax.set_title(f'Density of Samples at t={t:.1f}', fontsize=15)
        kdeplot_ax.set_xticks([])
        kdeplot_ax.set_yticks([])
        kdeplot_ax.set_xlabel("")
        kdeplot_ax.set_ylabel("")

    plt.show()
if __name__ == "__main__":
    # Construct the simulator
    target = GaussianMixture.random_2D(n_components=5, std=0.75, scale=15.0, seed=3.0).to(device)
    sde = LangevinSDE(sigma = 0.6, density = target)
    simulator = EulerMaruyamaSimulator(sde)

    # Graph the results!
    graph_dynamics(
        num_samples = 1000,
        source_distribution = Gaussian(mean=torch.zeros(2), cov=20 * torch.eye(2)).to(device),
        simulator=simulator,
        density=target,
        timesteps=torch.linspace(0,5.0,1000).to(device),
        plot_every=334,
        bins=200,
        scale=15
    )
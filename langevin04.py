
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
from langevin03 import every_nth_index,LangevinSDE
from celluloid import Camera
from IPython.display import HTML

def animate_dynamics(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator, 
    density: Density,
    timesteps: torch.Tensor, 
    animate_every: int,
    bins: int,
    scale: float,
    save_path: str = 'dynamics_animation.mp4'
):
    """
    Plot the evolution of samples from source under the simulation scheme given by simulator (itself a discretization of an ODE or SDE).
    Args:
        - num_samples: the number of samples to simulate
        - source_distribution: distribution from which we draw initial samples at t=0
        - simulator: the discertized simulation scheme used to simulate the dynamics
        - density: the target density
        - timesteps: the timesteps used by the simulator
        - animate_every: number of timesteps between consecutive frames in the resulting animation
    """
    # Simulate
    x0 = source_distribution.sample(num_samples)
    xts = simulator.generate_trajectory(x0, timesteps)
    indices_to_animate = every_nth_index(len(timesteps), animate_every)
    animate_timesteps = timesteps[indices_to_animate]
    animate_xts = xts[:, indices_to_animate]

    # Graph
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    camera = Camera(fig)
    for t_idx in range(len(animate_timesteps)):
        t = animate_timesteps[t_idx].item()
        xt = animate_xts[:,t_idx]
        # Scatter axes
        scatter_ax = axes[0]
        imshow_density(density, bins, scale, scatter_ax, vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues'))
        scatter_ax.scatter(xt[:,0].cpu(), xt[:,1].cpu(), marker='x', color='black', alpha=0.75, s=15)
        scatter_ax.set_title(f'Samples')

        # Kdeplot axes
        kdeplot_ax = axes[1]
        imshow_density(density, bins, scale, kdeplot_ax, vmin=-15, alpha=0.5, cmap=plt.get_cmap('Blues'))
        sns.kdeplot(x=xt[:,0].cpu(), y=xt[:,1].cpu(), alpha=0.5, ax=kdeplot_ax,color='grey')
        kdeplot_ax.set_title(f'Density of Samples', fontsize=15)
        kdeplot_ax.set_xticks([])
        kdeplot_ax.set_yticks([])
        kdeplot_ax.set_xlabel("")
        kdeplot_ax.set_ylabel("")
        camera.snap()
    
    animation = camera.animate()
    animation.save(save_path)
    plt.close()
    return HTML(animation.to_html5_video())

if __name__ == "__main__":
    # Construct the simulator
    target = GaussianMixture.random_2D(n_components=5, std=0.75, scale=15.0, seed=3.0).to(device)
    sde = LangevinSDE(sigma = 0.6, density = target)
    simulator = EulerMaruyamaSimulator(sde)

    # Graph the results!
    animate_dynamics(
        num_samples = 1000,
        source_distribution = Gaussian(mean=torch.zeros(2), cov=20 * torch.eye(2)).to(device),
        simulator=simulator,
        density=target,
        timesteps=torch.linspace(0,5.0,1000).to(device),
        bins=200,
        scale=15,
        animate_every=100
    )   
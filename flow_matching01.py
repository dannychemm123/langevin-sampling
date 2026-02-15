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



class ConditionalProbabilityPath(torch.nn.Module,ABC):
    """Abstract class representing the conditional probability path p(x(t)|x(0)) of a stochastic process.
    """
    def __init__(self,p_sample:Sampleable,p_data: Sampleable):
        super().__init__()
        self.p_sample = p_sample
        self.p_data = p_data

    def sample_marginal_path(self,t: torch.Tensor)-> torch.Tensor:
        """Generates a sample from the marginal path p(x(t)) = p_t(x|z)p(z).
        Args:
            t: Tensor of shape (num_samples,1) representing the time at which to sample
        returns:
            x: Tensor of shape (batch_size, dim) representing the sample at time t"""
        num_samples = t.shape[0]
        z = self.sample_conditional_path(num_samples)
        x = self.sample_conditional_path(z,t)
        return x
    
    @abstractmethod
    def sample_conditional_path(self, num_samples: int) -> torch.Tensor:
        """Generates samples from the conditional path p(z|x(0)).
        Args:
            num_samples: number of samples to generate
        returns:
            z: Tensor of shape (num_samples, dim) representing the samples from the conditional path"""
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generates samples from the conditional path p(x(t)|z).
        Args:
            z: Tensor of shape (num_samples, dim) representing the samples from the conditional path
            t: Tensor of shape (num_samples,1) representing the time at which to sample
        returns:
            x: Tensor of shape (num_samples, dim) representing the samples at time t"""
        pass
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes the conditional vector field u_t(x|z).
        Args:
            x: Tensor of shape (num_samples, dim) representing the samples at time t
            z: Tensor of shape (num_samples, dim) representing the samples from the conditional path
            t: Tensor of shape (num_samples,1) representing the time at which to compute the vector field
        returns:
            u_t(x|z): Tensor of shape (num_samples, dim) representing the conditional vector field at time t"""
        pass
# Constants for the duration of our use of Gaussian conditional probability paths, to avoid polluting the namespace...
PARAMS = {
    "scale": 15.0,
    "target_scale": 10.0,
    "target_std": 1.0,
}
p_simple = Gaussian.isotropic(dim=2, std = 1.0).to(device)
p_data = GaussianMixture.symmetric_2D(n_components=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]).to(device)

fig, axes = plt.subplots(1,3, figsize=(24,8))
bins = 200

scale = PARAMS["scale"]
x_bounds = [-scale,scale]
y_bounds = [-scale,scale]

axes[0].set_title('Heatmap of p_simple')
axes[0].set_xticks([])
axes[0].set_yticks([])
imshow_density(density=p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=axes[0], vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))


axes[1].set_title('Heatmap of p_data')
axes[1].set_xticks([])
axes[1].set_yticks([])
imshow_density(density=p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=axes[1], vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

axes[2].set_title('Heatmap of p_simple and p_data')
axes[2].set_xticks([])
axes[2].set_yticks([])
imshow_density(density=p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
imshow_density(density=p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
plt.show()
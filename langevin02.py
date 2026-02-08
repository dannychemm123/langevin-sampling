from ode1 import BrownianMotion, EulerMaruyamaSimulator, SDE, ODE, plot_trajectories_1d, Simulator
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


class Density(ABC):
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the log density at position x.
        Args:
            x: Tensor of shape (batch_size, dim)
        returns:
            log_density: Tensor of shape (batch_size,)"""
        pass

    def score(self,x: torch.Tensor)-> torch.Tensor:
        """return the score dx log p(x) / dx at position x.
        Args:
            x: Tensor of shape (batch_size, dim)
            returns:
            score: Tensor of shape (batch_size, dim)"""
        
        x = x.unsqueeze(1)
        score = vmap(jacrev(self.log_density))(x)
        return score.squeeze((1,2,3))


class Sampleable(ABC):
    @abstractmethod
    def sample(self, n_samples:int)-> torch.Tensor:
        """Generates samples from the distribution.
        Args:
            n_samples: number of samples to generate
        returns:
            samples: Tensor of shape (n_samples, dim)"""
        pass

# Several ploting utility functions
def hist2d_sampleable(sampleable: Sampleable, n_samples:int, ax: Optional[Axes] = None, **kwargs):
    """
    Plots a 2D histogram of samples from a Sampleable distribution.
    Args:
        - sampleable: instance of Sampleable class
        - n_samples: number of samples to generate
        - ax: matplotlib Axes object (optional)
        
    """
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(n_samples)
    ax.hist2d(samples[:,0].cpu(), samples[:,1].cpu(), **kwargs)

def scatter_sampleable(sampleable: Sampleable, n_samples:int, ax: Optional[Axes] = None, **kwargs):
    """
    Plots a scatter plot of samples from a Sampleable distribution.
    Args:
        - sampleable: instance of Sampleable class
        - n_samples: number of samples to generate
        - ax: matplotlib Axes object (optional)"""
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(n_samples)
    ax.scatter(samples[:,0].cpu(), samples[:,1].cpu(), **kwargs)



def imshow_density(density: Density, bins:int, scale:float=4.0, ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    density_vals = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(density_vals.cpu(), extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)
    
def contour_density(density: Density, bins: int, scale: float, ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.contour(density.cpu(), extent=[-scale, scale, -scale, scale], origin='lower', **kwargs)

class Gaussian(torch.nn.Module, Sampleable, Density):
    """Two-dimensional Gaussian distribution with mean and covariance matrix.
    with Density and Sampleable capabilities.
    Args:
        mean: Tensor of shape (2,)
        cov: Tensor of shape (2, 2)
    """
    def __init__(self, mean,cov):
        """Initializes the Gaussian distribution.
        Args:
            mean: Tensor of shape (2,)
            cov: Tensor of shape (2, 2)
        """
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)


    @property
    def distribution(self):
        """Returns the underlying torch Distribution object."""
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)
    
    def sample(self, n_samples:int)-> torch.Tensor:
        """Generates samples from the Gaussian distribution.
        Args:
            n_samples: number of samples to generate
        returns:
            samples: Tensor of shape (n_samples, 2)"""
        return self.distribution.sample((n_samples,))
    
    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1,1)
    
class GaussianMixture(torch.nn.Module, Sampleable,Density):
    """Two dimensional Gaussian Mixture Model with Density and Sampleable capabilities.
    """
    def __init__(self, means: torch.Tensor, covs:torch.Tensor, weights: torch.Tensor):
        """Initializes the Gaussian Mixture Model.
        Args:
            means: Tensor of shape (n_components, 2)
            covs: Tensor of shape (n_components, 2, 2)
            weights: Tensor of shape (n_components,)
        """
        super().__init__()
        self.n_components = means.shape[0]
        self.register_buffer('means', means)
        self.register_buffer('covs', covs)
        self.register_buffer('weights', weights)

    @property
    def dim(self):
        return self.means.shape[1]
    
    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(loc=self.means, covariance_matrix=self.covs, validate_args=False),
            validate_args=False
        )
    def log_density(self, x: torch.Tensor)-> torch.Tensor:
        """Returns the log density at position x.
        Args:
            x: Tensor of shape (batch_size, 2)
        returns:
            log_density: Tensor of shape (batch_size,)"""
        return self.distribution.log_prob(x).view(-1,1)
    

    def sample(self, n_samples:int)-> torch.Tensor:
        """Generates samples from the Gaussian Mixture Model.
        Args:
            n_samples: number of samples to generate
        returns:
            samples: Tensor of shape (n_samples, 2)"""
        return self.distribution.sample(torch.Size((n_samples,)))
    
    @classmethod
    def random_2D(
        cls,
        n_components:int,
        std:float,scale:float = 10.0, seed = 42
    )-> 'GaussianMixture':
        """Generates a random 2D Gaussian Mixture Model.
        Args:
            n_components: number of components
            std: standard deviation of each component
            scale: scale of the means
            seed: random seed
        returns:
            GaussianMixture: instance of GaussianMixture class"""
        torch.manual_seed(seed)
        means = (torch.rand(n_components, 2) - 0.5) * scale
        covs = torch.diag_embed(torch.ones(n_components,2) * std**2)
        weights = torch.ones(n_components)
        return cls(means, covs, weights)
    
    @classmethod
    def symmetric_2D(cls, n_components:int, std:float, scale:float=10.0)-> "GaussianMixture":
        """ Generates a symmetric 2D Gaussian Mixture Model.
        Args:
            n_components: number of components (should be even)
            std: standard deviation of each component
            scale: scale of the means
        returns:
            GaussianMixture: instance of GaussianMixture class"""
        angles = torch.linspace(0, 2*np.pi, n_components+1)[:n_components]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(n_components,2) * std**2)
        weights = torch.ones(n_components) / n_components
        return cls(means, covs, weights)
    
if __name__ == "__main__":

    densities = {
        "Gaussian": Gaussian(mean=torch.zeros(2), cov=10 * torch.eye(2)).to(device),
        "Random Mixture": GaussianMixture.random_2D(n_components=5, std=1.0, scale=20.0, seed=3.0).to(device),
        "Symmetric Mixture": GaussianMixture.symmetric_2D(n_components=5, std=1.0, scale=8.0).to(device),
    }

    fig, axes = plt.subplots(1,3, figsize=(18, 6))
    bins = 100
    scale = 15
    for idx, (name, density) in enumerate(densities.items()):
        ax = axes[idx]
        ax.set_title(name)
        imshow_density(density, bins, scale, ax, vmin=-15, cmap=plt.get_cmap('Blues'))
        contour_density(density, bins, scale, ax, colors='grey', linestyles='solid', alpha=0.25, levels=20)
    plt.show()


            
        
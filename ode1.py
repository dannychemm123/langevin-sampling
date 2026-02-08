from abc import ABC, abstractmethod
from typing import Optional
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor):
        """Returns the drift coefficient at time t and position xt.
        Args:
            xt: Tensor of shape (batch_size, dim) at time t
            t: Tensor of shape (batch_size, 1) or (1,)
            
        returns: 
            drift_coeff: Tensor of shape (batch_size, dim)"""
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficent(self,xt: torch.Tensor, t: torch.Tensor):
        """Returns the drift coefficient at time t and position xt.
        Args:
            xt: Tensor of shape (batch_size, dim) at time t
            t: Tensor of shape (batch_size, 1) or (1,)
        returns: 
            drift_coeff: Tensor of shape (batch_size, dim)"""
        pass

    def diffusion_coefficent(self, xt: torch.Tensor, t: torch.Tensor):
        """Returns the diffusion coefficient at time t and position xt.
        Args:
            xt: Tensor of shape (batch_size, dim) at time t
            t: Tensor of shape (batch_size, 1) or (1,)
        returns: 
            diffusion_coeff: Tensor of shape (batch_size, dim, dim)"""
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """Performs a single time step of size dt at time t and position xt.
        Args:
            xt: Tensor of shape (batch_size, dim) at time t
            t: Tensor of shape (batch_size, 1) or (1,)
            dt: Tensor of shape (1,) or float
        returns: 
            xt_next: Tensor of shape (batch_size, dim) at time t + dt"""
        pass
    @torch.no_grad()
    def simulate(self,x:torch.Tensor, ts:torch.Tensor):
        """Simulates the SDE/ODE from time ts[0] to ts[-1] starting from x.
        Args:
            x: Tensor of shape (batch_size, dim) at time ts[0]
            ts: Tensor of shape (n_steps,) containing the time points
        returns: 
            trajectory: List of Tensors of shape (batch_size, dim) at each time point"""
        for t_idx in range(len(ts)-1):
            t = ts[t_idx]
            h = ts[t_idx+1] - ts[t_idx]
            x = self.step(x,t,h)
        return x
    @torch.no_grad()
    def generate_trajectory(self,x:torch.Tensor, ts:torch.Tensor):
        """Generates the full trajectory from time ts[0] to ts[-1] starting from x.
        Args:
            x: Tensor of shape (batch_size, dim) at time ts[0]
            ts: Tensor of shape (n_steps,) containing the time points
        returns:
            xs: trajectory: List of Tensors of shape (batch_size, dim) at each time point"""
        xs = [x.clone()]
        for t_ixd in tqdm(range(len(ts)-1)):
            t = ts[t_ixd]
            h = ts[t_ixd+1] - ts[t_ixd]
            x = self.step(x,t,h)
            xs.append(x.clone())
        return torch.stack(xs,dim=1)  # (batch_size, n_steps, dim)
    

class EulerSimulator(Simulator):
    def __init__(self,ode:ODE):
        self.ode = ode

    def step(self,xt:torch.Tensor, t:torch.Tensor,h:torch.Tensor):
        drift = self.ode.drift_coefficient(xt,t)
        xt_next = xt + drift * h
        return xt_next
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde:SDE):
        self.sde = sde

    def step(self,xt:torch.Tensor, t:torch.Tensor,h:torch.Tensor):
        # drift = self.sde.drift_coefficent(xt,t)
        # diffusion = self.sde.diffusion_coefficent(xt,t)
        # batch_size, dim = xt.shape
        # noise = torch.randn(batch_size, dim, device=xt.device)
        # xt_next = xt + drift * h + torch.matmul(diffusion, noise.unsqueeze(-1)).squeeze(-1) * math.sqrt(h)
        return xt + self.sde.drift_coefficent(xt,t) * h + self.sde.diffusion_coefficent(xt,t) * torch.sqrt(h) * torch.randn_like(xt)
        
    
class BrownianMotion(SDE):
    def __init__(self,sigma:float):
        self.sigma = sigma

    def drift_coefficent(self,xt: torch.Tensor, t: torch.Tensor)-> torch.Tensor:

        """ Returns the drift coefficient at time t and position xt.
        Args:
            xt: Tensor of shape (batch_size, dim) at time t
            t: Tensor of shape (batch_size, 1) or (1,)
        returns:
            drift_coeff: Tensor of shape (batch_size, dim)"""
        return torch.zeros_like(xt)
    def diffusion_coefficent(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ Returns the diffusion coefficient at time t and position xt.
        Args:
            xt: Tensor of shape (batch_size, dim) at time t
            t: Tensor of shape (batch_size, 1) or (1,)
        returns:
            diffusion_coeff: Tensor of shape (batch_size, dim, dim)"""
        batch_size, dim = xt.shape
        return self.sigma * torch.eye(dim, device=xt.device).unsqueeze(0).expand(batch_size, -1, -1)
    
def plot_trajectories_1d(x0: torch.Tensor, simulator: Simulator, timesteps: torch.Tensor, ax: Optional[Axes] = None):
        """
        Graphs the trajectories of a one-dimensional SDE with given initial values (x0) and simulation timesteps (timesteps).
        Args:
            - x0: state at time t, shape (num_trajectories, 1)
            - simulator: Simulator object used to simulate
            - t: timesteps to simulate along, shape (num_timesteps,)
            - ax: pyplot Axes object to plot on
        """
        if ax is None:
            ax = plt.gca()
        trajectories = simulator.generate_trajectory(x0, timesteps) # (num_trajectories, num_timesteps, ...)
        for trajectory_idx in range(trajectories.shape[0]):
            trajectory = trajectories[trajectory_idx, :, 0] # (num_timesteps,)
            ax.plot(timesteps.cpu(), trajectory.cpu(), alpha=0.5)


if __name__ == "__main__":
    sigma = 0.5
    brownian_motion = BrownianMotion(sigma)
    simulator = EulerMaruyamaSimulator(sde=brownian_motion)
    x0 = torch.zeros(5,1).to(device) # Initial values - let's start at zero
    ts = torch.linspace(0.0,5.0,500).to(device) # simulation timesteps

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_title(r'Trajectories of Brownian Motion with $\sigma=$' + str(sigma), fontsize=18)
    ax.set_xlabel(r'Time ($t$)', fontsize=18)
    ax.set_ylabel(r'$X_t$', fontsize=18)
    plot_trajectories_1d(x0, simulator, ts, ax)
    plt.show()
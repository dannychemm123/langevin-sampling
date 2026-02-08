a = 0.0
b = 1.0

def F_uniform(x,k_potential_walls):
    if x < a:
        return -k_potential_walls*(x - a)
    elif x > b:
        return -k_potential_walls*(x - b)
    else:
        return 0.0
    
def sample_langevin(n_steps, eps,k):
    import numpy as np
    x = 0.5
    for _ in range(n_steps):
        z = np.random.normal(0,1)
        x  += F_uniform(x,k) + np.sqrt(2*eps)*z
    return x


samples = [
    sample_langevin(n_steps = 30000,eps = 0.00001, k =3) for _ in range(10000)
]
import matplotlib.pyplot as plt
plt.hist(samples, bins=50, density=True)
plt.title("Langevin dynamics with reflecting walls")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()
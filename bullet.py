import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from DynamiX import explicit_euler

class System:
    def __init__(self, z0, vx0, vz0):
        """
        Defines the system consisting of one particle moving under the influence of uniform gravity.

        Parameters:
        - z0: initial height
        - vx0: initial forward velocity
        - vz0: initial upward velocity
        """
        self.t0 = 0
        self.q0 = np.array([0, 0, z0])
        self.v0 = np.array([vx0, 0, vz0])
        self.g = 9.81
        self.m = 1    # mass is irrelevant as it will cancel in the end. However, the solver wants a mass matrix, which is just a scalar here.

    def f(self, t, r, v):
        return np.array([0, 0, -self.m * self.g])

if __name__ == "__main__":

    system = System(1, 10, 10) 

    # Define the simulation parameters
    tf = 2               # Final time (s)
    h = 1e-2             # Time step (s)

    # Simulate using both methods
    t, r, v = explicit_euler(system, h, tf)

    # Plot the lengths over time on the same plot
    plt.figure(figsize=(10, 5))
    plt.plot(r[:, 0], r[:, 2], label='simulated_solution',linestyle='-')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title('Trajectory of the particle')
    plt.legend()
    plt.grid(True)

    plt.show()

    
    
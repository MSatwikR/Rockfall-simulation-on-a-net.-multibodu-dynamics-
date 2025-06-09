# =============================================================================
# DynamiX - Dynamical Particle Simulation 
# =============================================================================
#
# Lecture: Computational Multibody Dynamics
# Author: Dr.-Ing. Giuseppe Capobianco
# Institution: Institute for Applied Dynamics
#              Friedrich-Alexander-University Erlangen-Nuremberg
# Description: This file is part of DynamiX, a Python-based educational 
#              software for simulating particle systems designed as part of 
#              coursework for the Computational Multibody Dynamics lecture.
#
# Last Modified: 17. March 2025 by G. Capobianco
#
# =============================================================================
import numpy as np
from numpy.linalg import norm
from numba import jit, cuda
from timeit import default_timer as timer
from DynamiX import get_space_dimension

class Spring:
    
    def __init__(self, particle1, particle2, stiffness,eta, undeformed_length):
        """
        Initializes a spring interaction between two particles in a mechanical system.

        Parameters:
        - particle1: The first particle that the spring is connected to.
        - particle2: The second particle that the spring is connected to.
        - stiffness: The stiffness constant of the spring, defining its resistance to deformation
        - undeformed_length: The natural (undeformed) length of the spring.

        Attributes:
        - c: Stiffness of the spring.
        - l0: Undeformed length of the spring.
        - particle1, particle2: Particles between which the spring is established.
        - DOF: Degrees of freedom associated with the spring, derived from the connected particles.
        - nr: Number of spatial dimensions, fetched from global configuration.
        """
        self.c = stiffness
        self.l0 = undeformed_length
        self.particle1 = particle1  # Reference to the first particle (P1)
        self.particle2 = particle2  # Reference to the second particle (P2)
        self.eta = eta

        # Verify that both particles have defined degrees of freedom (i.e., are part of a system)
        for p in (particle1, particle2):
            if p.DOF is None:
                raise LookupError(
                    'The particle needs to be added to the system before it can interact.'
                )

        # Concatenate degrees of freedom for both particles into a single array for the spring
        self.DOF = np.concatenate([particle1.DOF, particle2.DOF])

        # Number of coordinates of a particle (equals space dimension)
        self.nr = get_space_dimension()


    def l(self, q):
        """
        Computes the current length of the spring based on the positions of its connected particles.

        Parameters:
        - q: Array of global positions for all particles in the system.

        Returns:
        - The current length of the spring.
        """
        return norm(self.particle2.slice(q) - self.particle1.slice(q))


    def f(self, t, q, v):
        """
        Calculates the force exerted by the spring on the connected particles.

        Parameters:
        - t: Current time (unused in this method but kept for consistency with other forces that may depend on time).
        - q: Array of global positions for all particles in the system.
        - v: Array of global velocities for all particles (unused in this method but included for general compatibility).

        Returns:
        - f: A generalized force vector applied to both particles, with the first half pertaining to particle1
             and the second half to particle2.
        """
        f = np.zeros(2 * self.nr)  # Initialize force vector for both particles (F_P1, F_P2)

        # Calculate the vector from particle1 to particle2
        r12 = self.particle2.slice(q) - self.particle1.slice(q)

        # Calculate the current length of the spring
        l = norm(r12)
        #print('l',l)

        # Determine the force direction (normalized vector from P1 to P2)
        n = r12 / l
        #print('n', n)

        # Calculate the scalar force magnitude using Hooke's law
        la = -self.c * (l - self.l0) - self.eta * (np.dot((self.particle2.slice(v)-self.particle1.slice(v)),n))
        #print('la',la)

        # Apply the force vector to both particles:
        # - Particle 1 feels a force in the opposite direction of n (compression/tension)
        # - Particle 2 feels an equal magnitude force in the direction of n
        f[:self.nr] = -la * n
        f[self.nr:] = la * n

        return f
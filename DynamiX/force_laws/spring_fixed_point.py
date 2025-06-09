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

from DynamiX import get_space_dimension


class SpringFixedPoint:

    def __init__(self, fixed_point, particle, stiffness,eta, undeformed_length):
        """
        Initializes a spring interaction between a fixed point in space and a particle.

        Parameters:
        - fixed_point: A numpy array representing the fixed point in space where the spring is anchored.
        - particle: The particle that the spring is connected to.
        - stiffness: The stiffness constant of the spring, defining its resistance to deformation.
        - undeformed_length: The natural (undeformed) length of the spring.

        Attributes:
        - c: Stiffness of the spring.
        - l0: Undeformed length of the spring.
        - fixed_point: The fixed anchor point of the spring in space.
        - particle: The particle connected to the other end of the spring.
        - DOF: Degrees of freedom associated with the spring, derived from the connected particle.
        - nr: Number of spatial dimensions, fetched from global configuration.
        """
        self.c = stiffness
        self.l0 = undeformed_length
        self.fixed_point = np.array(fixed_point)  # Ensure fixed point is a numpy array
        self.particle = particle  # Reference to the connected particle
        self.eta = eta

        # Verify that the particle has defined degrees of freedom (i.e., is part of a system)
        if particle.DOF is None:
            raise LookupError(
                'The particle needs to be added to the system before it can interact with a spring.'
            )

        # Degrees of freedom for the particle (since the fixed point does not have DOF)
        self.DOF = particle.DOF

        # Number of coordinates per particle (spatial dimensions)
        self.nr = get_space_dimension()


    def l(self, q):
        """
        Computes the current length of the spring based on the position of the connected particle.

        Parameters:
        - r: Array of global positions for all particles in the system.

        Returns:
        - The current length of the spring.
        """
        # Calculate the length as the distance between the fixed point and the particle
        return norm(self.fixed_point - self.particle.slice(q))


    def f(self, t, q, v):
        """
        Calculates the force exerted by the spring on the connected particle.

        Parameters:
        - t: Current time (unused in this method but kept for consistency with other forces that may depend on time).
        - r: Array of global positions for all particles in the system.
        - v: Array of global velocities for all particles (unused in this method but included for general compatibility).

        Returns:
        - F: A force vector applied to the particle, accounting for spring tension/compression.
        """
        F = np.zeros(self.nr)  # Initialize force vector for the particle

        # Calculate the vector from the fixed point to the particle
        r_fp = self.particle.slice(q) - self.fixed_point

        # Calculate the current length of the spring
        l = norm(r_fp)

        # Determine the force direction (normalized vector from the fixed point to the particle)
        n = r_fp / l

        # Calculate the scalar force magnitude using Hooke's law
        la = -self.c * (l - self.l0) - self.eta * (np.dot(self.particle.slice(v),n))

        # Apply the force vector to the particle, directed along n
        F = la * n

        return F
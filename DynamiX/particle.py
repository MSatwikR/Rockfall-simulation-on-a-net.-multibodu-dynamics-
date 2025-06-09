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

from DynamiX import get_space_dimension

class Particle:
    def __init__(self, mass, r0=None, v0=None):
        """
        Initializes a Particle object representing a point mass in space.

        Parameters:
        - mass: The mass of the particle, a scalar value.
        - r0: Initial position vector of the particle in the simulation space. 
              Defaults to a zero vector if not provided.
        - v0: Initial velocity vector of the particle in the simulation space.
              Defaults to a zero vector if not provided.

        Attributes:
        - mass: Mass of the particle, affecting its dynamics under applied forces.
        - r0: Initial position of the particle. Initialized to a zero vector with a length
              equal to the space dimension if not explicitly specified.
        - v0: Initial velocity of the particle. Initialized similarly to r0 for consistency.
        - DOF: Degrees of freedom for the particle, which is populated when the particle 
               is added to a system. Represents indices in global position and velocity vectors.
        """
        self.mass = mass
        dim = get_space_dimension()
        self.r0 = r0 if r0 is not None else np.zeros(dim)
        self.v0 = v0 if v0 is not None else np.zeros(dim)
        self.DOF = None  # Will be set when the particle is integrated into a simulation system



    def slice(self, x):
        """
        Extracts the relevant position or velocity subarray for this particle 
        from a larger system-wide array using its degrees of freedom.

        Parameters:
        - x: A global array (e.g., positions or velocities of all particles in the system).

        Returns:
        - A subarray of the passed array, specifically corresponding to the degrees of freedom
          of this particle, allowing for independent manipulation or examination.
        """
        return x[self.DOF]


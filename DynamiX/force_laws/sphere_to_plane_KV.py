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
# Last Modified: 26. March 2025 by G. Capobianco
#
# =============================================================================
import numpy as np
from numpy.linalg import norm

from DynamiX import get_space_dimension

class SphereToPlaneKV:
    def __init__(self, particle, plane_normal, plane_point, radius, stiffness, damping):
        """
        Initializes a Kelvin--Voigt type contact between a spherical particle and a plane in a mechanical system.

        Parameters:
        - particle: The particle.
        - plane_normal: The normal vector of the plane.
        - plane_point: The point on the plane defining the location of the plane.
        - radius: The radius of the particle.
        - stiffness: The stiffness constant of the contact law.
        - damping: The damping constant of the contact law.

        Attributes:
        - particle1: Particle that interacts with plane.
        - n: Normal vector of the plane.
        - r_P: Point defining the location of the plane.
        - R: Radius of the particle.
        - c: Stiffness.
        - eta: Damping constant.
        - DOF: Degrees of freedom associated with the contact interaction.
        - nr: Number of spatial dimensions, fetched from global configuration.
        """
        self.n = plane_normal / norm(plane_normal)
        self.r_P = plane_point
        self.R = radius
        self.c = stiffness
        self.eta = damping
        self.particle = particle  # Reference to the particle (P)

        # Verify that the particle has defined degrees of freedom (i.e., is part of a system)
        if particle.DOF is None:
            raise LookupError(
                'The particle needs to be added to the system before it can interact.'
            )

        # save the connectivity
        self.DOF = particle.DOF

        # Number of coordinates of the particle (equals space dimension)
        self.nr = get_space_dimension()

    def g(self, q):
        """
        Computes the current gap between the particle and the plane.

        Parameters:
        - q: Array of global positions for all particles in the system.

        Returns:
        - The current gap.
        """
        r = self.particle.slice(q)

        return np.dot(self.n, r - self.r_P) - self.R
    
    def g_dot(self, q, v):
        """
        Computes the current gap velocity between the particle and the plane.

        Parameters:
        - q: Array of global positions for all particles in the system.
        - v: Array of global velocities for all particles in the system.

        Returns:
        - The current gap velocity.
        """
        v = self.particle.slice(v)
        return np.dot(self.n, v)

    def f(self, t, q, v):
        """
        Calculates the force exerted by the contact interaction on the partilce.

        Parameters:
        - t: Current time (unused in this method but kept for consistency with other forces that may depend on time).
        - q: Array of global positions for all particles in the system.
        - v: Array of global velocities for all particles (unused in this method but included for general compatibility).

        Returns:
        - f: A generalized force vector applied to the particle.
        """
     
        g = self.g(q)

        if g > 0:
            return np.zeros(self.nr)
        
        g_dot = self.g_dot(q, v)


        # Calculate the scalar force magnitude using the Kelvin-Voigt force law
        la = -self.c * g - self.eta * g_dot

        return la * self.n
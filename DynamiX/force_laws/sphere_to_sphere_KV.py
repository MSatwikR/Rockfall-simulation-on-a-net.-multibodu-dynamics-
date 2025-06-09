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

class SphereToSphereKV:
    def __init__(self, particle0, particle1, radius0,radius1, stiffness, damping):
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
        #self.n = plane_normal / norm(plane_normal)
        #self.r_P = plane_point
        self.R0 = radius0
        self.R1 = radius1
        self.c = stiffness
        self.eta = damping
        self.particle0 = particle0 # Reference to the particle (P)
        self.particle1 = particle1

        # Verify that the particle has defined degrees of freedom (i.e., is part of a system)
        if particle0.DOF is None:
            raise LookupError(
                'The particle needs to be added to the system before it can interact.'
            )

        '''if particle1.DOF is None:
            raise LookupError(
                'The particle needs to be added to the system before it can interact.'
            )'''
        # save the connectivity
        self.DOF = np.concatenate((particle0.DOF, particle1.DOF))

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
        r = self.particle0.slice(q) - self.particle1.slice(q)

        return np.linalg.norm(r) - self.R0 -self.R1
    
    def g_dot(self, q, v):
        """
        Computes the current gap velocity between the particle and the plane.

        Parameters:
        - q: Array of global positions for all particles in the system.
        - v: Array of global velocities for all particles in the system.

        Returns:
        - The current gap velocity.
        """
        q0 = self.particle0.slice(q)
        q1 = self.particle1.slice(q)
        v0 = self.particle0.slice(v)
        v1 = self.particle1.slice(v)

        self.n = (q0 - q1) / np.linalg.norm(q0-q1)
        return np.dot(self.n, v0-v1)

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
            return np.zeros(self.nr * 2)  
        
        g_dot = self.g_dot(q, v)


        # Calculate the scalar force magnitude using the Kelvin-Voigt force law
        la = -self.c * g - self.eta * g_dot
        f = np.zeros(2 * self.nr)
        f[0:self.nr] = la * self.n
        f[self.nr:2*self.nr] = -la * self.n
        

        return f
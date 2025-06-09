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

class System:
    def __init__(self, t0=0, gravity=None):
        """
        Initializes the System class for simulating multibody dynamics.

        Parameters:
        - t0: Initial time of the simulation. Default is 0.
        - gravity: Gravitational acceleration vector, defaulting to a zero vector 
          based on the space dimension if not provided.

        Attributes:
        - particles: List to store the particle objects in the system.
        - interactions: List to store interaction objects (e.g., springs) in the system.
        - q0: Initial positions of all particles.
        - v0: Initial velocities of all particles.
        - f0: Initial force acting on all particles due to gravity.
        - m: Mass of all particles; used for constructing the mass matrix.
        - nr: Number of spatial dimensions, fetched dynamically from the global setting.
        - gravity: Gravitational vector affecting all particles.
        - last_particle_index: Tracks the index for the next particle to be added.
        - nDOF: Total number of degrees of freedom in the system.
        """
        self.particles = []
        self.interactions = []
        self.t0 = t0
        self.q0 = []
        self.v0 = []
        self.f0 = []
        self.m = []
        self.space_dim = get_space_dimension()
        self.gravity = gravity if gravity is not None else np.zeros(self.space_dim)
        self.last_particle_index = 0
        self.nDOF = 0  # number of degrees of freedom

    def add_particles(self, *particles):
        """
        Adds particle(s) to the system.

        Updates each particle's degrees of freedom (DOF) indices and increments the system's total 
        degrees of freedom. Also updates the initial position, velocity, mass, and force lists.

        Parameters:
        - particles: Any number of Particle objects to be added to the system.
        """
        for p in particles:
            p.DOF = np.arange(self.space_dim) + self.last_particle_index
            self.last_particle_index += self.space_dim
            self.nDOF += self.space_dim
            self.particles.append(p)
            self.q0.extend(p.r0)
            self.v0.extend(p.v0)
            self.m.extend(p.mass * np.ones(self.space_dim))
            self.f0.extend(p.mass * self.gravity)

    def add_interactions(self, *interactions):
        """
        Adds interaction(s) to the system, such as springs between particles.

        Parameters:
        - interactions: Any number of interaction objects to be included in the system.
        """
        # self.interactions.extend(interactions)
        for i in interactions:
            self.interactions.append(i)

    def assemble(self):
        """
        Converts initial position, velocity, mass, and force lists into arrays.

        Prepares the system for simulation by assembling necessary data structure from 
        lists into numpy arrays.
        """
        self.q0 = np.array(self.q0)
        self.v0 = np.array(self.v0)
        self.m = np.array(self.m)
        self.f0 = np.array(self.f0)

        print('system summary:')
        print(' ')
        print(f'  -  physical space dimensions = {get_space_dimension()}')
        print(f'  -  number of particles = {len(self.particles)}')
        print(f'  -  number of degrees of freedom = {self.nDOF}')
        print(f'  -  number of interactions = {len(self.interactions)}')
        print(' ')
        print(50 * '-')

    def f(self, t, q, v):
        """
        Computes the net force on the system at a given time and state.

        Parameters:
        - t: Current simulation time.
        - r: Current particle positions.
        - v: Current particle velocities.

        Returns:
        - F: Global force vector resulting from all interactions in the system.
        """
        f = self.f0.copy()  # Start with gravity-induced forces
        for i in self.interactions:
            f[i.DOF] += i.f(t, q, v)  # Add interaction forces
        return f
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
from tqdm import tqdm

def stoermer_verlet(system, h, tf):
    """
    Performs numerical integration using the Störmer-Verlet method for a dynamical system.
    Note, the scheme has been slightly modified, such that it is explict. 
    The change only affects systems with velocity dependent forces.

    Parameters:
    - system: An instance of the System class containing the initial state and dynamics.
    - h: Time step size for the simulation.
    - tf: Final time up to which the system is simulated.

    Returns:
    - t: Array of time instants at which the system state is computed.
    - q: Array of position vectors at each time instant.
    - v: Array of velocity vectors at each time instant.
    """
    # Extract initial conditions from the system
    t0 = system.t0  # Initial time
    q0 = system.q0  # Initial positions
    v0 = system.v0  # Initial velocities

    # Calculate the number of time steps based on the total simulation time and step size
    N = int((tf - t0) / h)  # Number of time steps
    nt = N + 1              # Total number of time nodes, including the initial state

    # Print solver info
    print('Solver - Störmer-Verlet method:')
    print(' ')
    print(f'  -  simulation time interval = {[t0, tf]}')
    print(f'  -  time step = {h}')
    print(f'  -  number of time steps = {N}')
    print(' ')
    print(50 * '-')

    # Determine the number of positions and velocities (degrees of freedom)
    nq = len(q0)            # Number of position coordinates
    nv = len(v0)            # Number of velocity coordinates

    # Initialize arrays to record time instants, positions, and velocities
    t = np.zeros(nt)        # Array to store time instants
    q = np.zeros((nt, nq))  # Array to store positions at each time instant
    v = np.zeros((nt, nv))  # Array to store velocities at each time instant

    # Set initial conditions in the respective arrays
    t[0] = t0
    q[0] = q0
    v[0] = v0

    # Calculate the inverse of the mass matrix for efficiency in velocity updates
    M_inv = 1 / system.m

    # Begin the main simulation loop using the explicit Euler integration method
    for k in tqdm(range(N)):
        # Update the current time by the time step
        t[k + 1] = t[k] + h

        # first half-step (TODO: modified to be explicit!)
        v_k_half = v[k] + (h / 2) * M_inv * system.f(t[k], q[k], v[k])
        
        # Update the position using the mid-point velocity
        q[k + 1] = q[k] + h * v_k_half
        
        # second half-step for the velocity
        v[k + 1] = v_k_half + (h / 2) * M_inv * system.f(t[k+1], q[k+1], v_k_half)

    # Return the arrays of time, positions, and velocities
    return t, q, v
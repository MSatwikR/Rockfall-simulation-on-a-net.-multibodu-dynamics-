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
print(50 * '=')
print(r'    _____                              ___   __')
print(r'   |  __ \                            (_) \ / /')
print(r'   | |  | |_   _ _ __   __ _ _ __ ___  _ \ V / ')
print(r"   | |  | | | | | '_ \ / _` | '_ ` _ \| | > <  ")
print(r'   | |__| | |_| | | | | (_| | | | | | | |/ . \ ')
print(r"   |_____/ \__, |_| |_|\__,_|_| |_| |_|_/_/ \_\ ")
print(r'            __/ |                              ')
print(r'           |___/                               ')
print(' ')
print(50 * '=')
print(' ')

SPACE_DIM = 3

def set_space_dimension(dim):
    global SPACE_DIM
    SPACE_DIM = dim

def get_space_dimension():
    return SPACE_DIM
from .system import System
from .particle import Particle
from .solver.explicit_euler import explicit_euler
from .solver.semi_implicit_euler import semi_implicit_euler
from .solver.stoermer_verlet import stoermer_verlet
from .force_laws.spring import Spring
from .force_laws.spring_fixed_point import SpringFixedPoint
from .force_laws.sphere_to_plane_KV import SphereToPlaneKV
from .force_laws.Newtonian_gravity import NewtonianGravity
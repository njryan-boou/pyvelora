"""Module for constants used in pyvelora."""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------

pi = math.pi
tau = 2 * pi
e = math.e
golden_ratio = (1 + math.sqrt(5)) / 2
euler_mascheroni = 0.5772156649015328606  # Euler–Mascheroni constant γ
sqrt2 = math.sqrt(2)
sqrt3 = math.sqrt(3)
ln2 = math.log(2)
ln10 = math.log(10)
inf = float("inf")
nan = float("nan")

# ---------------------------------------------------------------------------
# Universal physical constants  (SI units)
# ---------------------------------------------------------------------------

# Speed of light in vacuum (m/s)
c = 299_792_458.0

# Planck's constant (J·s)
h = 6.62607015e-34

# Reduced Planck's constant (J·s)
hbar = h / (2 * pi)

# Gravitational constant (m³ kg⁻¹ s⁻²)
G = 6.67430e-11

# Elementary charge (C)
q_e = 1.602176634e-19

# Boltzmann constant (J/K)
k_B = 1.380649e-23

# Avogadro's number (mol⁻¹)
N_A = 6.02214076e23

# Gas constant (J mol⁻¹ K⁻¹)
R = N_A * k_B

# Electron mass (kg)
m_e = 9.1093837015e-31

# Proton mass (kg)
m_p = 1.67262192369e-27

# Neutron mass (kg)
m_n = 1.67492749804e-27

# Vacuum permittivity (F/m)
epsilon_0 = 8.8541878128e-12

# Vacuum permeability (H/m)
mu_0 = 1.25663706212e-6

# Stefan–Boltzmann constant (W m⁻² K⁻⁴)
sigma = 5.670374419e-8

# Fine-structure constant (dimensionless)
alpha = 7.2973525693e-3

# Bohr radius (m)
a_0 = 5.29177210903e-11

# Rydberg energy (J)
Ry = 2.1798723611035e-18

# Standard acceleration of gravity (m/s²)
g = 9.80665

# Atmospheric pressure (Pa)
atm = 101_325.0

# Electron volt (J)
eV = q_e





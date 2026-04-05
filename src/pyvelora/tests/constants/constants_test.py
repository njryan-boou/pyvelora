"""Tests for pyvelora/constants/constants.py."""
import math
import numpy as np
import pytest
from pyvelora.constants.constants import (
    pi, tau, e, golden_ratio, euler_mascheroni,
    sqrt2, sqrt3, ln2, ln10,
    inf, nan,
    c, h, hbar, G, q_e, k_B, N_A, R,
    m_e, m_p, m_n,
    epsilon_0, mu_0, sigma, alpha,
    a_0, Ry, g, atm, eV,
)


# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------

def test_pi_value():
    assert pi == np.pi


def test_e_value():
    assert e == np.e


def test_golden_ratio_value():
    assert np.isclose(golden_ratio, (1 + np.sqrt(5)) / 2)


def test_golden_ratio_approx():
    assert np.isclose(golden_ratio, 1.6180339887, rtol=1e-9)


def test_pi_is_float():
    assert isinstance(pi, float)


def test_e_is_float():
    assert isinstance(e, float)


def test_golden_ratio_is_float():
    assert isinstance(golden_ratio, float)


def test_euler_mascheroni_approx():
    assert np.isclose(euler_mascheroni, 0.5772156649, rtol=1e-9)


def test_sqrt2_value():
    assert np.isclose(sqrt2, math.sqrt(2))


def test_sqrt3_value():
    assert np.isclose(sqrt3, math.sqrt(3))


def test_ln2_value():
    assert np.isclose(ln2, math.log(2))


def test_ln10_value():
    assert np.isclose(ln10, math.log(10))


def test_tau_value():
    assert np.isclose(tau, 2 * np.pi)


def test_inf_sentinel():
    assert math.isinf(inf)


def test_nan_sentinel():
    assert math.isnan(nan)


# ---------------------------------------------------------------------------
# Physical constants — type checks and known values
# ---------------------------------------------------------------------------

def test_speed_of_light_exact():
    assert c == 299_792_458.0


def test_planck_constant_approx():
    assert np.isclose(h, 6.62607015e-34, rtol=1e-8)


def test_hbar_relation():
    assert np.isclose(hbar, h / (2 * pi), rtol=1e-10)


def test_gravitational_constant_approx():
    assert np.isclose(G, 6.674e-11, rtol=1e-3)


def test_elementary_charge_approx():
    assert np.isclose(q_e, 1.602176634e-19, rtol=1e-8)


def test_boltzmann_constant_approx():
    assert np.isclose(k_B, 1.380649e-23, rtol=1e-8)


def test_avogadro_approx():
    assert np.isclose(N_A, 6.02214076e23, rtol=1e-8)


def test_gas_constant_relation():
    assert np.isclose(R, N_A * k_B, rtol=1e-10)


def test_electron_mass_approx():
    assert np.isclose(m_e, 9.1093837015e-31, rtol=1e-8)


def test_proton_mass_approx():
    assert np.isclose(m_p, 1.67262192369e-27, rtol=1e-8)


def test_neutron_mass_approx():
    assert np.isclose(m_n, 1.67492749804e-27, rtol=1e-8)


def test_vacuum_permittivity_approx():
    assert np.isclose(epsilon_0, 8.8541878128e-12, rtol=1e-8)


def test_vacuum_permeability_approx():
    assert np.isclose(mu_0, 1.25663706212e-6, rtol=1e-8)


def test_stefan_boltzmann_approx():
    assert np.isclose(sigma, 5.670374419e-8, rtol=1e-8)


def test_fine_structure_constant_approx():
    assert np.isclose(alpha, 7.2973525693e-3, rtol=1e-8)


def test_bohr_radius_approx():
    assert np.isclose(a_0, 5.29177210903e-11, rtol=1e-8)


def test_standard_gravity():
    assert g == 9.80665


def test_standard_atmosphere():
    assert atm == 101_325.0


def test_ev_equals_elementary_charge():
    assert eV == q_e


# ---------------------------------------------------------------------------
# Import via constants package namespace
# ---------------------------------------------------------------------------

def test_import_via_package():
    from pyvelora.constants import c as speed_of_light, k_B as boltzmann
    assert speed_of_light == 299_792_458.0
    assert np.isclose(boltzmann, 1.380649e-23)

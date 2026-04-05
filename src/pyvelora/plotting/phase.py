from pyvelora.plotting.vector_field import vector_field
from pyvelora.plotting.trajectory import trajectory


def phase_portrait(F, sol=None):
    """
    Plot vector field + optional trajectory
    """
    vector_field(F)

    if sol is not None:
        trajectory(sol)
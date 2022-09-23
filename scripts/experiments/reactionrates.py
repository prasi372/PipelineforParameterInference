
"""
reactionrates.py

Used to compute the well-mixed exit and entry rates for the compartment-based model.

"""

import numpy as np
import math
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


def entry_exit_rates(r_nuc = 2.5, r_cell = 6.0, D = 1.0):
    """
    Computes the exit and entry rates out from and in to 
    the nucleus.
    """
    k_entry = D * (r_cell**3 - r_nuc**3) / (r_nuc**2*r_cell**3/3 - 3*r_cell**5/5 + r_cell**6/(3*r_nuc) - r_nuc**5/15)
    k_exit = D * r_nuc**2 / 6

    return (k_entry, k_exit)


def well_mixed_rates(ka, kd, sigma, D, Vol):
    """
    Computes the well-mixed rates for a bimolecular reaction.
    Input is microscopic reaction rates, sum of the reaction radii
    sigma, and sum of diffusion constants D. Vol = volume of reactive
    domain.
    """

    h = pow(Vol, 1.0/3.0)
    G = 1.5164/(6.0*h)-1.0/(4*math.pi*sigma)
    ka_meso = ka/Vol*pow(1-ka/D*G, -1)
    kd_meso = Vol*kd*ka_meso/ka
    return (ka_meso, kd_meso)

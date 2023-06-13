import numpy as np
from unyt import mp
from unyt.constants import G, k_B, critical_density


def cal_T200c(M200c, rho_c):
    mu = 0.59
    return mu * mp/ (3 * k_B) * G * np.power(M200c, 2/3) * np.power(200 * critical_density, 1/3)

def cal_solarabun():
    
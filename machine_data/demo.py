# -*- coding: utf-8 -*-
"""
Small ring with h=20 to run tests.
"""

import numpy as np
from mbtrack2.tracking import Synchrotron, Electron
from mbtrack2.utilities import Optics 

def demo():
    
    h = 20
    L = 100
    E0 = 1e9
    particle = Electron()
    ac = 1e-3
    U0 = 250e3
    tau = np.array([10e-3, 10e-3, 5e-3])
    tune = np.array([18.2, 10.3])
    emit = np.array([50e-9, 50e-9*0.01])
    sigma_0 = 30e-12
    sigma_delta = 1e-3
    chro = [1.0,1.0]
    
    # mean values
    beta = np.array([1, 1])
    alpha = np.array([0, 0])
    dispersion = np.array([0, 0, 0, 0])
    optics = Optics(local_beta=beta, local_alpha=alpha, 
                      local_dispersion=dispersion)
    
    ring = Synchrotron(h, optics, particle, L=L, E0=E0, ac=ac, U0=U0, tau=tau,
                       emit=emit, tune=tune, sigma_delta=sigma_delta, 
                       sigma_0=sigma_0, chro=chro)
    
    return ring
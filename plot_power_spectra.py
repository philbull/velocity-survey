#!/usr/bin/env python
"""
Plot how each cosmological/nuisance parameter affects the power spectra.
"""
import numpy as np
import pylab as P
from fisher_velocity import signal_power, build_ccl_cosmo, noise_covariance

# Define fiducial parameters
params0 = {        
    # Cosmological parameters
    'Omega_c':  0.2647,
    'Omega_b':  0.0492,
    'h':        0.673,
    '10^9A_s':  2.207,
    'n_s':      0.9645,
    'w0':       -1.,
    'wa':       0.,
    
    # Astrophysical/nuisance parameters
    'b0':           1.,       # bias amplitude
    'r_g':          0.98,     # gal-vel correlation coefficient
    'sigma_g':      5.8/0.67, # Mpc, z-space smoothing scale (galaxies)
    'sigma_u':      13./0.67, # Mpc, z-space smoothing scale (velocities)
    'sigma_disp':   300.,     # km/s, non-linear velocity dispersion
    'vel_err':      0.2,      # fractional error on velocities
    
    # Rescaling of cosmological functions (f -> x*f)
    'x_f':          1.,       # Scaling of growth rate, f(z)
    'x_H':          1.,       # Scaling of Hubble rate, H(z)
}

z = 0.05
k = np.logspace(-4., 0., 1500)
cosmo, params = build_ccl_cosmo(params0)

P.subplot(111)
cols = ['r', 'y', 'g', 'c', 'b']
for i, _mu in enumerate([-1., -0.5, 0., 0.5, 1.]):
    
    # Calculate power spectra
    pk_gg, pk_gv, pk_vv = signal_power(z, k, _mu, cosmo, params)
    
    P.plot(k, pk_gg, lw=1.8, color=cols[i], label="$\mu = %1.1f$" % _mu)
    P.plot(k, pk_gv.imag, lw=1.8, color=cols[i], dashes=[2,2])
    P.plot(k, pk_vv, lw=1.8, color=cols[i], dashes=[4,4])

# Noise curves
N = noise_covariance(z, params0)
P.axhline(N[0,0], lw=1.8, color='k', alpha=0.4)
P.axhline(N[1,1], lw=1.8, color='k', alpha=0.4, dashes=[4,4])

P.xscale('log')
P.yscale('log')
P.xlabel(r"$k$ $[{\rm Mpc}^{-1}]$", fontsize=18)
P.ylabel(r"$P(k)$", fontsize=18)

P.legend(loc='lower left')
P.tight_layout()
P.show()

#!/usr/bin/env python
"""
Fisher matrix calculation for 21cm line profile detection.
"""
import numpy as np
import pylab as P

sigma0 = 10. # uJy (noise rms per channel)


def profile(nu, params):
    nu_c = params['nu_c']
    w = params['w']
    return np.exp(-0.5*(nu - nu_c)**2. / w**2.) / np.sqrt(2.*np.pi) / w


def fisher(nu, params):
    """
    Calculate Fisher matrix for detectability of 21cm line.
    """
    # Fetch fiducial parameters
    S_HI = params['S_HI']
    nu_c = params['nu_c']
    w = params['w']
    x_fill = params['x_fill']
    
    # 21cm line profile
    f_HI = profile(nu, params)
    
    # Continuum profile
    f_cont = 1.
    
    # Noise per frequency channel
    # FIXME
    sigma_nu = sigma0 * np.ones(nu.size)
    
    # Fisher derivatives
    y = x_fill * f_HI
    ds_dSHI = y
    ds_dnuc = (nu - nu_c) * y / w**2.
    ds_dw = -(w + nu_c - nu) * (w + nu - nu_c) * y / w**3.
    
    pnames = ['S_HI', 'nu_c', 'w']
    derivs = [ds_dSHI, ds_dnuc, ds_dw]
    Nparam = len(derivs)
    
    # Build Fisher matrix
    F = np.zeros((Nparam, Nparam))
    for i in range(Nparam):
        for j in range(Nparam):
            F[i,j] = np.sum(derivs[i] * derivs[j] / sigma_nu**2.)
    
    return F


params = {
    'S_HI':     20000.,    # uJy
    'nu_c':     1350.,   # MHz
    'w':        20.,     # MHz
    'x_fill':   1.
}

# Specify frequency channels
nu = np.linspace(1050., 1420., 500) # FIXME

F = fisher(nu, params)
cov = np.linalg.inv(F)

print "S_HI = %3.3e +/- %3.3e" % (params['S_HI'], np.sqrt(cov[0,0]))
print "nu_c = %3.3e +/- %3.3e" % (params['nu_c'], np.sqrt(cov[1,1]))
print "w    = %3.3e +/- %3.3e" % (params['w'], np.sqrt(cov[2,2]))

P.matshow(cov)
P.colorbar()
P.show()

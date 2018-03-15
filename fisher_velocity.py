#!/usr/bin/env python
"""
Fisher forecasts for joint density + velocity survey with an SKA1 HI galaxy 
survey, following the formalism in arXiv:1312.1022.
"""
import numpy as np
import pylab as P
from scipy.integrate import simps
import pyccl as ccl
import time, copy

KMIN = 1e-4 # Mpc^-1
KMAX = 0.3 # Mpc^-1
NK = 800
NMU = 300

SAREA = 5000. # deg^2


def bias(z):
    """
    Galaxy bias as a function of redshift, b(z).
    """
    # FIXME
    return np.sqrt(1. + z)

def corrfac(z):
    """
    Correlation coefficient between galaxy distribution and velocity field, r(z).
    """
    r_g = 0.98
    return r_g + 0.*z # FIXME

def sigma_g(z):
    """
    Non-linear velocity rms, used as the smoothing scale in the Fingers of God 
    expression for galaxies.
    """
    sigma_g = 5.1 # Mpc/h # FIXME: Wrong units!
    return sigma_g + 0.*z # FIXME

def sigma_u(z):
    """
    Non-linear velocity rms, used as the smoothing scale for the redshift-space 
    velocity field.
    """
    sigma_u = 13. # Mpc/h # FIXME: Wrong units!
    return sigma_u + 0.*z # FIXME

def sigma_unoise(zc):
    """
    Noise rms on the peculiar velocity measurement. FIXME: Units
    """
    return 1. # FIXME
    
def n_gal(zc):
    """
    Number density of detected HI galaxies in a given redshift bin, n ~ Mpc^-3.
    """
    return 1. # FIXME

def n_vel(zc):
    """
    Number density of detected HI galaxies with peculiar velocity measurements, 
    n_vel ~ Mpc^-3.
    """
    return 1. # FIXME
    

def build_ccl_cosmo(params):
    """
    Build CCL Cosmology object for a given set of parameters.
    """
    # Define a default set of parameters
    default_params = {
        'Omega_c':  0.2647,
        'Omega_b':  0.0492,
        'h':        0.673,
        'A_s':      2.207e-9,
        'n_s':      0.9645,
        'w0':      -1.,
        'wa':      0.,
    }
    p = copy.copy(default_params)
    
    # Replace parameters that were defined
    for pn in params.keys():
        
        # Add parameter to dictionary if allowed
        if pn == '10^9A_s':
            # Treat A_s separately; rescale by 1e9
            p['A_s'] = params[pn] / 1e9
            
        elif pn in default_params.keys():
            p[pn] = params[pn]
            
        else:
            # Ignores non-cosmological parameters
            pass
    
    # Create CCL Cosmology() object and return
    return ccl.Cosmology(**p)

def signal_covariance(zc, cosmo):
    """
    Signal power spectrum matrix, containing (cross-)spectra for gg, gv, vv. 
    These are simply the galaxy auto-, galaxy-velocity cross-, and velocity 
    auto-spectra.
    """
    # Scale factor at central redshift
    a = 1. / (1. + zc)
    
    # Grid of Fourier wavenumbers
    k = np.logspace(KMIN, KMAX, NK)
    mu = np.linspace(-1., 1., NMU)
    K, MU = np.meshgrid(k, mu)
    
    # Get matter power spectrum
    pk = ccl.linear_matter_power(cosmo, k, a=1.)
    
    # Get redshift-dep. functions
    b = bias(zc)
    rg = corrfac(zc)
    f = ccl.growth_rate(cosmo, a)
    beta = f / b
    H = ccl.h_over_h0(cosmo, a) * 100. * cosmo['h'] # km/s/Mpc
    
    # Redshift-space suppression factors
    D_g = 1. / np.sqrt(1. + 0.5*(K*MU*sigma_g(zc))**2.)
    D_u = np.sinc(K*sigma_u(zc))
    
    # Build 2x2 matrix of mu- and k-dependent pre-factors of P(k)
    fac = np.zeros((2, 2, mu.size, k.size)).astype(complex)
    
    # galaxy-galaxy
    fac[0,0] = b**2. * (1. + 2.*rg*beta*MU**2. + beta**2.*MU**4.) * D_g**2.
    
    # velocity-velocity
    fac[0,1] = 1.j * a*H*f*MU * (rg + beta*MU**2.) * D_g * D_u / K
    fac[1,0] = -1. * fac[0,1] # Complex conjugate
    
    # galaxy-velocity
    fac[1,1] = (a*H*f*MU)**2. * (D_u / K)**2.
    
    # Multiply all elements by P(k) and return
    return fac * pk[np.newaxis,np.newaxis,np.newaxis,:]


def noise_covariance(zc):
    """
    Noise covariance matrix for galaxy and velocity fields (assuming no 
    dependence on k or mu).
    """
    noise = np.zeros((2,2))
    noise[0,0] = 1. / n_gal(zc)
    noise[1,1] = sigma_unoise(zc)**2. / n_vel(zc)
    return noise


def inverse_covariance(cs, cn):
    """
    Calculate inverse of signal + noise covariance.
    """
    # cs shape: (2,2, NMU, MK)
    # cn shape: (2,2)
    ctot = cs + cn[:,:,np.newaxis,np.newaxis]
    
    # Calculate inverse using analytic inversion formula for 2x2 matrices
    cinv = np.zeros(ctot.shape).astype(complex)
    cinv[0,0] = ctot[1,1]; cinv[1,1] = ctot[0,0]
    cinv[0,1] = -ctot[0,1]; cinv[1,0] = -ctot[1,0]
    det = ctot[0,0]*ctot[1,1] - ctot[0,1]*ctot[1,0]
    return cinv / det


def signal_derivs(zc, params, dparams):
    """
    Calculate derivatives of signal covariance w.r.t. cosmological parameters.
    """
    derivs = []
    # Loop over parameters
    pnames = params.keys()
    pnames.sort()
    for pname in pnames:
        print "  Derivative for %s" % pname
        dp = dparams[pname]
        
        # Setup new cosmologies with +/- dparam
        pp = copy.copy(params); pm = copy.copy(params)
        pp[pname] += dp; pm[pname] -= dp
        cosmo_p = build_ccl_cosmo(pp)
        cosmo_m = build_ccl_cosmo(pm)
        
        # Calculate derivatives for each z bin
        deriv_z = []
        for _z in zc:
            print "    zc = %3.3f" % _z
            cs_p = signal_covariance(_z, cosmo_p)
            cs_m = signal_covariance(_z, cosmo_m)
            deriv_z.append((cs_p - cs_m) / (2.*dp))
        derivs.append(deriv_z)
    return pnames, np.array(derivs)


def integrate_fisher(zc, cs, cn, derivs):
    """
    Integrate Fisher matrix integrand over k and mu, given fiducial signal and 
    noise covariances, and derivatives w.r.t. parameters.
    """
    # Grid of Fourier wavenumbers
    k = np.logspace(KMIN, KMAX, NK)
    mu = np.linspace(-1., 1., NMU)
    
    # Calculate inverse covariance
    cinv = inverse_covariance(cs, cn)
    
    # Form integrand for Fisher integral by looping over params
    Nparam = derivs.shape[0]
    Fij = np.zeros((Nparam, Nparam))
    for zidx in range(zc.size):
        for i in range(Nparam):
            integ_i = np.einsum('ij...,jk...->ik...', cinv, derivs[i])
            
            for j in range(i, Nparam):
                #print("      i,j = %d,%d" % (i,j))
                integ_j = np.einsum('ij...,jk...->ik...', cinv, derivs[j])
                
                # Get total integrand and take trace
                integ = np.einsum('ij...,jk...->ik...', integ_i, integ_j)
                integ = np.einsum('ii...', integ)
                
                # FIXME: Make sure it's actually real!
                #print "Imaginary part:", np.sum(np.abs(integ.imag))
                #print(integ.shape)
                
                # Integrate over mu and k axes
                y = simps(integ.real, mu, axis=0)
                Fij[i,j] = Fij[j,i] = simps(k**2. * y, k) # FIXME: Factor of k?
    
    # Apply constant factors
    Fij *= 1. / (8.*np.pi**2.)
    return Fij
            
    

if __name__ == '__main__':
    
    # Define fiducial parameters
    params = {
        'Omega_c':  0.2647,
        'Omega_b':  0.0492,
        'h':        0.673,
        '10^9A_s':      2.207,
        'n_s':      0.9645,
        'w0':      -1.,
        'wa':      0.,
    }
    dparams = {
        'Omega_c':  0.005,
        'Omega_b':  0.001,
        'h':        0.005,
        '10^9A_s':  0.1,
        'n_s':      2e-3,
        'w0':       0.02,
        'wa':       0.05,
    }
    
    # Define fiducial cosmology
    cosmo0 = build_ccl_cosmo(params)
    
    # Define redshift bins
    zbins = np.linspace(0., 0.4, 5)
    zc = 0.5 * (zbins[1:] + zbins[:-1])
    
    t0 = time.time()
    
    # Get signal covariance
    cs = signal_covariance(zc[0], cosmo0)
    
    # Noise covariance
    cn = noise_covariance(zc[0])
    
    # Calculate derivatives of signal covariance w.r.t. cosmological parameters
    pnames, derivs = signal_derivs([zc[0],], params, dparams)
    print "derivs:", derivs.shape
    
    Fij = integrate_fisher(zc[0], cs, cn, derivs[:,0])
    print "Fij:", Fij.shape
    print Fij
    print pnames
    
    # Multiply Fisher matrix for each z bin by volume factor
    fsky = SAREA / (4.*np.pi*(180./np.pi)**2.) # SAREA assumed to be in deg^2
    rbins = ccl.comoving_radial_distance(cosmo0, 1./(1. + zbins))
    Veff = fsky * (4./3.)*np.pi * (rbins[:-1]**3. - rbins[1:]**3.)
    
    print "Run took %2.2f sec." % (time.time() - t0)
    
    

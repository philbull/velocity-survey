#!/usr/bin/env python
"""
Plot Fisher matrix output by fisher_velocity.py.
"""
import numpy as np
import pylab as P

def load_fisher_matrix(fname):
    """
    Load Fisher matrix and parameter names from file.
    """
    # Load Fisher matrix array
    Fz = np.load("%s.npy" % fname) # (Nz, Nparam, Nparam)
    
    # Load parameter names
    f = open("%s.params" % fname, 'r')
    pnames = f.readline().split(' ')
    f.close()
    return Fz, pnames

def corr(m):
    """
    Return correlation matrix.
    """
    corr = np.zeros(m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            corr[i,j] = m[i,j] / np.sqrt(m[i,i] * m[j,j])
    return corr
    

#Fz, pnames = load_fisher_matrix("Fz_test_lownoise")
Fz, pnames = load_fisher_matrix("Fz_test_novel")
print Fz.shape
print pnames

#P.subplot(111)
P.matshow(corr(Fz[0]), cmap='RdBu', vmin=-1., vmax=1.)
P.colorbar()

#P.tight_layout()
P.show()

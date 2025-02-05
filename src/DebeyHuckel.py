# the following equations considers that the RSP can varies with respect to temperature compositio and density.

# the derivatives with respect to temperature should be used with caution, because they were not yet used and can have errors.
# By dimensional analysis, rho must be in 1/m³ and the ion diameter in m, permittivity in C²/J.m

import numpy as np
from Const import kb, eCharge as eC
from numpy import pi as pi


def kdebyeF(ro_metro, T, perm, x, z):
    
    ro = ro_metro
    k = ro * eC ** 2 / (kb * T * perm) * np.dot(x, z ** 2)
    
    return k ** 0.5


def XjF(k, a_metro):
    
    a = a_metro
    
    T1 = 3 / (k * a) ** 3
    T2 = 3 / 2
    T3 = np.log(1 + k * a)
    T4 = - 2 * (1 + k * a)
    T5 = 1 / 2 * (1 + k * a) ** 2
    
    Xj = T1 * (T2 + T3 + T4 + T5)
    
    return Xj
    

def aionF(k, T, perm, x, z, Xj):
    
    T1 = - k * eC ** 2
    T2 = 12 * pi * kb * T * perm
    T3 = np.dot(x * z ** 2, Xj)  # sum of x * z² * Xj
    
    aion = T1 / T2 * T3
    
    return aion


def dkdebyedro_metroF(k, ro_metro, perm, dpermdro_metro):
    
    ro = ro_metro
    dperm = dpermdro_metro
    
    
    dK = k / 2 * (1 / ro - 1 / perm * dperm)
    
    return dK


def dXjdkF(k, a_metro, Xj):
        
    a = a_metro
    
    dX = 3 / k * ( 1 / (1 + k * a) - Xj)
    
    return dX


def dXjdro_metroF(dXjdk, dkdro_metro):
    
    dXro = dXjdk * dkdro_metro
    
    return dXro


def zion(ro_metro, aion, k, dkdro_metro, perm, dpermdro_metro, T, x, z, dxjdro_metro):
    
    ro = ro_metro
    dperm = dpermdro_metro
        
    T1 = 1 / k * dkdro_metro - 1 /perm * dperm
    T2 = k * eC ** 2 / (12 * pi * kb * T * perm) * np.dot(x * z ** 2, dxjdro_metro)
    
    zele = (aion * T1 - T2) * ro
    
    return zele


def dkdebyedxF(k, x, z, perm, dpermdx):
    
    eps = perm
    depsdx = dpermdx
    
    dkdx = k / 2 * (z ** 2 / np.dot(x, z ** 2) - 1 / eps * depsdx)
    
    return dkdx


def dXjdxF(dxjdk, dkdx):
    
    
    dxjdx = np.zeros((len(dxjdk), len(dxjdk)))
    
    for j, i in enumerate(dxjdk):
    
        dxjdx[j] = i * dkdx
    
    return dxjdx


def daiondxF(aion, k, dkdx, eps, depsdx, T, z, x, xj, dxjdx):

    dadx = np.zeros(len(x)) 
    dxdx = np.eye(len(x))
    
    T2 = - k * eC ** 2 / (12 * pi * kb * T * eps)
    
    for i, m in enumerate(x):
        
        T1 = 1 / k * dkdx[i] - 1 / eps * depsdx[i]
        T3 = np.dot(z ** 2, dxdx[:,i] * xj + x * dxjdx[:,i])
        dadx[i] = aion * T1 + T2 * T3
        
    return dadx


def dkdebyedtF(k, T, eps, depsdt):
    
    dkdt = - k / 2 * (1 / T + 1 / eps * depsdt)
    
    return dkdt

def dXjdTF(dxjdk, dkdt):
    
    dxjdt = dxjdk * dkdt
    
    return dxjdt


def daiondtF(aion, k, dkdt, T, eps, depsdt, x, z, dxjdt):
    
    T1 = aion * (1 / k * dkdt - 1 / T - 1 / eps * depsdt)
    T2 = - k * eC ** 2 / (12 * pi * kb * T * eps)
    T3 = np.dot(x * z ** 2, dxjdt)
    
    da = T1 + T2*  T3
    
    return da

    
import numpy as np
from numpy import pi
from  Const import kb, eCharge as chargeE, Permvac as epsv

# this module contains the functions to calculate de Born term and its first order derivatives
# the derivatives with respect to temperature should be used with caution, because they were not yet used and can have errors.


def abornF(x, T, z, rborn, epsr):
    
    # epsr is dimensionless
    # sig is in Angstrom and is converted to meter inside the function
    
    n = len(x)
    ai = rborn / 1e10
    
    eps0 = epsv  # esp0 is the vacuum permittivity in C²/N.m²
    
    
    soma = 0
    for i in range(n):
        if ai[i] != 0:
            soma +=  x[i] * z[i] ** 2 / ai[i]
       
    T1 = chargeE ** 2 / (4 * pi * kb * T * eps0)
    aborn = - T1 * (1 - 1 / epsr) * soma
    
    return aborn


def daborndxF(x, T, Prop, epsr, deps):
    
    # epsr is dimensionless
    # deps is the RSP (dimensionless) derivative with respect to molar fractions
    
    nc = len(x)
    z = Prop.charge
    a = Prop.rborn / 1e10
    eps0 = epsv
    da = np.zeros(nc)
    
    C = - (chargeE ** 2) / (4 * pi * kb * T * eps0)    

    soma = 0
    for i in range(nc):
        if a[i] != 0:
            soma+= x[i] / a[i] * z[i] ** 2

    for i in range(nc):
        T3 = 0
        T2 = 0
        T1 = (1 / epsr ** 2) * deps[i] * soma
        if a[i] != 0:
            T2 = (1 - 1 / epsr) * z[i] ** 2 / a[i]
        T3 = T1 + T2
        
        da[i] = C * T3
    
    return da

def daborndroF(T, epsr, depsrdro_metro, x, Prop):
    
    # depsdrois the RSP (dimensionless) derivative with respect to rho
    
    nc = len(x)
    deps = depsrdro_metro
    a = Prop.rborn / 1e10
    z = Prop.charge
    eps0 = epsv
    
    C = - (chargeE ** 2) / (4 * pi * kb * T * eps0)
    
    soma = 0
    for i in range(nc):
        if a[i] != 0:
            soma += x[i] * z[i] ** 2 / a[i]
        
    da = C / epsr ** 2 * deps * soma
    
    return da


def daborndTF(T, epsr, x, Prop, depsdT):
    
    nc = len(x)
    eps0 = epsv
    a = Prop.rborn / 1e10
    z = Prop.charge
    deps = depsdT
    
    C = - (chargeE ** 2) / (4 * pi * kb * T * eps0)
    
    soma = 0
    for i in range(nc):
        if a[i] != 0:
            soma += x[i] * z[i] ** 2 / a[i]
        
    da = C * soma * (1 / epsr ** 2 * deps - (1 / T) * (1 - 1/epsr))
    
    return da
    
    
    
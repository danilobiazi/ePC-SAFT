import numpy as np
from scipy import optimize

# This module contains all functions required to calculate the Association term and its first order derivatives
# Derivatives with respect to density and composition extensively tested. 
# However, the derivatives with respect to temperature should be used with caution, because they were not yet used and can have errors.

# Derivatives calculations based on Tan et al: Ind. Eng. Chem. Res. 2004, 43, 203-208
# DeltaAB: using sigij, not dij. However, Sadowski uses dij.

def DeltaF(sigij, ghsij, kAB, eAB, T):

    s = sigij
    g = ghsij
    k = kAB
    e = eAB

    Delta = s ** 3 * g * k * (np.exp(e / T) - 1)

    return Delta


def eABijF(eAB, lambij, Nc):

    eABij = np.zeros([Nc, Nc])

    for i in range(Nc):
        for j in range(Nc):
            eABij[i][j] = 0.5 * (eAB[i] + eAB[j]) * (1 - lambij[i][j])  # lambij is the binary interaction parameter for the association energy
    return eABij


def kABijF(kAB, sig, Nc):

    k = kAB
    s = sig

    kABij = np.zeros([Nc, Nc])

    for i in range(Nc):
        for j in range(Nc):
            a = (k[i] * k[j]) ** 0.5
            b = (s[i] * s[j]) ** 0.5
            c = 0.5 * (s[i] + s[j])
            kABij[i][j] = a * (b / c) ** 3

    return kABij


def DeltaABF(S, Sinais, sigij, ghsij, kABij, eABij, T, Nc, di, charge, flag_ion):


    Ite = len(Sinais[0])
    Delta = np.zeros([Ite, Nc, Ite, Nc])

    for i in range(Ite):
        for j in range(Nc):
            if i > 0:
                x1 = i
            else:
                x1 = 0
            for k in range(x1, Ite):
                if k == i:
                    x2 = j
                else:
                    x2 = 0
                for m in range(x2, Nc):
                    # site i, component j
                    # site k, component m
                    
                    # the restrictions to association are inserted below, by setting Delta AiBj = 0 when conditions are met.
                    
                    if Sinais[j, i] == '0' or Sinais[m, k] == '0':  # disables calculation to non-associating components
                        Delta[i, j, k, m] = 0
                    elif charge[j] * charge[m] != 0:      # association between ions not allowed
                        Delta[i, j, k, m] = 0
                    elif Sinais[j, i] == Sinais[m,k]:
                        Delta[i, j, k, m] = 0           # only donor-acceptor association allowed
                    elif np.logical_and(flag_ion == 0, np.logical_or(charge[j] != 0, charge[m] != 0)):    # disables ions associations in the "normal" association term
                        Delta[i, j, k, m] = 0
                    elif np.logical_and(flag_ion == 1, np.logical_and(charge[j] == 0, charge[m] == 0)):   # association between solvent molecules not allowed in assoc ion-solv
                        Delta[i, j, k, m] = 0

                    else:  # sigij in DeltaF (not dii)
                        Delta[i, j, k, m] = DeltaF((sigij[j,j] + sigij[m,m])/2, ghsij[j, m], kABij[j, m], eABij[j, m], T)

                    Delta[k, m, i, j] = Delta[i, j, k, m]
    return Delta


def XF(XA, ro, x, S, DeltaAB, SiteType, Nc):

    n = np.shape(SiteType)[0]

    Xa = np.zeros(Nc * n)

    for i in range(Nc):
        for j in range(n):
            soma1 = 0
            for k in range(Nc):
                soma2 = 0
                for m in range(n):
                    soma2 += S[k, m] * XA[m + k * n] * DeltaAB[j, i, m, k]
                soma1 += x[k] * soma2
            Xa[j + i * n] = 1 / (1 + ro * soma1)

    return Xa


def XFzero(XA_old, ro, x, S, DeltaAB, SiteType, Nc):
    
    # function in which its result must be zero if Xa converges (Xa_old - Xa_new)

    n = np.shape(SiteType)[0]

    deltaXa = np.zeros(Nc * n) 

    for i in range(Nc):
        for j in range(n):
            soma1 = 0
            for k in range(Nc):
                soma2 = 0
                for m in range(n):
                    soma2 += S[k, m] * XA_old[m + k * n] * DeltaAB[j, i, m, k]
                soma1 += x[k] * soma2
            deltaXa[j + i * n] = 1 / (1 + ro * soma1) - XA_old[j + i * n]

    return deltaXa


def XConv(ro, x, S, DeltaAB, SiteType, Nc):

    n = len(SiteType)

    Xs = np.zeros(Nc * n)

    for i in range(Nc * n):   # initial guess
        Xs[i] = 0.5

    for i in range(10):  # 10 x successive substitution
        Xs = XF(Xs, ro, x, S, DeltaAB, SiteType, Nc)   
        
    # using scipy fsolve to find Xa 
    Xs = optimize.fsolve(XFzero, Xs, args=(ro, x, S, DeltaAB, SiteType, Nc),full_output=1,xtol=1.e-10)
    

    a = np.zeros((Nc, n))
    Xr = Xs[0]

    indice = 0
    for i in range(Nc):
        for j in range(n):
            a[i, j] = Xr[indice]
            indice += 1

    return [a, Xs]


def LambdaF(ro, x, S, DeltaAB, Xa, nSitios, Nc):
    
    # Lambda function according to Tan's method

    st = nSitios
    Lambda = np.zeros((st * Nc, st * Nc))

    p = -1
    q = -1

    for j in range(st):
        for i in range(Nc):
            p += 1
            for L in range(st):
                for k in range(Nc):
                    q += 1
                    if L != j:
                        Lambda[p, q] = ro * x[k] * S[k, L] * DeltaAB[L, k, j, i] * Xa[i, j] ** 2
                    else:
                        if k == i:
                            Lambda[p, q] = 1
                        else:
                            Lambda[p, q] = 0
            q = -1

    return Lambda


def dDeltadRoF(di, sigij, kABij, eABij, T, dghsdro, s, DeltaAB, ro, Nc):

    # s = the number of site types

    dDelta = np.zeros((s, Nc, s, Nc))

    for i in range(Nc):
        for j in range(s):
            for k in range(Nc):
                for L in range(s):
                    if DeltaAB[j, i, L, k] == 0:
                        dDelta[j, i, L, k] = 0
                    else:
                        # sigij, not dij                        
                        a = ((sigij[i,i]+sigij[k,k])/2) ** 3 * kABij[i, k]
                        b = (1 / ro * dghsdro[i, k])
                        c = (np.exp(eABij[i, k] / T) - 1)
                        dDelta[j, i, L, k] = a * b * c

    return dDelta


def PsiRoF(Xa, ro, x, S, s, dDeltadRo, Nc):

    # according to Tan's method
    Psi = np.zeros(s * Nc)

    for i in range(Nc):
        for j in range(s):
            soma2 = 0
            for k in range(Nc):
                soma1 = 0
                for L in range(s):
                    soma1 += S[k, L] * Xa[k, L] * dDeltadRo[L, k, j, i]
                soma2 += x[k] * soma1
            Psi[Nc * j + i] = -(Xa[i, j] ** 2) * ((1 / ro) * (1 / Xa[i, j] - 1) + ro * soma2)

    return Psi


def dXadroF(Lambda, PsiRo, Nc, s):
    
    # according to Tan's method
    dXa = np.dot(Lambda, PsiRo)

    a = np.zeros((Nc, s))

    p = 0
    for j in range(s):
        for i in range(Nc):
            a[i, j] = dXa[p]
            p += 1

    return a


def ZassocF(ro, Xa, x, S, dXadRo, Nc, s):

    soma1 = 0
    for i in range(Nc):
        soma2 = 0
        for j in range(s):
            soma2 += S[i, j] * dXadRo[i, j] * (1 / Xa[i, j] - 0.5)
        soma1 += x[i] * soma2

    Zassoc = ro * soma1
    return Zassoc


def dDeltadXF(eABij, di, sigij, kABij, dghsdx, DeltaAB, T, s, Nc):

    dDelta = np.zeros((Nc, s, Nc, s, Nc))

    for m in range(Nc):
        for i in range(Nc):
            for j in range(s):
                for k in range(Nc):
                    for L in range(s):
                        if DeltaAB[j, i, L, k] == 0:
                            dDelta[m, j, i, L, k] = 0
                        else:
                            # sigij, not dij
                            a = ((sigij[i,i]+sigij[k,k])/2) ** 3
                            b = kABij[i, k] * dghsdx[m, i, k]
                            c = np.exp(eABij[i, k] / T) - 1
                            dDelta[m, j, i, L, k] = a * b * c

    return dDelta


def PsiXmF(Xa, ro, S, DeltaAB, x, dDeltadx, s, Nc):
    
    # according to Tan's method
    Psi = np.zeros((Nc, Nc * s))

    for m in range(Nc):
        for i in range(Nc):
            for j in range(s):
                soma1 = 0
                for L in range(s):
                    soma2 = 0
                    for k in range(Nc):
                        soma2 += x[k] * S[k, L] * Xa[k, L] * dDeltadx[m, j, i, L, k]

                    soma1 += S[m, L] * Xa[m, L] * DeltaAB[j, i, L, m] + soma2
                Psi[m, Nc * j + i] = -(Xa[i, j] ** 2) * ro * soma1

    return Psi


def dXadxF(Lambda, PsiX, s, Nc):
    
    # according to Tan's method
    dXa = Lambda @ np.transpose(PsiX)

    a = np.zeros((Nc, Nc, s))
    b = np.zeros((Nc, Nc * s))

    for i in range(Nc):
        for j in range(Nc * s):
            b[i, j] = dXa[j, i]

    for m in range(Nc):
        p = 0
        for j in range(s):
            for i in range(Nc):
                a[m, i, j] = b[m, p]
                p += 1

    return a


def daassdxF(S, Xa, x, dXadx, s, Nc):

    dass = np.zeros(Nc)

    for i in range(Nc):
        soma1 = 0
        for j in range(s):
            soma1 += S[i, j] * (np.log(Xa[i, j]) - 0.5 * Xa[i, j] + 0.5)

        soma2 = 0
        for k in range(Nc):
            soma3 = 0
            for j in range(s):
                soma3 += S[k, j] * dXadx[i, k, j] * (1 / Xa[k, j] - 0.5)

            soma2 += x[k] * soma3

        dass[i] = soma1 + soma2

    return dass


def AassocF(Nc, s, x, S, Xa):

    soma1 = 0
    for i in range(Nc):
        soma2 = 0
        for j in range(s):
            soma2 += S[i, j] * (np.log(Xa[i, j]) - 0.5 * Xa[i, j] + 0.5)

        soma1 += x[i] * soma2

    return soma1



def dDeltadTF(di, kAB, ghs, ekAB, dghsT, T, s, Delta, Nc, sig, eps):


    a = np.zeros((s, Nc, s, Nc))

    for i in range(Nc):
        for j in range(s):
            for k in range(Nc):
                for L in range(s):
                    if Delta[L, k, j, i] == 0:
                        a[L, k, j, i] = 0
                    else:
                        f = ghs[i, k]
                        g = dghsT[i, k]
                        h = np.exp(ekAB[i, k] / T) - 1
                        p = -ekAB[i, k] / T ** 2 * np.exp(ekAB[i, k] / T)
                        a[L, k, j, i] = sig[i, k] ** 3 * kAB[i, k] * (f * p + h * g)

    return a

def PsiTF(Xa, ro, S, x, s, dDeltadT, Nc):

    # s = Prop.SiteType.Count

    a = np.zeros(Nc * s)

    for i in range(Nc):
        for j in range(s):
            soma1 = 0
            for k in range(Nc):
                soma2 = 0
                for L in range(s):
                    soma2 += S[k][L] * Xa[k][L] * dDeltadT[j][i][L][k]

                soma1 += x[k] * soma2

            a[Nc * j + i] = -(Xa[i][j] ** 2) * ro * soma1

    return a


def dXadTF(Lambda, PsiT, Nc, s):

    dXa = np.dot(Lambda, PsiT)

    a = np.zeros((Nc, s))

    p = 0
    for j in range(s):
        for i in range(Nc):
            a[i][j] = dXa[p]
            p += 1

    return a


def daassocdTF(x, S, Xa, dXdT, s, Nc):

    soma1 = 0
    for i in range(Nc):
        soma2 = 0
        for j in range(s):
            soma2 += S[i][j] * dXdT[i][j] * (1 / Xa[i][j] - 0.5)

        soma1 += x[i] * soma2

    return soma1

        
    
    
    
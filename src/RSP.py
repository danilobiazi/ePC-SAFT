import numpy as np
from Const import Permvac as e0, kb
from scipy import optimize as opt


# This module contains all the equations required to calculate the RSP and its first order derivatives
# the derivatives with respect to temperature should be used with caution, because they were not yet used and can have errors.


def eps00(x, ro_metro, pol):
    
# Claussius-Mossoti equation
# Newton method to find root

# ro_metro: number density in 1/m³
# pol is the polarizability, in C².m²/J
   
    def sole00(e00, x, ro_metro, alpha):
        
        soma = np.dot(x, alpha)
        
        zerar = ro_metro / (3 * e0) * soma - (e00 - 1)/(e00 + 2)  # Claussius-Mossoti
        
        return zerar
    
    e00 = 3  # initial guess
    
    eps00 = opt.newton(sole00, e00, args=(x, ro_metro, pol),full_output=1,tol=1e-8, maxiter=100, disp=1)


    return eps00[1].root


def PAiBjF(ro, x, DeltaAB, Xa, S):
    
    # rho must be in 1/A^3
    
    nc = len(x)
    site = len(S[0])
    P = np.zeros([site, nc, site, nc])
    
    for A in range(site):
        for i in range(nc):
            for B in range(site):
                for j in range(nc):
                    P[B, j, A, i] = ro * x[j] * S[j,B] * DeltaAB[B,j,A,i] * Xa[i, A] * Xa[j, B]
    
    return P



def PijF(PAiBj):
    
    nc = len(PAiBj[0])
    site = len(PAiBj)
    
    P = np.ones([nc ,nc])
    P = P * -1
    
    for i in range(nc):
        for j in range(nc):
            for A in range(site):
                for B in range(site):
                    if PAiBj[B, j, A, i] != 0:
                        P[i,j] *= 1 - PAiBj[B, j, A, i]

    return P + 1



def PiF(Xa):
    
    nc = len(Xa)
    site = len(Xa[0])
    Pi = np.ones(nc)
    
    for i in range(nc):
        for j in range(site):
            if Xa[i,j] != 1:
                Pi[i] *= (Xa[i,j])
    
    return 1 - Pi


def thetaiF(PAiBj, charge):
    
    nc = len(PAiBj[0])
    site = len(PAiBj)
    th = np.ones(nc)
    
    for i in range(nc):
        for A in range(site):
            soma = 0
            for B in range(site):
                for j in range(nc):
                    if charge[j] !=0:
                        soma += PAiBj[B, j, A, i]
            
            th[i] *= 1 - soma
            
    return th


def giF(Pi, Pij, x, zij, dipolo, ang_gama, ang_theta):
    
    # cross angles are important for solvent mixtures. Can be define here or modify function to define outside
    # can be obtained having the single solvent angles (each solvent) and only one experimental point for the mixture

    gamaij = 90
    thetaij = 90
    
    
    nc = len(x)
    
    g = np.zeros(nc)
       
    
    for i in range(nc):
        if dipolo[i] != 0:
            g[i] = 1
            for j in range(nc):
                P0 = np.cos(np.radians(ang_gama[i]))
                P00 = np.cos(np.radians(ang_theta[i]))
                
                if i != j:
                    P0 = np.cos(np.radians(gamaij))
                    P00 = np.cos(np.radians(thetaij))
                
                C = P0 * dipolo[j] / dipolo[i]
                T1 = Pij[i, j] / (Pi[i] * P00 + 1)                                                   

                g[i] += zij[i]* C * T1
            
    return g


def EpsMariboF(e00, ro_m, T, x, tetai, gi, dipolo):
    
    def epszero(er, e00, ro_m, T, x, tetai, gi, dipolo): # function to find the root (both sides of Maribo's equation)
        
        nc = len(x)
        soma = 0
        for i in range(nc):                                 # convert dipole moment from D to C.m
            soma += x[i] * tetai[i] * gi[i] * (dipolo[i] * 3.33564095198 * 1e-30) ** 2
        
        P1 = (2 * er + e00) * (er - e00) / er
        P2 = (((e00 + 2) / 3) ** 2) * ro_m / (kb * T * e0) * soma
        
        zero = P1 - P2
    
        return zero
    
    er = 40  # initial guess
    
    # Newton method
    eps = opt.newton(epszero, er, args=(e00, ro_m, T, x, tetai, gi, dipolo),full_output=1,tol=1e-9, maxiter=100, disp=1)
    
    return eps[1].root


def dPAiBjdro_metroF(PAiBj, ro, x, DeltaAB, dDeltadro, Xa, dXadro):
    
    # rho in 1/A³
    
    dP = PAiBj * 0  # creating a zero matriz with the same dimensions of PAiBj
    site = len(PAiBj)
    nc = len(PAiBj[0])
    P = PAiBj
    D = DeltaAB
    dD = dDeltadro
    dX = dXadro
    
    for A in range(site):
        for i in range(nc):
            for B in range(site):
                for j in range(nc):
                    
                    T1 = P[B, j, A, i] / ro
                    
                    if D[B, j, A, i] == 0:
                        T2 = 0
                    else:
                        T2 = P[B, j, A, i] / D[B, j, A, i] * dD[B, j, A, i]
                    
                    if Xa[i, A] == 0:
                        T3 = 0
                    else:
                        T3 =  P[B, j, A, i] / Xa[i, A] * dX[i, A]
                    
                    
                    if Xa[j, B] == 0:
                        T4 = 0
                    else:
                        T4 = P[B, j, A, i] / Xa[j, B] * dX[j, B]
                    
                    
                    dP[B, j, A, i] = T1 + T2 + T3 + T4
    
    return dP / 1e30
 

def dPijdro_metroF(Pij, PAiBj, dPAiBjdro_metro):
    
    nc = len(PAiBj[0])
    site = len(PAiBj)
    
    dPij = np.zeros([nc, nc])
    
    dPA = dPAiBjdro_metro
    Pa = PAiBj
    
    for i in range(nc):
        for j in range(nc):
            soma = 0
            for A in range(site):
                for B in range(site):
                    soma += dPA[B,j,A,i] / (1 - Pa[B,j,A,i])
            
            dPij[i,j] = (1 - Pij[i,j]) * soma
    
    return dPij



def dPidro_metroF(Pi, Xa, dXadro):
    
    nc = len(Xa)
    dPi = np.zeros(nc)
    site = len(dXadro[0])
    
    for i in range(nc):
        soma = 0
        for A in range(site):
            soma += dXadro[i, A] / Xa[i, A]
            
        dPi[i] = (Pi[i] - 1) * soma / 1e30
        
    return dPi
        
        
    
def dthetaidro_metroF(Thetai, PAiBj, dPAiBjdro_metro):
    
    nc = len(PAiBj[0])
    site = len(PAiBj)
    dtheta = np.ones(nc)
    
    ti = Thetai
    dPA = dPAiBjdro_metro
    Pa = PAiBj
    
    for i in range(nc):
        soma = 0
        for A in range(site):
            soma1 = 0
            soma2 = 0
            for B in range(site):
                for j in range(nc):
                    soma1 += dPA[B,j,A,i]
                    soma2 = Pa[B,j,A,i]
            
            soma += (-soma1) / (1 - soma2)
            
        dtheta[i] = ti[i] * soma
    
    return dtheta


def dgidro_metroF(Prop, Pi, Pij, dPijdro_metro, dPidro_metro,zij):
    
    
    # cross angles
    gamaij = 90
    thetaij = 90
    
    nc = len(Prop.name)
    dgi = np.zeros(nc)
    gama = np.radians(Prop.ang_gama)
    theta = np.radians(Prop.ang_theta)
    dip = Prop.dipolo
    dPi = dPidro_metro
    dPij = dPijdro_metro

    
    for i in range(nc):
        if dip[i] == 0:
            dgi[i] = 0
            
        else:
            for j in range(nc):
                t1 = zij[i] * np.cos(gama[i]) * dip[j] / dip[i]
                t4 = (Pi[i] * np.cos(theta[i]) + 1)
                t2 = dPij[i, j] / t4
                t3 = Pij[i, j] * np.cos(theta[i]) * dPi[i]
                if i != j:
                    t1 = np.cos(gamaij) * dip[j] / dip[i]
                    t4 = (Pi[i] * np.cos(thetaij) + 1)
                    t2 = dPij[i, j] / t4
                    t3 = Pij[i, j] * np.cos(thetaij) * dPi[i]
                    
                dgi[i] += t1 * (t2 - t3/(t4 ** 2))
            
    return dgi
    
    
def deps00dromF(e00, x, Prop):
    
    deps = (e00 + 2) ** 2 / (9 * e0) * np.dot(x, Prop.polarizab)
    
    return deps

def depsrdro_m(er, e00, de00dro, rom, T, x, dip, gi, thetai, dgi, dthetai):
    
    dipc = dip * 3.33564095198 * 1e-30
    
    C2 = (2 * er + e00)*(er - e00) / er
    C0 = er / (4 * er - e00 - C2)
    C1a = de00dro * (er + 2 * e00) / er
    C1b = C2 * (de00dro * 2 / (e00 + 2) + 1 / rom)
    C1 = C1a + C1b
    
    soma = 0
    for i, xi in enumerate(x):
        soma += xi * dipc[i] ** 2 * (gi[i] * dthetai[i] + thetai[i] * dgi[i])
        
    S = ((e00 + 2) / 3) ** 2 * rom / (kb * T * e0) * soma
    
    return [C0 * (C1 + S), C0, C2]
    
    
def dPAiBjdxF(x, DeltaAB, dDeltadx, Xa, dXadx, PAiBj):
    
    site = len(Xa[0])
    nc = len(x)
    dP = np.zeros([nc, site, nc, site, nc])
    D = DeltaAB
    dD = dDeltadx
    dX = dXadx
    P = PAiBj
    
    for i in range(nc):
        for A in range(site):
            for k in range(nc):
                for B in range(site):
                    T1 = 0
                    for j in range(nc):
                        if j == i:
                            T1 = P[B, j, A, k] / x[i]
                        else:
                            T1 = 0
                            
                        if D[B, j, A, k] == 0:
                            T2 = 0
                        else:
                            T2 = P[B, j, A, k] / D[B, j, A, k] * dD[i, B, j, A, k]
                        
                        if Xa[k, A] == 0:
                            T3 = 0
                        else:
                            T3 = P[B, j, A, k] / Xa[k, A] * dX[i, k, A]
                        
                        if Xa[j, B] == 0:
                            T4 = 0
                        else:
                            T4 = P[B, j, A, k] / Xa[j, B] * dX[i, j, B]
                        
                        dP[i, B, j, A, k] = T1 + T2 + T3 + T4
    return dP



def dPijdxF(Pij, PAiBj, dPAiBjdx):
    
    nc = len(PAiBj[0])
    site = len(PAiBj)
    
    P = PAiBj
    dP = dPAiBjdx
    dPij = np.zeros([nc, nc, nc])
    
    for i in range(nc):
        for k in range(nc):
            for j in range(nc):
                soma = 0
                for A in range(site):
                    for B in range(site):
                        soma += dP[i, B, j, A, k] / (1 - P[B, j, A, k])
                
                dPij[i, k, j] = (1 - Pij[j, k]) * soma
    
    return dPij
                        

def dPidxF(Pi, Xa, dXadx):
    
    nc = len(Xa)
    site = len(Xa[0])
    dPi = np.zeros([nc, nc])
    
    for i in range(nc):
        for k in range(nc):
            
            soma = 0
            for A in range(site):
                soma += dXadx[i, k, A] / Xa[k, A]
            
            dPi[i, k] = (Pi[k] - 1) * soma
            
    return dPi


def dthetaidxF(Thetai, PAiBj, dPAiBjdx, charge):
    
    nc = len(PAiBj[0])
    site = len(PAiBj)
    
    dThetadx = np.zeros([nc, nc])
    
    for i in range(nc):
        for k in range(nc):
            soma = 0
            for A in range(site):
                soma1 = 0
                soma2 = 0
                for B in range(site):
                    for j in range(nc):
                        if charge[j] !=0:
                            soma1 += dPAiBjdx[i, B, j, A, k]
                            soma2 += PAiBj[B, j, A, k]
                
                soma += soma1 / (1 - soma2)
            
            dThetadx[i, k] = - Thetai[k] * soma
    
    return dThetadx
            
        
def dgidxF(Pi, Pij, dPijdx, dPidx, zij, dipolo, ang_theta, ang_gama):
    
    # cross angles
    gamaij = 90
    thetaij = 90

    nc = len(Pi)
    dgidx = np.zeros([nc, nc])

    
    for k in range(nc):
        for i in range(nc):
            if dipolo[i] != 0:
                
                soma = 0
                for j in range(nc):
                    
                    if i == j:
                        costheta = np.cos(np.radians(ang_theta[i]))
                        cosgama = np.cos(np.radians(ang_gama[i]))
                    else:
                        costheta = np.cos(np.radians(thetaij))
                        cosgama = np.cos(np.radians(gamaij))
                    
                    Cij = zij[i] * cosgama * dipolo[j] / dipolo[i]
                    t1 = (Pi[i] * costheta) + 1
                    t2 = 1 / t1 * dPijdx[k, i, j]
                    t3 = Pij[i, j] * costheta / (t1 ** 2)
                    t4 = t3 * dPidx[k, i]
                    
                    soma += Cij * (t2 - t4)
                
                dgidx[k, i] = soma
                
    
    return dgidx


def deps00dxF(e00, x, rom, Prop):

    # esta função está validada após verificação da derivada numérica.
    # a validação utilizou 3 componentes com associação 2B, 4C e 3B. Parece ok.
    # validação: metanol+água+MDEA
    nc = len(x)
    de00dx = np.zeros(nc)
    
    for i in range(nc):
        de00dx[i] = ((e00 + 2) ** 2) * rom * Prop.polarizab[i] / 9 / e0
    
    return de00dx


def depsrMaribodxF(er, e00, rom, T, de00dx, x, theta, g, Prop, dthetadx, dgdx):
    
    nc = len(x)
    de00 = de00dx
    dteta = dthetadx
    deps = np.zeros(nc)
    dip = Prop.dipolo * 3.33564 * 1e-30
    
    t1 = (2 * er + e00) * (er - e00) / er
    t2 = er / (4 * er - e00 - t1)
    t3 = rom / (kb * T * e0)
    t4 = (2 * e00 + 4) / 9
    t5 = ((e00 + 2) / 3) ** 2
    
    soma = 0
    for k in range(nc):
        soma += x[k] * theta[k] * g[k] * dip[k] ** 2
        
    for i in range(nc):
        
        soma1 = 0
        for k in range(nc):
            soma1 += x[k] * dip[k] ** 2 * (dteta[i, k] * g[k] + theta[k] * dgdx[i, k])
        
        t6 = 1 / er * de00[i] * (er + 2 * e00)
        t7 = theta[i] * g[i] * dip[i] ** 2
        deps[i] = t2 * (t3 * (t4 * de00[i] * soma + t5 * (t7 + soma1)) + t6)
        
    return deps


def desprdxF(C0, C2, de00dx, er, e00, rom, T, x, thetai, gi, dipolo, dthetadx, dgidx):
    
    nc = len(x)
    dip = dipolo * 3.33564 * 1e-30
    depsdx = np.zeros(nc)
    
    for i in range(nc):
        
        soma = 0
        for k in range(nc):
            soma += x[k] * dip[k] ** 2 * (gi[k] * dthetadx[i, k] + thetai[k] * dgidx[i, k])
            
        C3 = ((e00 + 2) / 3) ** 2 * rom / (kb * T * e0) * (thetai[i] * gi[i] * dip[i] ** 2 + soma)
        
        depsdx[i] = C0 * (de00dx[i] * ( (er + 2* e00) / er + 2 * C2 / (e00 + 2)) + C3)
        
    return depsdx
        
        
   
def dPAiBjdTF(ro, x, DeltaAB, dDeltadT, Xa, dXadT):
    
    # ro entra em 1/A³
    
    site = len(Xa[0])
    nc = len(x)
    dP = np.zeros([site, nc, site, nc])
    D = DeltaAB
    dD = dDeltadT
    dX = dXadT
    

    for A in range(site):
        for i in range(nc):
            for B in range(site):
                for j in range(nc):
                    T1 = ro * x[j]
                    T2 = dD[B, j, A, i] * Xa[i, A] * Xa[j, B]
                    T3 = D[B, j, A, i] * dX[i, A] * Xa[j, B]
                    T4 = D[B, j, A, i] * dX[j, B] * Xa[i, A]
                    dP[B, j, A, i] = T1 * (T2 + T3 + T4)
    
    return dP
    
 
def dPijdTF(PAiBj, dPAiBjdT):
    
    nc = len(PAiBj[0])
    site = len(PAiBj)

    P = PAiBj
    dP = dPAiBjdT
    T = np.zeros(site ** 2)
    dT = np.zeros(site ** 2)
    dPij = np.zeros([nc, nc])
    
    
    for i in range(nc):
        for j in range(nc):
            
            cont = 0
            for A in range(site):
                for B in range(site):
                    T[cont] = (1 - P[B, j, A, i]) #* S[j, B])                          
                    dT[cont] = -dP[B, j, A, i] #*S[j, B]
                    cont += 1
            

            for k, r in enumerate(dT):
                aux = np.copy(T)
                aux[k] = r
                dPij[i,j] -= np.prod(aux)
                    
    return dPij


def dPidTF(Xa, dXadT):
    
    dXa = dXadT
    nc = len(dXa)
    dP = np.zeros(nc)
        
    for i in range(nc):
        for j, k in enumerate(dXa[i]):
            aux = np.copy(Xa[i])
            aux[j] = k
            dP[i] -= np.prod(aux)
        
    return dP


def dthetaidTF(PAiBj, dPAiBjdT, Prop):
    
    nc = len(Prop.name)
    site = len(Prop.S[0])
    
    P = PAiBj
    dP = dPAiBjdT
    T = np.zeros(site)
    dT = np.ones(site)
    dTh = np.zeros(nc)
    
    for i in range(nc):
        
        for A in range(site):
            soma = 0
            somad = 0
            for B in range(site):
                for j in range(nc):
                    # if j == 1 or j == 2:
                    if Prop.charge[j] !=0:
                        soma += P[B, j, A, i] #* S[j, B]
                        somad += dP[B, j, A, i] #*S[j, B]
            
            T[A] = 1 - soma
            dT[A] = - somad

 
        for k, r in enumerate(dT):
            aux = np.copy(T)
            aux[k] = r
            dTh[i] += np.prod(aux)
        
    
    return dTh


def dgidT(Prop, Pi, dPidT, Pij, dPijdT, zij):

    
    # cross angles
    gamaij = 90
    thetaij = 90
    
    nc = len(Prop.name)
    dPi = dPidT
    dPij = dPijdT
    dgi = np.zeros(nc)

    
    for i in range(nc):
        T1 = Pi[i] * np.cos(np.radians(Prop.ang_theta[i])) + 1
        soma = 0
        if Prop.dipolo[i] != 0:
            for j in range(nc):
                T2 = zij[i] * np.cos(np.radians(Prop.ang_gama[i])) * Prop.dipolo[j] / Prop.dipolo[i]
                T3 = Pij[i, j] * np.cos(np.radians(Prop.ang_theta[i]))
                if i != j:
                    T2 = np.cos(np.radians(gamaij)) * Prop.dipolo[j] / Prop.dipolo[i]
                    T3 = Pij[i, j] * np.cos(np.radians(thetaij))
                    
                soma += T2 * (1 / T1 * dPij[i, j] - T3 / T1 ** 2 * dPi[i])
        
        dgi[i] = soma
        
    return dgi
        

def depsrMaribodT(er, e00, T, rom, x, Prop, g, theta, dthetadT, dgdT):
    
    nc = len(x)
    dth = dthetadT
    dip = dip = Prop.dipolo * 3.33564 * 1e-30
    
    soma = 0
    for i in range(nc):
        T1 = x[i] * dip[i] ** 2
        T2 = dth[i] * g[i] + theta[i] * dgdT[i]
        soma += T1 * T2
    
    T3 = (2 * er + e00) * (er - e00) / er
    T4 = er / (4 * er - e00 - T3)
    T5 = ((e00 + 2) / 3) ** 2
    T6 = rom / (kb * e0)
    
    deps = T4 * (1 / T) * (-T3 + T5 * T6 * soma) * 1e30
    
    return deps

    
    
        
    
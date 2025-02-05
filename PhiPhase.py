import Pressure
from scipy import optimize
import numpy as np
import PCSAFT as f
from Const import a, b, Nav, Permvac
import Association as fAs
import DebeyHuckel as dh
from numpy import zeros, log
import Born
import RSP as dm


def coeffugacidadeF(T, P, x, fase, Prop):
    
    # os parametros de interacao binaria, ser forem diferentes de zero, devem ser inseridos
    # no objeto Prop antes de chamar a funcao da fugacidade
    
    nc = Prop.nc

    if fase == 'L':
        eta0 = 0.45
    elif fase == 'V':
        eta0 = 1e-5

    di = f.diF(Prop.sig, Prop.eps, T)
    Prop.di = di
    
    dij = f.dijF(di)
    Prop.dij = dij
    
    sigij = f.sigijF(Prop.sig, Prop.lij)
    Prop.sigij = sigij


    ions = np.where(Prop.charge != 0)[0]  # the components which are ions (location in Prop.charge vector)
    cations = np.where(Prop.charge == 1)[0]
    anions = np.where(Prop.charge == -1)[0]
    
    # dispersive forces between equal charged ions are disabled by setting binary interaction parameters equal to 1
    if len(ions) > 0:
        for i in cations:
            for j in cations:
                Prop.kij[i,j] = 1
        for i in anions:
            for j in anions:
                Prop.kij[i,j] = 1
                
    ekij = f.ekijF(Prop.eps, Prop.kij); Prop.ekij = ekij
    MeanS = f.meanSegF(x, Prop.m); Prop.MeanS = MeanS
    mesig = f.mesigF(x, Prop.m, ekij, sigij, T); Prop.mesig = mesig
    me2sig = f.me2sigF(x, Prop.m, ekij, sigij, T); Prop.me2sig = me2sig
    Prop.MW_mean = np.dot(Prop.MM, x)
    
    

    # Método de Newton para encontrar eta que fornece a P especificada
    # Dentro do módulo 'Pressao', com a convergência, obtém-se z, ro, além de Xa e Lambda do termo assoc.
    
    etaR = optimize.newton(Pressure.Press, eta0, args=(x, P, T, Prop),full_output=1,tol=1e-09, maxiter=100, disp=0) 
    Prop.eta = etaR[1].root
    Prop.msgP = 'Pressão: ' + etaR[1].flag + ' em ' + str(etaR[1].iterations) + ' iterações.'
    
    eta = etaR[1].root
    Pressure.Press(eta, x, P, T, Prop)
    
    # For code debug
    #print(Prop.msgP)
    #print(Prop.msgXa)
    #print(Prop.msgXa_ion)
    
    Prop.Density_Mol = Prop.ro / (1e-30 * Nav)
    Prop.Density_Mass = Prop.Density_Mol * Prop.MW_mean / 1000

    ahs = f.ahsF(Prop.csi); Prop.ahs = ahs

    ahc = f.ahcF(MeanS, ahs, x, Prop.m, Prop.ghsij)
    Prop.ahc = ahc

    adisp = f.adispF(Prop.I1, mesig, Prop.ro, MeanS, Prop.C1, Prop.I2, me2sig)
    Prop.adisp = adisp

    csik = f.csikF(Prop.ro, Prop.m, di)
    Prop.csik = csik
    
    dghsdx = f.dghsdxF(csik, Prop.csi, dij)
    dahsdx = f.dahsdxF(csik, ahs, Prop.csi, nc)
    dahcdx = f.dahcdxF(Prop.m, ahs, MeanS, dahsdx,
                       x, Prop.ghsij, dghsdx, nc); Prop.dahcdx = dahcdx

    I1k = f.I1kF(Prop.m, a, MeanS, csik, eta, nc)
    I2k = f.I2kF(Prop.m, b, MeanS, csik, eta, nc)
    C1k = f.C1kF(Prop.C1, Prop.C2, eta, csik, Prop.m, nc)
    mesigk = f.mesigkF(Prop.m, x, ekij, T, sigij, nc)
    me2sigk = f.me2sigkF(Prop.m, x, ekij, T, sigij, nc)

    dadispdx = f.dadispdxF(Prop.ro, I1k, mesig, Prop.I1, mesigk, Prop.m,
                           Prop.C1, Prop.I2, MeanS, C1k, I2k, me2sig,
                           me2sigk, nc); Prop.dadispdx = dadispdx

    daassdx = zeros(nc)
    aassoc = 0

    if np.sum(Prop.S[np.where(Prop.charge == 0)]) > 0:
        # verifies if the non-charge components have associating sites
        
        s = len(Prop.SiteType)
        aassoc = fAs.AassocF(nc, s, x, Prop.S, Prop.Xa); Prop.aassoc = aassoc
        dDeltadx = fAs.dDeltadXF(Prop.eABij, di, Prop.sigij, Prop.kABij, dghsdx,
                                 Prop.DeltaAB, T, s, nc)
        Prop.dDeltadx = dDeltadx

        PsiX = fAs.PsiXmF(Prop.Xa, Prop.ro, Prop.S, Prop.DeltaAB, x,
                          dDeltadx, s, nc); Prop.PsiX = PsiX
        dXadx = fAs.dXadxF(Prop.Lambda, PsiX, s, nc); Prop.dXadx = dXadx
        daassdx = fAs.daassdxF(Prop.S, Prop.Xa, x, dXadx, s, nc); Prop.daassdx = daassdx


    aion = 0
    aassoc_ion = 0
    aborn = 0
    ro_metro = Prop.ro * 1e30  # converting ro to 1/m³
    a_metro = Prop.sig / 1e10  # converting sig to meter
    daiondx = zeros(nc)
    daborndx = zeros(nc)
    daassdx_ion = zeros(nc)
    

    if Prop.charge.any() != 0:
                
        aion = dh.aionF(Prop.Kd, T, Prop.perm, x, Prop.charge, Prop.Xj)
                  
        s = len(Prop.SiteType)
        aassoc_ion = fAs.AassocF(nc, s, x, Prop.S_ion, Prop.Xa_ion); Prop.aassoc_ion = aassoc_ion
        dDeltadx_ion = fAs.dDeltadXF(Prop.eABij_ion, di, Prop.sigij, Prop.kABij_ion, dghsdx, Prop.DeltaAB_ion, T, s, nc)
        Prop.dDeltadx_ion = dDeltadx_ion
        PsiX_ion = fAs.PsiXmF(Prop.Xa_ion, Prop.ro, Prop.S_ion, Prop.DeltaAB_ion, x, dDeltadx_ion, s, nc)
        Prop.PsiX_ion = PsiX_ion
        dXadx_ion = fAs.dXadxF(Prop.Lambda_ion, PsiX_ion, s, nc); Prop.dXadx_ion = dXadx_ion
        daassdx_ion = fAs.daassdxF(Prop.S_ion, Prop.Xa_ion, x, dXadx_ion, s, nc); Prop.daassdx_ion = daassdx_ion
    
        dPAiBjdx = dm.dPAiBjdxF(x, Prop.DeltaAB, dDeltadx, Prop.Xa, dXadx, Prop.PAiBj)
        dPAiBjdx_ion = dm.dPAiBjdxF(x, Prop.DeltaAB_ion, dDeltadx_ion, Prop.Xa_ion, dXadx_ion, Prop.PAiBj_ion)   
        dPijdx = dm.dPijdxF(Prop.Pij, Prop.PAiBj, dPAiBjdx)
        dPidx = dm.dPidxF(Prop.Pi, Prop.Xa, dXadx)
        dthetadx = dm.dthetaidxF(Prop.theta, Prop.PAiBj_ion, dPAiBjdx_ion, Prop.charge)
        dgidx = dm.dgidxF(Prop.Pi, Prop.Pij, dPijdx, dPidx, Prop.zij, Prop.dipolo, Prop.ang_theta, Prop.ang_gama)
        
        de00dx = dm.deps00dxF(Prop.e00, x, ro_metro, Prop)
        dEpsdx = dm.desprdxF(Prop.C0eps, Prop.C2eps, de00dx, Prop.RSP, Prop.e00, ro_metro, T, x, Prop.theta, Prop.gi, Prop.dipolo, dthetadx, dgidx)
        dpermdx = dEpsdx * Permvac
        Prop.de00dx = de00dx
        Prop.dgidx = dgidx
        Prop.dPAiBjdx = dPAiBjdx
        Prop.dPAiBjdx_ion = dPAiBjdx_ion
        Prop.dPijdx = dPijdx
        Prop.dthetadx = dthetadx
        Prop.dPidx = dPidx
    
        dKdx = dh.dkdebyedxF(Prop.Kd, x, Prop.charge, Prop.perm, dpermdx)
        dXjdk = dh.dXjdkF(Prop.Kd, a_metro, Prop.Xj)
        dXjdx = dh.dXjdxF(dXjdk, dKdx)       
        daiondx = dh.daiondxF(aion, Prop.Kd, dKdx, Prop.perm, dpermdx, T, Prop.charge, x, Prop.Xj, dXjdx)
        
        Prop.aion = aion
        Prop.dEpsrdx = dEpsdx
        Prop.dKdx = dKdx
        Prop.dXjdx = dXjdx
        Prop.daiondx = daiondx
        
        aborn = Born.abornF(x, T, Prop.charge, Prop.rborn, Prop.RSP)
        daborndx = Born.daborndxF(x, T, Prop, Prop.RSP, dEpsdx); Prop.daborndx = daborndx
        Prop.aborn = aborn
    
    
    ares = f.aresF(ahc, adisp, aassoc, aassoc_ion, aion, aborn); Prop.ares = ares

    daresdx = f.daresdxF(dahcdx, dadispdx, daassdx, daassdx_ion, daiondx, daborndx, nc); Prop.daresdx = daresdx

    ureskTV = f.ureskF(ares, Prop.z, daresdx, x, nc); Prop.ureskTV = ureskTV
    
    gres = ares + (Prop.z - 1) - log(Prop.z)
    
    Prop.gres = gres

    # u res transforming ures from T,V to T,P
    phi = f.phiF(ureskTV, Prop.z, nc)
    Prop.phi = phi
    

    return phi



    
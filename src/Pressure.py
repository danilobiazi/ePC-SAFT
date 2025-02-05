from Const import a, b, kb, Permvac
import PCSAFT as f
import numpy as np
import Association as fAs
import DebeyHuckel as dh
import RSP as dm
import Born


def Press(eta, x, Pspec, T, Prop):
    
    I1 = f.I1F(eta, a, Prop.MeanS); Prop.I1 = I1
    I2 = f.I2F(eta, b, Prop.MeanS); Prop.I2 = I2
    
    ro = f.roF(eta, x, Prop.m, Prop.di); Prop.ro = ro
    
    csi = f.csiF(ro, x, Prop.m, Prop.di); Prop.csi = csi
    C1 = f.C1F(Prop.MeanS, eta); Prop.C1 = C1
    ghsij = f.ghsijF(csi, Prop.dij); Prop.ghsij = ghsij
    C2 = f.C2F(Prop.MeanS, eta, C1); Prop.C2 = C2
    dI1 = f.dI1F(eta, a, Prop.MeanS); Prop.dI1 = dI1
    dI2 = f.dI2F(eta, b, Prop.MeanS); Prop.dI2 = dI2
    dghs = f.dghsF(csi, Prop.dij); Prop.dghs = dghs

    zdisp = f.zdispF(dI1, Prop.mesig, ro, Prop.MeanS, C1, dI2, C2, eta, I2, Prop.me2sig)
    Prop.zdisp = zdisp

    zhs = f.zhsF(csi); Prop.zhs = zhs

    zhc = f.zhcF(Prop.MeanS, zhs, x, Prop.m, ghsij, dghs); Prop.zhc = zhc

    zassoc = 0   
    flag_ion = 0    # flag_ion = 1 activates electrolyte terms
    zassoc_ion = 0
    
    
    if np.sum(Prop.S[np.where(Prop.charge == 0)]) > 0:
        # verifies if the non-charge components have associating sites
        
        
        eABij = fAs.eABijF(Prop.eAB, Prop.lambij, Prop.nc)
        Prop.eABij = eABij

        kABij = fAs.kABijF(Prop.kAB, Prop.sig, Prop.nc)
        Prop.kABij = kABij
        
        DeltaAB = fAs.DeltaABF(Prop.S, Prop.Signals, Prop.sigij, ghsij, kABij, eABij, T, Prop.nc, Prop.di, Prop.charge, flag_ion)
        Prop.DeltaAB = DeltaAB
        
        Xar = fAs.XConv(ro, x, Prop.S, DeltaAB, Prop.SiteType, Prop.nc)
        Xa = Xar[0]
        Xs = Xar[1]
        
        # message to degub convergence
        Prop.msgXa = 'Xa: ' + Xs[-1] + ' in ' + str(Xs[1]['nfev']) + ' iterations.'
        
        Prop.Xa = Xa

        Lambda = fAs.LambdaF(ro, x, Prop.S, DeltaAB, Xa, len(Prop.SiteType), Prop.nc)
        Lambda = np.linalg.inv(Lambda)
        Prop.Lambda = Lambda

        dDeltadro = fAs.dDeltadRoF(Prop.di, Prop.sigij, kABij, eABij, T, dghs,
                                   len(Prop.SiteType), DeltaAB, ro, Prop.nc)
        Prop.dDeltadro = dDeltadro

        PsiRo = fAs.PsiRoF(Xa, ro, x, Prop.S, len(Prop.SiteType), dDeltadro, Prop.nc)
        Prop.PsiRo = PsiRo

        dXadRo = fAs.dXadroF(Lambda, PsiRo, Prop.nc, len(Prop.SiteType))
        Prop.dXadRo = dXadRo

        zassoc = fAs.ZassocF(ro, Xa, x, Prop.S, dXadRo, Prop.nc, len(Prop.SiteType))
        Prop.zassoc = zassoc
        
    elif np.sum(Prop.S) == 0:
        Prop.checkeps = -1
        
    ro_metro = ro * 1e30  # converting ro to 1/m³
    a_metro = Prop.sig / 1e10  # converting sig to m
    dpermdro_metro = 0
    
    PAiBj = dm.PAiBjF(ro, x, DeltaAB, Xa, Prop.S); Prop.PAiBj = PAiBj
    Pij = dm.PijF(PAiBj); Prop.Pij = Pij
    Pi = dm.PiF(Xa); Prop.Pi = Pi
    gi = dm.giF(Pi, Pij, x, Prop.zij,Prop.dipolo, Prop.ang_gama, Prop.ang_theta); Prop.gi = gi
    e00 = dm.eps00(x, ro_metro, Prop.polarizab); Prop.e00 = e00
    thetai = np.ones(Prop.nc); Prop.theta = thetai
    dThetadro_metro = np.zeros(Prop.nc)

    if Prop.charge.any() != 0:

        # flag ion used in Delta function to disable solvent-solvent association in ion-solvent association term
        # and to disable ion-solvent association inside the "normal" association term
        flag_ion = 1
        
        eABij_ion = np.array(Prop.eABij)        
        kABij_ion = np.array(Prop.kABij)
        Prop.eABij_ion = eABij_ion
        Prop.kABij_ion = kABij_ion
        
        # matrix S: column: sites (H, O, cation, anion). Lines: components
        # For water-ion association, the water is considered to have only 1 donor and 1 acceptor site (2B scheme) 

        Prop.S_ion = Prop.S * 1
        Prop.S_ion[np.where(Prop.S[np.where(Prop.charge == 0)] !=0)] = 1


        DeltaAB_ion = fAs.DeltaABF(Prop.S_ion, Prop.Signals, Prop.sigij, ghsij, kABij_ion, eABij_ion, T, Prop.nc, Prop.di, Prop.charge, flag_ion)
        Prop.DeltaAB_ion = DeltaAB_ion
        
        Xar_ion = fAs.XConv(ro, x, Prop.S_ion, DeltaAB_ion, Prop.SiteType, Prop.nc)
        Xa_ion = Xar_ion[0]
        Xs_ion = Xar_ion[1]
        
        # message to degub Xa convergence
        Prop.msgXa_ion = 'Xa_ion: ' + Xs_ion[-1] + ' in ' + str(Xs_ion[1]['nfev']) + ' iterations.'
        
        Prop.Xa_ion = Xa_ion

        Lambda_ion = fAs.LambdaF(ro, x, Prop.S_ion, DeltaAB_ion, Xa_ion, len(Prop.SiteType), Prop.nc)
        Lambda_ion = np.linalg.inv(Lambda_ion)
        Prop.Lambda_ion = Lambda_ion

        dDeltadro_ion = fAs.dDeltadRoF(Prop.di, Prop.sigij, kABij_ion, eABij_ion, T, dghs,
                                   len(Prop.SiteType), DeltaAB_ion, ro, Prop.nc)
        Prop.dDeltadro_ion = dDeltadro_ion

        PsiRo_ion = fAs.PsiRoF(Xa_ion, ro, x, Prop.S_ion, len(Prop.SiteType), dDeltadro_ion, Prop.nc)
        Prop.PsiRo_ion = PsiRo_ion

        dXadRo_ion = fAs.dXadroF(Lambda_ion, PsiRo_ion, Prop.nc, len(Prop.SiteType))
        Prop.dXadRo_ion = dXadRo_ion

        zassoc_ion = fAs.ZassocF(ro, Xa_ion, x, Prop.S_ion, dXadRo_ion, Prop.nc, len(Prop.SiteType))
        Prop.zassoc_ion = zassoc_ion
        
        PAiBj_ion = dm.PAiBjF(ro, x, DeltaAB_ion, Xa_ion, Prop.S_ion); Prop.PAiBj_ion = PAiBj_ion
        thetai = dm.thetaiF(PAiBj_ion, Prop.charge); Prop.theta = thetai
        

        dPAiBjdro_metro_ion = dm.dPAiBjdro_metroF(PAiBj_ion, ro, x, DeltaAB_ion, dDeltadro_ion, Xa_ion, dXadRo_ion)
        Prop.dPAiBjdro_metro_ion = dPAiBjdro_metro_ion
        
        dThetadro_metro = dm.dthetaidro_metroF(thetai, PAiBj_ion, dPAiBjdro_metro_ion)
        
        Prop.dThetadro_metro = dThetadro_metro

        epsr = dm.EpsMariboF(e00, ro_metro, T, x, thetai, gi, Prop.dipolo)
        perm = epsr * Permvac; Prop.perm = perm; Prop.RSP = epsr


        dPAiBjdro_metro = dm.dPAiBjdro_metroF(PAiBj, ro, x, DeltaAB, dDeltadro, Xa, dXadRo)
        Prop.dPAiBjdro_metro = dPAiBjdro_metro
        dPijdro_metro = dm.dPijdro_metroF(Pij, PAiBj, dPAiBjdro_metro)
        Prop.dPijdro_metro = dPijdro_metro
        dPidro_metro = dm.dPidro_metroF(Pi, Xa, dXadRo)
        Prop.dPidro_metro = dPidro_metro
        dgidro_metro = dm.dgidro_metroF(Prop, Pi, Pij, dPijdro_metro, dPidro_metro, Prop.zij)
        Prop.dgidro_metro = dgidro_metro
        deps00dro_metro = dm.deps00dromF(e00, x, Prop)
        Prop.deps00dro_metro = deps00dro_metro
        
        res = dm.depsrdro_m(epsr, e00, deps00dro_metro, ro_metro, T, x, Prop.dipolo, gi, thetai, dgidro_metro, dThetadro_metro)
        depsrdro_metro = res[0]
        Prop.C0eps = res[1]  # will be used in depsrdx
        Prop.C2eps = res[2]  # will be used in depsrdx
        
        dpermdro_metro = Permvac * depsrdro_metro
        Prop.depsrdro_metro = depsrdro_metro
        Prop.dpermdro_metro = dpermdro_metro

    zion = 0
    zborn = 0
    if Prop.charge.any() != 0: #and np.sum(x[Prop.charge !=0]) > 1e-12:
        
        Kd = dh.kdebyeF(ro_metro, T, perm, x, Prop.charge)
        Xj = dh.XjF(Kd, a_metro)
        dkdro_metro = dh.dkdebyedro_metroF(Kd, ro_metro, perm, dpermdro_metro)
        aion = dh.aionF(Kd, T, perm, x, Prop.charge, Xj)
        dXjdk = dh.dXjdkF(Kd, a_metro, Xj)
        dxjdro_metro = dh.dXjdro_metroF(dXjdk, dkdro_metro)
        zion = dh.zion(ro_metro, aion, Kd, dkdro_metro, perm, dpermdro_metro, T, x, Prop.charge, dxjdro_metro)
        Prop.Kd = Kd
        Prop.Xj = Xj
        Prop.zion = zion   
        
        if Prop.checkborn == 1:
            Prop.msgBorn = 'Born term enabled'
            aborn = Born.abornF(x, T, Prop.charge, Prop.rborn, Prop.RSP); Prop.aborn = aborn
            daborndro_metro = Born.daborndroF(T, Prop.RSP, Prop.depsrdro_metro, x, Prop)
            zborn = ro_metro * daborndro_metro
            Prop.daborndro_metro = daborndro_metro
            Prop.zborn = zborn
            
    z = 1 + zhc + zdisp + zassoc + zassoc_ion + zion + zborn
    Prop.z = z

    Pcalc = z * kb * T * ro * 10 ** 30  # Pressure in Pa
    Prop.Pcalc = Pcalc

    return Pcalc - Pspec

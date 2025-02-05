import numpy as np


class Component:

    def __init__(self):
        self.checkborn = 1  # Born term enabled = 1
        self.name = []
        self.nc = 0
        self.MM = 0
        self.MW_mean = 0
        self.m = 0
        self.sig = 0
        self.eps = 0
        self.kAB = 0
        self.eAB = 0
        self.rborn = 0
        self.charge = 0
        self.dipolo = 0
        self.polarizab = 0
        self.ang_gama = 0
        self.ang_theta = 0
        self.RSP = 0
        self.SiteP1T = []
        self.SiteP2T = []
        self.SiteP3T = []
        self.SiteP4T = []
        self.SiteP5T = []
        self.SiteN1T = []
        self.SiteN2T = []
        self.SiteN3T = []
        self.SiteN4T = []
        self.SiteN5T = []
        self.SiteP1N = []
        self.SiteP2N = []
        self.SiteP3N = []
        self.SiteP4N = []
        self.SiteP5N = []
        self.SiteN1N = []
        self.SiteN2N = []
        self.SiteN3N = []
        self.SiteN4N = []
        self.SiteN5N = []
        self.SiteType = 0
        self.S = 0
        self.S_ion = 0
        self.Signals = 0
        self.di = 0
        self.dij = 0
        self.sigij = 0
        self.ekij = 0
        self.MeanS = 0
        self.mesig = 0
        self.me2sig = 0
        self.I1 = 0
        self.I2 = 0
        self.ro = 0
        self.csi = 0
        self.C1 = 0
        self.C2 = 0
        self.ghsij = 0
        self.dI1 = 0
        self.dI2 = 0
        self.dghs = 0
        self.zdisp = 0
        self.zhs = 0
        self.zhc = 0
        self.eABij = 0
        self.eABij_ion = 0
        self.kABij = 0
        self.kABij_ion = 0
        self.DeltaAB = 0
        self.Xa = 0
        self.Lambda = 0
        self.dDeltadro = 0
        self.PsiRo = 0
        self.dXadRo = 0
        self.zassoc = 0
        self.PAiBj = 0
        self.DeltaAB_ion = 0
        self.Xa_ion = 0
        self.Lambda_ion = 0
        self.dDeltadro_ion = 0
        self.PsiRo_ion = 0
        self.dXadRo_ion = 0
        self.zassoc_ion = 0
        self.PAiBj_ion = 0      
        self.Pij = 0
        self.Pi = 0
        self.theta = 0
        self.gi = 0
        self.e00 = 0
        self.perm = 0
        self.depsrdx = 0
        self.dPAiBjdro_metro = 0
        self.dPAiBjdro_metro_ion = 0
        self.dPijdro_metro = 0
        self.dPidro_metro = 0
        self.dThetadro_metro = 0
        self.dgidro_metro = 0
        self.deps00dro_metro = 0
        self.depsrdro_metro = 0
        self.Kd = 0
        self.Xj = 0
        self.zion = 0
        self.msgBorn = ''
        self.aborn = 0
        self.zborn = 0
        self.daborndro_metro = 0
        self.z = 0
        self.Pcalc = 0
        self.eta = 0
        self.msgP = ''
        self.MM_media = 0
        self.Density_Mol = 0
        self.Density_Mass = 0
        self.ahs = 0
        self.ahc = 0
        self.adisp = 0
        self.dadispdx = 0
        self.aassoc = 0
        self.aassoc_ion = 0
        self.dDeltadx = 0
        self.PsiX = 0
        self.dXadx = 0
        self.daassdx = 0
        self.aion = 0
        self.dEpsrdx = 0
        self.dKdx = 0
        self.dXjdx = 0
        self.daiondx = 0
        self.de00dx = 0
        self.dgidx = 0
        self.dPAiBjdx = 0
        self.dPijdx = 0
        self.dthetadx = 0
        self.dPidx = 0
        self.daborndx = 0
        self.ares = 0
        self.daresdx = 0
        self.ureskTV = 0
        self.gres = 0
        self.phi = 0
        self.kij = 0
        self.lij = 0
        self.lambij = 0
        self.zij = 0
        self.msgXa = ''
        self.msgXa_ion = ''
        self.csik = 0



def GenPropF(comp, listG, Prop):


    lista=np.zeros(0)
    for i in comp:
        lista = np.append(lista, listG[int(i)-1])

    nc = np.shape(comp)[0]
    
    nc1 = nc
    if nc == 1:
        nc1 = 2
    
    kij = np.zeros([nc1, nc1])
    lij = np.zeros([nc1, nc1])
    lambij = np.zeros([nc1, nc1])
    MM = np.zeros(nc)
    m = np.zeros(nc)
    sig = np.zeros(nc)
    eps = np.zeros(nc)
    kAB = np.zeros(nc)
    eAB = np.zeros(nc)
    charge = np.zeros(nc)
    dipolo = np.zeros(nc)
    polarizab = np.zeros(nc)
    ang_gama = np.zeros(nc)
    ang_theta = np.zeros(nc)
    rborn = np.zeros(nc)
    zij = np.zeros(nc)

    for p, j in enumerate(comp):
        Prop.name.append(lista[p]['name'])
        MM[p] = float(lista[p]['MM'])
        m[p] = float(lista[p]['mi'])
        sig[p] = float(lista[p]['sig'])
        eps[p] = float(lista[p]['epsilon'])
        kAB[p] = float(lista[p]['kAB'])
        eAB[p] = float(lista[p]['eAB'])
        charge[p] = int(lista[p]['charge'])
        if lista[p]['dipolo'] == '':
            lista[p]['dipolo'] = 0
        dipolo[p] = float(lista[p]['dipolo'])
        if lista[p]['polarizab'] == '':
            lista[p]['polarizab'] = 0
        polarizab[p] = float(lista[p]['polarizab'])
        if lista[p]['ang_gama'] == '':
            lista[p]['ang_gama'] = 0
        ang_gama[p] = float(lista[p]['ang_gama'])
        if lista[p]['ang_theta'] == '':
            lista[p]['ang_theta'] = 0
        ang_theta[p] = float(lista[p]['ang_theta'])
        
        if lista[p]['zij'] == '':
            lista[p]['zij'] = 0
        zij[p] = float(lista[p]['zij'])
        
        if lista[p]['raio_born'] == '':
            lista[p]['raio_born'] = 0
        rborn[p] = float(lista[p]['raio_born'])
        Prop.SiteP1T.append(lista[p]['P1T'])
        Prop.SiteP2T.append(lista[p]['P2T'])
        Prop.SiteP3T.append(lista[p]['P3T'])
        Prop.SiteP4T.append(lista[p]['P4T'])
        Prop.SiteP5T.append(lista[p]['P5T'])
        Prop.SiteN1T.append(lista[p]['N1T'])
        Prop.SiteN2T.append(lista[p]['N2T'])
        Prop.SiteN3T.append(lista[p]['N3T'])
        Prop.SiteN4T.append(lista[p]['N4T'])
        Prop.SiteN5T.append(lista[p]['N5T'])
        Prop.SiteP1N.append(lista[p]['P1N'])
        Prop.SiteP2N.append(lista[p]['P2N'])
        Prop.SiteP3N.append(lista[p]['P3N'])
        Prop.SiteP4N.append(lista[p]['P4N'])
        Prop.SiteP5N.append(lista[p]['P5N'])
        Prop.SiteN1N.append(lista[p]['N1N'])
        Prop.SiteN2N.append(lista[p]['N2N'])
        Prop.SiteN3N.append(lista[p]['N3N'])
        Prop.SiteN4N.append(lista[p]['N4N'])
        Prop.SiteN5N.append(lista[p]['N5N'])
    
    Prop.nc = nc
    Prop.kij = kij
    Prop.lij = lij
    Prop.lambij = lambij
    Prop.MM = MM
    Prop.m = m
    Prop.sig = sig
    Prop.eps = eps
    Prop.kAB = kAB
    Prop.eAB = eAB
    Prop.charge = charge
    Prop.rborn = rborn
    Prop.dipolo = dipolo
    Prop.polarizab = polarizab
    Prop.ang_gama = ang_gama
    Prop.ang_theta = ang_theta
    Prop.zij = zij
       
    
    auxT = np.array([Prop.SiteP1T, Prop.SiteP2T, Prop.SiteP3T, Prop.SiteP4T, Prop.SiteP5T,
           Prop.SiteN1T, Prop.SiteN2T, Prop.SiteN3T, Prop.SiteN4T, Prop.SiteN5T])
    
    auxN = np.array([Prop.SiteP1N, Prop.SiteP2N, Prop.SiteP3N, Prop.SiteP4N, Prop.SiteP5N,
           Prop.SiteN1N, Prop.SiteN2N, Prop.SiteN3N, Prop.SiteN4N, Prop.SiteN5N])


    # the following code determines the type of sites, its signals and organize them
    # generates 3 matrices:
    # SiteType: vector with the type of sites presented in the selected components Ex: ['H', 'O', 'N']
    # SiteSignals: vector with the signals of the vector SiteType. Ex: ['+', '-', '-']
    # S: matrix wich stores the number of each site type for each component
    # The S matrix determines the association scheme for each component


    #                        'H''O''N'
    # matrix S example:     [ 0  0  0 ] - component 1: no sites.
    #                       [ 2  2  0 ] - component 2: water 4C 2 'H' sites and 2 'O' sites
    #                       [ 1  0  1 ] - component 3: amine 2B


    # creating 2 auxiliary arrays SiteS e SiteT
    SiteS = np.transpose(np.array(auxN))
    SiteT = np.transpose(np.array(auxT))
    
    # Identifies site types presented ex. H, O, N
    rows, cols = np.nonzero(SiteT)
    
    SiteType = list(dict.fromkeys(SiteT[rows, cols]))
    Prop.SiteType = SiteType

    # Identifies if site is donor or acceptor (+ or  -)
    # At the moment, a component can have up to 5 donor and 5 acceptor diferent types of sites (SiteP1 to SiteP5) and (SiteN1 to SiteN5)


    SiteSignals = [''] * len(cols)

    for i in range(len(cols)):
        if cols[i] < 5:
            SiteSignals[i] = '+'
        else:
            SiteSignals[i] = '-'


    # Generating vector SiteSignals,  + or - (donor or acceptor)
    r, l = np.nonzero(SiteT)
    aux1 = SiteT[(r, l)]

    aux2 = [''] * len(SiteType)

    for i in range(len(SiteType)):
        for j in range(len(aux1)):
            if SiteType[i] == aux1[j]:
                aux2[i] = SiteSignals[j]
                break

    SiteSignals = aux2
    Prop.SiteSignals = SiteSignals

    # Generating S matrix: line = component - Column = number of sites from SiteType
    aux3 = np.zeros((len(comp), len(SiteType)))
    

    for i in range(len(comp)):
        for j in range(len(SiteType)):
            for k in range(10):
                if np.logical_and(SiteS[i][k] != 0, SiteT[i][k] == SiteType[j]):
                    aux3[i][j] = SiteS[i][k]

    Prop.S = aux3
    Prop.S_ion = aux3 * 0


    w, h = len(SiteType), len(comp)
    aux4 = [[0 for x in range(w)] for y in range(h)]

    for i in range(len(comp)):
        for j in range(len(SiteType)):
            if aux3[i, j] == 0:
                aux4[i][j] = '0'

            else:
                aux4[i][j] = SiteSignals[j]

    Prop.Signals = np.array(aux4)

    return Prop
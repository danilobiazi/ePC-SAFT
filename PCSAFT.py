from numpy import pi, exp, transpose, zeros, log, dot

# this module carries the PC-SAFT functions for hard chain and dispersive terms
# contains all first order derivatives functions with respect to density, molar fractions and temperature
# the derivatives with respect to temperature should be used with caution, because they were not yet used and can have errors.


def diF(sig, eps, T):
    di = sig * (1 - 0.12 * exp(-3 * eps / T))
    return di


def sigijF(sig, lij):

    sigij = zeros((sig.shape[0], sig.shape[0]))
    for i in range(sig.shape[0]):
        for j in range(sig.shape[0]):
            sigij[i, j] = (sig[j] + sig[i]) / 2 * (1 - lij[i, j])
    return sigij


def ekijF(eps, kij):

    ekij = zeros((eps.shape[0], eps.shape[0]))
    for i in range(eps.shape[0]):
        for j in range(eps.shape[0]):
            ekij[(i, j)] = (eps[i] * eps[j]) ** 0.5 * (1 - kij[i, j])
    return ekij


def meanSegF(x, m):

    mS = dot(x, transpose(m))
        
    return mS


def mesigF(x, m, ekij, sigij, T):
    soma = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            soma = soma + (x[i] * x[j] * m[i] * m[j] * ekij[i, j] / T *
                           sigij[i, j] ** 3)
    return soma


def me2sigF(x, m, ekij, sigij, T):
    soma = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            soma = soma + x[i] * x[j] * m[i] * m[j] * ((ekij[i, j] / T) **
                                                       2) * sigij[i, j] ** 3
    return soma


def I1F(eta, a, m):
    soma = 0
    for i in range(7):
        ai = a[i, 0] + (m - 1) / m * a[i, 1] + (m - 1) * (m - 2) / (m ** 2) * a[i, 2]
        soma = soma + ai * eta ** i
    return soma


def I2F(eta, b, m):
    soma = 0
    for i in range(7):
        bi = b[i, 0] + (m - 1) / m * b[i, 1] + (m - 1) * (m - 2) / (m ** 2) * b[i, 2]
        soma = soma + bi * eta ** i
    return soma


def roF(eta, x, m, d):
    soma = 0
    for i in range(x.shape[0]):
        soma = soma + x[i] * m[i] * d[i] ** 3
    ro = 6 / pi * eta / soma
    return ro


def csiF(ro, x, m, d):
    a = zeros(4)
    nc = len(x)
    for n in range(4):
        soma = 0
        for i in range(nc):
            soma += x[i] * m[i] * d[i] ** n
        a[n] = pi * ro / 6 * soma
    return a


def C1F(MeanS, eta):
    m = MeanS
    C1F = 1 / (1 + m * (8 * eta - 2 * eta ** 2) / (1 - eta) ** 4 + (1 - m) * (20 * eta - 27 * eta ** 2 + 12 * eta ** 3 - 2 * eta ** 4) / ((1 - eta) * (2 - eta)) ** 2)
    return C1F


def dijF(d):
    dij = zeros([d.shape[0], d.shape[0]])
    for i in range(d.shape[0]):
        for j in range(d.shape[0]):
            dij[i, j] = d[i] * d[j] / (d[i] + d[j])
    return dij


def ghsijF(csi, dij):
    c = csi
    ghs = zeros([dij.shape[0], dij.shape[0]])
    for i in range(dij.shape[0]):
        for j in range(dij.shape[0]):
            d = dij[i, j]
            e = d ** 2 * 2 * c[2] ** 2 / (1 - c[3]) ** 3
            ghs[i, j] = 1 / (1 - c[3]) + d * 3 * c[2] / (1 - c[3]) ** 2 + e
    return ghs


def C2F(MeanS, eta, C1):
    y = MeanS
    x = eta

    q = y * (-4 * x ** 2 + 20 * x + 8) / (1 - x) ** 5
    s = ((1 - x) * (2 - x)) ** 3
    R = (1 - y) * (2 * x ** 3 + 12 * x ** 2 - 48 * x + 40) / s

    C2F = -C1 ** 2 * (q + R)
    return C2F


def dI1F(eta, a, m):
    soma = 0
    for i in range(7):
        t = (m - 1) * (m - 2) / (m ** 2) * a[i, 2]
        ai = a[i, 0] + (m - 1) / m * a[i, 1] + t
        soma = soma + ai * (i + 1) * eta ** i
    return soma


def dI2F(eta, b, m):
    soma = 0
    for i in range(7):
        t = (m - 1) * (m - 2) / (m ** 2) * b[i, 2]
        bi = b[i, 0] + (m - 1) / m * b[i, 1] + t
        soma = soma + bi * (i + 1) * eta ** i
    return soma


def dghsF(csi, dij):
    c = csi
    d = dij
    r = c[3] / (1 - c[3]) ** 2
    s = 3 * c[2] / (1 - c[3]) ** 2 + 6 * c[2] * c[3] / (1 - c[3]) ** 3
    u = 4 * c[2] ** 2 / (1 - c[3]) ** 3
    t = u + 6 * c[2] ** 2 * c[3] / (1 - c[3]) ** 4

    dghs = zeros([dij.shape[0], dij.shape[0]])

    for i in range(dij.shape[0]):
        for j in range(dij.shape[0]):
            dghs[i, j] = r + d[i, j] * s + d[i, j] ** 2 * t

    return dghs


def zdispF(dI1, mesig, ro, MeanS, C1, dI2, C2, eta, I2, me2sig):
    a = pi * ro * MeanS * (C1 * dI2 + C2 * eta * I2) * me2sig
    zdisp = -2 * pi * ro * dI1 * mesig - a

    return zdisp


def zhsF(csi):
    c = csi
    a = c[3] / (1 - c[3])
    b = 3 * c[1] * c[2] / (c[0] * (1 - c[3]) ** 2)
    c = (3 * c[2] ** 3 - c[3] * c[2] ** 3) / (c[0] * (1 - c[3]) ** 3)
    zhs = a + b + c
    return zhs


def zhcF(MeanS, zhs, x, m, ghs, dghs):
    soma = 0
    for i in range(x.shape[0]):
        soma = soma + x[i] * (m[i] - 1) / ghs[i, i] * dghs[i, i]
    zhc = MeanS * zhs - soma
    return zhc


def ahsF(csi):
    x = csi
    a = 3 * x[1] * x[2] / (1 - x[3])
    b = x[2] ** 3 / (x[3] * (1 - x[3]) ** 2)
    c = (x[2] ** 3 / x[3] ** 2 - x[0]) * log(1 - x[3])
    ahs = 1 / x[0] * (a + b + c)
    return ahs


def ahcF(MeanS, ahs, x, m, ghs):
    soma = 0
    for i in range(x.shape[0]):
        soma = soma + x[i] * (m[i] - 1) * log(ghs[i, i])

    ahc = MeanS * ahs - soma
    return ahc


def adispF(I1, mesig, ro, MeanS, C1, I2, me2sig):
    adisp = -2 * pi * ro * I1 * mesig - pi * ro * MeanS * C1 * I2 * me2sig
    return adisp


def csikF(ro, mi, di):
    m = mi
    d = di

    csik = zeros([m.shape[0], 4])

    for k in range(m.shape[0]):
        for n in range(4):
            csik[k, n] = pi / 6 * ro * m[k] * d[k] ** n

    return csik


def dghsdxF(csik, csi, dij):
    d = dij
    c = csi
    b = csik

    dgdx = zeros([d.shape[0], d.shape[0], d.shape[0]])

    for k in range(d.shape[0]):
        for i in range(d.shape[0]):
            for j in range(d.shape[0]):
                a = b[k, 3] / (1 - c[3]) ** 2
                e = 3 * b[k, 2] / (1 - c[3]) ** 2
                f = 6 * c[2] * b[k, 3] / (1 - c[3]) ** 3
                g = 4 * c[2] * b[k, 2] / (1 - c[3]) ** 3
                h = 6 * c[2] ** 2 * b[k, 3] / (1 - c[3]) ** 4
                dgdx[k, i, j] = a + d[i, j] * (e + f) + d[i, j] ** 2 * (g + h)

    return dgdx


def dahsdxF(csik, ahs, csi, Nc):
    s = csik
    a = ahs
    i = csi

    dahs = zeros(Nc)

    for k in range(Nc):
        j = 1 - i[3]
        b = -s[k, 0] * a / i[0]
        c = 3 * (s[k, 1] * i[2] + i[1] * s[k, 2]) / j
        d = 3 * i[1] * i[2] * s[k, 3] / j ** 2
        e = 3 * i[2] ** 2 * s[k, 2] / (i[3] * j ** 2)
        f = i[2] ** 3 * s[k, 3] * (3 * i[3] - 1) / (i[3] ** 2 * j ** 3)
        m = 3 * i[2] ** 2 * s[k, 2] * i[3] - 2 * i[2] ** 3 * s[k, 3]
        g = (m / i[3] ** 3 - s[k, 0]) * log(j)
        h = (i[0] - i[2] ** 3 / i[3] ** 2) * s[k, 3] / j
        dahs[k] = b + (1 / i[0]) * (c + d + e + f + g + h)

    return dahs


def dahcdxF(m, ahs, MeanS, dahsdx, x, ghs, dghsdx, Nc):

    dahc = zeros(Nc)

    for k in range(Nc):
        soma = 0
        for i in range(Nc):
            soma = soma + x[i] * (m[i] - 1) / ghs[i, i] * dghsdx[k, i, i]

        a = m[k] * ahs + MeanS * dahsdx[k]
        dahc[k] = a - soma - (m[k] - 1) * log(ghs[k, k])

    return dahc


def I1kF(m, a, MeanS, csik, eta, Nc):

    c = csik

    I1k = zeros(Nc)

    for k in range(Nc):
        soma = 0
        ai = 0
        for i in range(7):
            t = (MeanS - 1) * (MeanS - 2) / (MeanS ** 2) * a[i, 2]
            ai = a[i, 0] + (MeanS - 1) / MeanS * a[i, 1] + t
            b = ai * i * c[k, 3] * eta ** (i - 1)
            d = a[i, 1] + (3 - 4 / MeanS) * a[i, 2]
            soma = soma + b + (m[k] / MeanS ** 2 * d) * eta ** i

        I1k[k] = soma

    return I1k


def I2kF(m, b, MeanS, csik, eta, Nc):

    c = csik

    I2k = zeros(Nc)

    for k in range(Nc):
        soma = 0
        bi = 0
        for i in range(7):
            t = (MeanS - 1) * (MeanS - 2) / (MeanS ** 2) * b[i, 2]
            bi = b[i, 0] + (MeanS - 1) / MeanS * b[i, 1] + t
            e = bi * i * c[k, 3] * eta ** (i - 1)
            d = b[i, 1] + (3 - 4 / MeanS) * b[i, 2]
            soma = soma + e + (m[k] / MeanS ** 2 * d) * eta ** i

        I2k[k] = soma

    return I2k


def C1kF(C1, C2, eta, csik, m, Nc):

    c = csik
    C1k = zeros(Nc)

    for k in range(Nc):
        b = C2 * c[k, 3]
        e = m[k] * (8 * eta - 2 * eta ** 2) / (1 - eta) ** 4
        f = 20 * eta - 27 * eta ** 2 + 12 * eta ** 3 - 2 * eta ** 4
        d = m[k] * f / ((1 - eta) * (2 - eta)) ** 2
        C1k[k] = b - C1 ** 2 * (e - d)

    return C1k


def mesigkF(m, x, ekij, T, sigij, Nc):
    e = ekij
    s = sigij

    mk = zeros(Nc)

    for k in range(Nc):
        soma = 0
        for j in range(Nc):
            soma = soma + x[j] * m[j] * e[k, j] / T * s[k, j] ** 3

        mk[k] = 2 * m[k] * soma

    return mk


def me2sigkF(m, x, ekij, T, sigij, Nc):
    e = ekij
    s = sigij

    m2k = zeros(Nc)

    for k in range(Nc):
        soma = 0
        for j in range(Nc):
            soma = soma + x[j] * m[j] * (e[k, j] / T) ** 2 * s[k, j] ** 3

        m2k[k] = 2 * m[k] * soma

    return m2k


def dadispdxF(ro, I1k, mesig, I1, mesigk, m, C1, I2, MeanS, C1k, I2k, me2sig, me2sigk, Nc):

    dadx = zeros(Nc)
    for k in range(Nc):
        a = -2 * pi * ro * (I1k[k] * mesig + I1 * mesigk[k])
        b = m[k] * C1 * I2 + MeanS * C1k[k] * I2 + MeanS * C1 * I2k[k]
        dadx[k] = a - pi * ro * (b * me2sig + MeanS * C1 * I2 * me2sigk[k])

    return dadx


def daresdxF(dahcdx, dadispdx, dassdx, daassdx_ion, daiondx, daborndx, nc):

    daresdx = zeros(nc)

    for k in range(nc):
        daresdx[k] = dahcdx[k] + dadispdx[k] + dassdx[k] + daassdx_ion[k] + daborndx[k] + daiondx[k]

    return daresdx


def aresF(ahc, adisp, aassoc, aassoc_ion, aion, aborn):
    ares = ahc + adisp + aassoc + aassoc_ion + aion + aborn
    return ares


def ureskF(ares, z, daresdx, x, Nc):

    uresk = zeros(Nc)

    soma = 0
    for j in range(Nc):
        soma = soma + x[j] * daresdx[j]

    for k in range(Nc):
        uresk[k] = ares + (z - 1) + daresdx[k] - soma

    return uresk


def phiF(uresk, z, Nc):

    phi = zeros(Nc)

    for k in range(Nc):
        phi[k] = exp(uresk[k] - log(z))

    return phi


def didTF(di, sigi, eki, T, Nc):

    diT = zeros(Nc)

    for i in range(Nc):
        diT[i] = (di[i] - sigi[i]) * 3 * eki[i] / (T ** 2)
        
    return diT


def dcsidTF(ro, x, m, di, didT, Nc):

    dcsi = zeros(4)
    Nc = len(x)

    for n in range(4):
        soma = 0
        for i in range(Nc):
            soma += x[i] * m[i] * (n) * di[i] ** (n-1) * didT[i]
        
        dcsi[n] = pi / 6 * ro * soma

    return dcsi

def dahsdTF(csi, dcsidT):

    c = csi
    d = dcsidT

    e = 1 - c[3]
    f = 3 * (d[1] * c[2] + c[1] * d[2]) / e
    g = 3 * c[1] * c[2] * d[3] / e ** 2
    h = 3 * c[2] ** 2 * d[2] / (c[3] * e ** 2)
    i = c[2] ** 3 * d[3] * (3 * c[3] - 1) / (c[3] ** 2 * e ** 3)
    j = (3 * c[2] ** 2 * d[2] * c[3] - 2 * c[2] ** 3 * d[3]) / c[3] ** 3 * log(e)
    k = (c[0] - c[2] ** 3 / c[3] ** 2) * d[3] / e

    return 1 / c[0] * (f + g + h + i + j + k)


def dahcdTF(MeanS, dahsT, x, m, ghs, dgdT, Nc):

    soma = 0
    for i in range(Nc):
        soma = soma + x[i] * (m[i] - 1) / ghs[i][i] * dgdT[i][i]

    return MeanS * dahsT - soma


def dI1dTF(a, MeanS, dcs3, eta):

    m = MeanS

    soma = 0
    for i in range(7):
        t = (m - 1) * (m - 2) / (m ** 2) * a[i][2]
        ai = a[i][0] + (m - 1) / m * a[i][1] + t
        soma = soma + ai * i * dcs3 * eta ** (i - 1)

    return soma


def dI2dTF(b, MeanS, dcs3, eta):

    m = MeanS

    soma = 0
    for i in range(7):
        t = (m - 1) * (m - 2) / (m ** 2) * b[i][2]
        bi = b[i][0] + (m - 1) / m * b[i][1] + t
    soma = soma + bi * i * dcs3 * eta ** (i - 1)

    return soma


def dadispdTF(ro, dI1, I1, T, mesig, MeanS, dcs3, C2, I2, C1, dI2, me2sig):

    a = (-2) * pi * ro * (dI1 - I1 / T) * mesig
    b = dcs3 * C2 * I2 + C1 * dI2 - 2 * C1 * I2 / T

    dadispdT = a - pi * ro * MeanS * b * me2sig

    return dadispdT


def dghsijdTF(cs2, cs3, dcs2, dcs3, di, didT, Nc):

    a = zeros((Nc, Nc))

    den = 1 - cs3

    g = 3 * cs2 / den ** 2
    h = 2 * cs2 ** 2 / den ** 3
    de = dcs3 / den ** 2
    dg = 6 * cs2 * dcs3 / den ** 3 + 3 * dcs2 / den ** 2
    dh = 6 * cs2 ** 2 / den ** 4 * dcs3 + 4 * cs2 / den ** 3 * dcs2
    for i in range(Nc):
        for j in range(Nc):
            f = di[i] * di[j] / (di[i] + di[j])
            k = di[i] + di[j]
            m = di[i] * di[j] / k ** 2 * (didT[i] + didT[j])
            df = di[i] / k * didT[j] + di[j] / k * didT[i] - m
            a[i][j] = de + f * dg + g * df + f ** 2 * dh + h * 2 * f * df

    return a


from numpy import pi, array


pi = pi
Nav = 6.022140857e23            # Avogadro number
kb = 1.380648465952442093e-23   # Boltzmann constant J/K or m².kg / s².K
Rgas = 8.3144621                # J/mol.K ou m³.Pa/mol.K
eCharge = 1.6021766208E-19        # elementar charge im Coulomb
Permvac = 8.854187817E-12       # Coulomb/Volt.m or C²/N.m² or C²/J.m


# universal constants for dispersive term
a = array([
    [0.9105631445, -0.3084016918, -0.0906148351],
    [0.6361281449, 0.1860531159, 0.4527842806],
    [2.6861347891, -2.5030047259, 0.5962700728],
    [-26.547362491, 21.419793629, -1.7241829131],
    [97.759208784, -65.255885330, -4.1302112531],
    [-159.59154087, 83.318680481, 13.776631870],
    [91.297774084, -33.746922930, -8.6728470368]
])

b = array([
    [0.7240946941, -0.57554980753, 0.09768831158],
    [2.2382791861, 0.69950955214, -0.25575749816],
    [-4.00258494846, 3.89256733895, -9.15585615297],
    [-21.0035768149, -17.2154716478, 20.6420759744],
    [26.8556413627, 192.672264465, -38.8044300521],
    [206.551338407, -161.826461649, 93.626774077],
    [-355.602356122, -165.207693456, -29.6669055852]
])


import csv
import PhiPhase as pf
import GenProp
import numpy as np


bank = []
with open('Components.csv', encoding='utf-8-sig') as comp:
    reader = csv.DictReader(comp, delimiter=';')
    for row in reader:
        bank.append(row)

# =============================================================================
# # uncomment to show components in bank (not so many at the moment)
# for i in bank:
#     print(i['number'], i['name'])
# =============================================================================


#define components number from loaded bank
    # 1 = water
    # 2 = Na+
    # 4 = Cl-

components = np.array([1, 2, 4])

# gets components parameters and create a general object in which every calculated property will be stored
u = GenProp.GenPropF(components, bank, GenProp.Component())


molality = 1 # define solution's molality


# converting from molality to molar fractions. Valid for 1:1 salts
MW_water = 18.015
x_salt = molality  / (molality + 1000 / MW_water)
x_water = 1 - x_salt
x_t = x_water + 2 * x_salt

x = np.array([x_water / x_t, x_salt / x_t, x_salt / x_t])


# Define T and P
T = 298.15  # Kelvin
P = 101325  # Pascal


# Define the phase in which fugacities will be calculated
phase = "L"


# Calculates fugacity coefficients
phi = pf.coeffugacidadeF(T, P, x, phase, u)


# Same calculations as above,  at infinite diluition

u0 = GenProp.GenPropF(components, bank, GenProp.Component())
xw0 = 0.999999999999
xi = (1-xw0)/2

fmol0 = np.array([xw0, xi, xi])

T = 298.15  # Kelvin
P = 101325  # Pascal

phase = "L"

phi00 = pf.coeffugacidadeF(T, P, fmol0, phase, u0)

IIAC_cation = phi[1]/phi00[1]
IIAC_anion = phi[2]/phi00[2]
MIAC_r = (IIAC_cation * IIAC_anion)**0.5  # valid for 1:1 salt
miac = (MIAC_r) / (1 + 0.001 * 2 * MW_water * molality)  # valid for 1:1 salt

print('IIAC cation = ', IIAC_cation)
print('IIAC anion = ', IIAC_anion)
print('MIAC rational = ', MIAC_r)  


print('MIAC molal = ', miac)
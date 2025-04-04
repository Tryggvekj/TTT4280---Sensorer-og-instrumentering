import numpy as np


muabo = np.genfromtxt("C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\muabo.txt", delimiter=",")
muabd = np.genfromtxt("C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 510 # Replace with wavelength in nanometres
blue_wavelength = 460 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth
delta = np.sqrt(1/(3*(musr+mua)*mua))
deltaBlood = np.sqrt(1/(3*(musr+mua_blood)*mua_blood))

#Print values for task 11.1 b)
def printMuValues():
    print("Mua = ", mua)
    print("Musr = ", musr)

#Calculate C for the different colors
def calculateC():
    constantC = np.sqrt(3*mua*(musr+mua))
    print("C = ", constantC)
    return constantC

#Calculate transmission for each color
def transmissionCalc():
    constantC = calculateC()
    depth = 300*10**(-6) #change with depth in m
    transmission = np.exp(-constantC*depth)*100
    print("Transmission for depth of ", depth, " gives a transmission of ", transmission)

def contrastCalc():
    tLow_blood = [82.6, 70.97, 60.46]
    tHigh = [15.2, 0.27, 0.0029]
    contrastVector = []
    for i in range(0, len(tLow_blood)):
        contrastValue = (np.abs(tHigh[i]-tLow_blood[i]))/tLow_blood[i]
        contrastVector.append(contrastValue)
    print(contrastVector)

#Skriv inn funksjon under for å kjøre
print(delta)
print(deltaBlood)
print(calculateC())
transmissionCalc()
contrastCalc()

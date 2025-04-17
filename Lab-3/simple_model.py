import numpy as np

# Load absorption data
muabo = np.genfromtxt("./muabo.txt", delimiter=",")  # Oxygenated blood
muabd = np.genfromtxt("./muabd.txt", delimiter=",")  # Deoxygenated blood

# Define wavelengths in nanometers
wavelengths = np.array([600, 510, 460])  # Red, Green, Blue

# Helper functions to interpolate absorption values
def mua_blood_oxy(wl):
    return np.interp(wl, muabo[:, 0], muabo[:, 1])

def mua_blood_deoxy(wl):
    return np.interp(wl, muabd[:, 0], muabd[:, 1])

# Blood properties
bvf = 0.01       # Blood volume fraction
oxy = 0.8        # Oxygenation level
mua_other = 25   # Background absorption (e.g., from collagen)

# Total absorption coefficient (μa), in 1/m
mua_blood = mua_blood_oxy(wavelengths) * oxy + mua_blood_deoxy(wavelengths) * (1 - oxy)
mua = mua_blood * bvf + mua_other

# Reduced scattering coefficient (μs'), in 1/m
# Based on Bashkatov et al., 2011
musr = 100 * (17.6 * (wavelengths / 500) ** -4 + 18.78 * (wavelengths / 500) ** -0.22)

# Penetration depths
delta = np.sqrt(1 / (3 * (musr + mua) + mua))
delta_blood = np.sqrt(1 / (3 * (musr + mua_blood) + mua_blood))

# Utility functions
def print_mu_values():
    print("μa (total absorption) =", mua)
    print("μs' (reduced scattering) =", musr)

def calculate_C():
    C = np.sqrt(3 * mua * (musr + mua))
    print("C =", C)
    return C

def transmission_calc():
    C = calculate_C()
    depth = 1.4e-2  # 1.4 cm typical vessel depth
    transmission = np.exp(-C * depth) * 100
    print(f"Transmission at {depth*100:.1f} mm depth: {transmission} %")

def contrast():
    # Transmission values for two different blood volume fractions (example data)
    t_low = np.array([82.6, 70.97, 60.46])  # Low BVF
    t_high = np.array([15.2, 0.27, 0.0029]) # High BVF

    contrast_values = np.abs(t_high - t_low) / t_low
    print("Contrast :", contrast_values)

# Run contrast calculation
contrast()

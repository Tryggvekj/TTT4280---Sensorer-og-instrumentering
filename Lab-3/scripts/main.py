import matplotlib.pyplot as plt
import numpy as np
import plot as p
import os

# Load data from file
# Liste over filstier
file_paths = [
    'C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\transmittans\\transmittansdata_kald_finger',
    'C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\reflektans\\reflektansdata_kald_finger',
    'C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\transmittans\\transmittansdata_med_lys',
    'C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\reflektans\\reflektansdata_med_lys',
    'C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\reflektans\\reflektansdata2',
    'C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\transmittans\\transmittansdata_erik'
]

theoretical_pulse = [
    65,
    64,
    68,
    71,
    67,
    71
]

data = np.loadtxt(file_paths[0])

# Ekstraher bare filnavnene
file_names = [os.path.basename(path) for path in file_paths]

# Extract columns and ignore the first two seconds of data
counter = 0
for dat in file_paths:
    data = np.loadtxt(dat)
    file_name = file_names[counter]
    record_time = 60
    start_time = 2
    stop_time = 60
    fps = 30
    index_start = start_time*len(data)//fps
    index_stop = stop_time*len(data)//fps

    r = data[index_start : index_stop, 0]
    g = data[index_start : index_stop, 1]
    b = data[index_start : index_stop, 2]

    #splitting the array into 5
    r_split = np.array_split(r, 5)
    g_split = np.array_split(g, 5)
    b_split = np.array_split(b, 5)

    # Variabler for å lagre målinger
    measurements = []  # Liste for målingene
    num = 1

    for i, k, j in zip(r_split, g_split, b_split):
        # Finn toppfrekvensene og pulsverdiene
        r_peak_value, g_peak_value, b_peak_value = p.find_peak_values(i, k, j, fps, file_name)

        # Legg til målingen i measurements-listen
        measurements.append((num, r_peak_value, g_peak_value, b_peak_value))

        # Plott dataene
        p.plot_rgb_and_fft(i, k, j, start_time, stop_time, num, fps, file_name, theoretical_pulse[counter])
        num += 1

    # Lagre målingene i LaTeX-tabellformat
    p.save_pulse_to_latex(file_name, measurements, theoretical_pulse[counter])

    counter += 1
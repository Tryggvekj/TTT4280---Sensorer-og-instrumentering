import matplotlib.pyplot as plt
import numpy as np
import plot2 as p2
import std_calc as sc

# Load data from file
#data = np.loadtxt('C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\reflektans\\data_walter')
data = np.loadtxt('C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\robusthetstest\\data_puls_med_lys')
#data = np.loadtxt('C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\robusthetstest\\data_puls_kms')
data = np.loadtxt('C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\ny_data\\data')
data = np.loadtxt('C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\transmittans\\data_erik')
data = np.loadtxt('C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\ny_data\\data2')

# Extract columns and ignore the first two seconds of data
record_time = 60
start_time = 2
stop_time = 60
fps = 30
index_start = start_time*len(data)//fps
index_stop = stop_time*len(data)//fps

r = data[index_start : index_stop, 0]
g = data[index_start : index_stop, 1]
b = data[index_start : index_stop, 2]

# Remove DC-offset by subtracting the mean
#r -= np.mean(r)
#g -= np.mean(g)
#b -= np.mean(b)

# Create a time axis of 28 seconds (after ignoring the first 2 seconds)

#splitting the array into 5
r_split = np.array_split(r, 5)
g_split = np.array_split(g, 5)
b_split = np.array_split(b, 5)

pulse_b = []
pulse_r = []
pulse_g = []   
std_rgb = []

snr_b = []
snr_r = []
snr_g = []
snr_rgb = []

for i, k, j in zip(r_split, g_split, b_split):
    r_peak_index, g_peak_index, b_peak_index = sc.find_peak_indices(i, k, j, start_time, stop_time, fps)
    pulse_b.append(b_peak_index)
    pulse_r.append(r_peak_index)
    pulse_g.append(g_peak_index)
    r_sn, g_sn, b_sn = sc.calculate_snr(i, k, j, start_time, stop_time, fps)
    snr_b.append(b_sn)
    snr_r.append(r_sn)
    snr_g.append(g_sn)

    #p2.plot_rgb_and_fft(i, k, j, start_time, stop_time, fps)

std_rgb.append((np.std(pulse_r), np.mean(pulse_r)))
std_rgb.append((np.std(pulse_g), np.mean(pulse_g)))
std_rgb.append((np.std(pulse_b), np.mean(pulse_b)))

for i in std_rgb:
    print(i[0], i[1])
    
for i in range(5):
    print()
    print(snr_r[i], snr_g[i], snr_b[i])

for i in std_rgb:
    print(i[1]*60)
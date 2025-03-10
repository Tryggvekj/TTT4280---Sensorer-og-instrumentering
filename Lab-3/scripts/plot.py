import matplotlib.pyplot as plt
import numpy as np

# Load data from file
data = np.loadtxt('C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\transmitans\\data')

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
time = np.linspace(start_time, stop_time, len(r))

# Create a figure for all time-domain plots
plt.figure(figsize=(15, 10))

# Plot for the red component
plt.subplot(3, 1, 1)
plt.plot(time, r, label='Red component')
plt.xlabel('Time (seconds)')
plt.ylabel('Red Component Values')
plt.title('Zoomed Plot of Red Component against Time')
plt.legend()
plt.grid(True)

# Plot for the green component
plt.subplot(3, 1, 2)
plt.plot(time, g, label='Green component')
plt.xlabel('Time (seconds)')
plt.ylabel('Green Component Values')
plt.title('Zoomed Plot of Green Component against Time')
plt.legend()
plt.grid(True)

# Plot for the blue component
plt.subplot(3, 1, 3)
plt.plot(time, b, label='Blue component')
plt.xlabel('Time (seconds)')
plt.ylabel('Blue Component Values')
plt.title('Zoomed Plot of Blue Component against Time')
plt.legend()
plt.grid(True)


# Compute FFT for each component
r_fft = np.fft.fft(r)
g_fft = np.fft.fft(g)
b_fft = np.fft.fft(b)
freq = np.fft.fftfreq(len(r), 1/fps)

# Filter FFT to show only from 0.5Hz to 4Hz
mask = (freq >= 0.5) & (freq <= 4)

# Create a figure for all FFT plots
plt.figure(figsize=(15, 10))

# Plot FFT for the red component
plt.subplot(3, 1, 1)
plt.plot(freq[mask], np.abs(r_fft)[mask], label='FFT of Red component')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Red Component (0.5Hz to 4Hz)')
plt.xlim([0.5, 4])
plt.legend()
plt.grid(True)

# Plot FFT for the green component
plt.subplot(3, 1, 2)
plt.plot(freq[mask], np.abs(g_fft)[mask], label='FFT of Green component')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Green Component (0.5Hz to 4Hz)')
plt.xlim([0.5, 4])
plt.legend()
plt.grid(True)

# Plot FFT for the blue component
plt.subplot(3, 1, 3)
plt.plot(freq[mask], np.abs(b_fft)[mask], label='FFT of Blue component')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('FFT of Blue Component (0.5Hz to 4Hz)')
plt.xlim([0.5, 4])
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


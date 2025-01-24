import pandas as pd
import matplotlib.pyplot as plt

# Read the sampling frequency from the first line
with open('volts.txt', 'r') as file:
    time_interval = float(file.readline().strip())

# Read the data from volts.txt, skipping the first line
data = pd.read_csv('volts.txt', skiprows=1)

# Extract the first column (ADC1)
adc1 = data['ADC1']

# Calculate the time axis
time = [i * time_interval for i in range(len(adc1))]

# Plot every 100th data point to reduce clutter
plt.plot(time, adc1)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Voltage Readings')
plt.grid(True)

# Set x-axis limits to zoom in between 0 and 0.05 seconds
plt.xlim(0, 0.005)

# Save the plot to a file
plt.savefig('Sinus_signal_zoomed.png')
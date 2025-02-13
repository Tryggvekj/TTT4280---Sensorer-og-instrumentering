import numpy as np
import scipy
import matplotlib.pyplot as plt

dt = 0.0001 # delay til signalene

fs = 10000 
t = np.linspace(0, 1, fs)  

signal1 = np.sin(2 * np.pi * 10 * t) 
signal2 = np.sin(2 * np.pi * 10 * (t - dt))  

correlation = scipy.signal.correlate(signal1, signal2)
delay = (np.argmax(np.abs(correlation)) - len(signal1) + 1 )/ fs 
print(delay, "sekunder")

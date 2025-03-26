import matplotlib.pyplot as plt
import numpy as np

def plot_rgb_and_fft(r, g, b, start_time, stop_time, fps=30):
    # Create a time axis
    time = np.linspace(0, (stop_time - start_time) / 5, len(r))

    # Create a figure for all time-domain plots
    plt.figure(figsize=(12, 5))

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

    plt.tight_layout()
    plt.show()

    # Compute FFT for each component
    n_fft = 2**20  # Length of FFT
    r_fft = np.fft.fft(r - np.mean(r), n=n_fft)
    g_fft = np.fft.fft(g - np.mean(g), n=n_fft)
    b_fft = np.fft.fft(b - np.mean(b), n=n_fft)
    freq = np.fft.fftfreq(n_fft, 1/fps)  # Use n_fft for frequency axis

    min = 0.01
    max = 4

    # Filter FFT to show only from 0.5Hz to 4Hz
    mask = (freq >= min) & (freq <= max)

    # Create a figure for all FFT plots
    plt.figure(figsize=(12, 5))

    # Plot FFT for the red component
    plt.subplot(3, 1, 1)
    plt.plot(freq[mask], np.abs(r_fft)[mask], label='FFT of Red component')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of Red Component (0.5Hz to 4Hz)')
    plt.xlim([min, max])
    plt.legend()
    plt.grid(True)

    # Plot FFT for the green component
    plt.subplot(3, 1, 2)
    plt.plot(freq[mask], np.abs(g_fft)[mask], label='FFT of Green component')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of Green Component (0.5Hz to 4Hz)')
    plt.xlim([min, max])
    plt.legend()
    plt.grid(True)

    # Plot FFT for the blue component
    plt.subplot(3, 1, 3)
    plt.plot(freq[mask], np.abs(b_fft)[mask], label='FFT of Blue component')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of Blue Component (0.5Hz to 4Hz)')
    plt.xlim([min, max])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    r_spectral = np.abs(r_fft)[mask]**2
    g_spectral = np.abs(g_fft)[mask]**2
    b_spectral = np.abs(b_fft)[mask]**2

    plt.figure(figsize=(12, 5))

    # Plot spectral density for the red component
    plt.subplot(3, 1, 1)
    plt.plot(freq[mask], r_spectral, label='Spectral Density of Red component')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Spectral Density of Red Component (0.5Hz to 4Hz)')
    plt.xlim([min, max])
    plt.legend()
    plt.grid(True)

    # Plot spectral density for the green component
    plt.subplot(3, 1, 2)
    plt.plot(freq[mask], g_spectral, label='Spectral Density of Green component')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Spectral Density of Green Component (0.5Hz to 4Hz)')
    plt.xlim([min, max])
    plt.legend()
    plt.grid(True)
    
    # Plot spectral density for the blue component
    plt.subplot(3, 1, 3)
    plt.plot(freq[mask], b_spectral, label='Spectral Density of Blue component')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Spectral Density of Blue Component (0.5Hz to 4Hz)')
    plt.xlim([min, max])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

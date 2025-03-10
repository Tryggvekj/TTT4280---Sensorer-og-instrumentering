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
    r_fft = np.fft.fft(r)
    g_fft = np.fft.fft(g)
    b_fft = np.fft.fft(b)
    freq = np.fft.fftfreq(len(r), 1/fps)

    # Filter FFT to show only from 0.5Hz to 4Hz
    mask = (freq >= 0.5) & (freq <= 4)

    # Create a figure for all FFT plots
    plt.figure(figsize=(12, 5))

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
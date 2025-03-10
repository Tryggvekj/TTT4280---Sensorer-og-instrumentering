import numpy as np

def find_peak_indices(r, g, b, start_time, stop_time, fps=30):
    # Compute FFT for each component
    r_fft = np.fft.fft(r)
    g_fft = np.fft.fft(g)
    b_fft = np.fft.fft(b)
    freq = np.fft.fftfreq(len(r), 1/fps)

    # Filter FFT to show only from 0.5Hz to 4Hz
    mask = (freq >= 0.5) & (freq <= 4)

    # Find the index of the highest value in the specified frequency range
    r_peak_index = np.argmax(np.abs(r_fft)[mask])
    g_peak_index = np.argmax(np.abs(g_fft)[mask])
    b_peak_index = np.argmax(np.abs(b_fft)[mask])

    # Convert masked indices to original indices
    r_peak_index = np.where(mask)[0][r_peak_index]
    g_peak_index = np.where(mask)[0][g_peak_index]
    b_peak_index = np.where(mask)[0][b_peak_index]

    return freq[r_peak_index], freq[g_peak_index], freq[b_peak_index]

def calculate_snr(r, g, b, start_time, stop_time, fps=30):
    # Compute FFT for each component
    r_fft = np.fft.fft(r)
    g_fft = np.fft.fft(g)
    b_fft = np.fft.fft(b)
    freq = np.fft.fftfreq(len(r), 1/fps)

    # Filter FFT to show only from 0.5Hz to 4Hz
    mask = (freq >= 0.5) & (freq <= 4)

    # Find the index of the highest value in the specified frequency range
    r_peak_index = np.argmax(np.abs(r_fft)[mask])
    g_peak_index = np.argmax(np.abs(g_fft)[mask])
    b_peak_index = np.argmax(np.abs(b_fft)[mask])

    # Calculate dB values
    r_db = 20 * np.log10(np.abs(r_fft)[mask])
    g_db = 20 * np.log10(np.abs(g_fft)[mask])
    b_db = 20 * np.log10(np.abs(b_fft)[mask])

    # Calculate SNR
    r_snr = r_db[r_peak_index] - np.mean(r_db)
    g_snr = g_db[g_peak_index] - np.mean(g_db)
    b_snr = b_db[b_peak_index] - np.mean(b_db)

    return r_snr, g_snr, b_snr


if __name__ == "__main__":
    # Example usage
    data = np.loadtxt('C:\\Users\\trygg\\OneDrive - NTNU\\6. semester\\TTT4280 - Sensorer og instrumentering\\Lab\\Github\\Lab-3\\transmitans\\data_erik')
    record_time = 60
    start_time = 2
    stop_time = 60
    fps = 30
    index_start = start_time * len(data) // fps
    index_stop = stop_time * len(data) // fps

    r = data[index_start : index_stop, 0]
    g = data[index_start : index_stop, 1]
    b = data[index_start : index_stop, 2]

    r_peak_index, g_peak_index, b_peak_index = find_peak_indices(r, g, b, start_time, stop_time, fps)

    print(f"Peak index for Red component: {r_peak_index}")
    print(f"Peak index for Green component: {g_peak_index}")
    print(f"Peak index for Blue component: {b_peak_index}")
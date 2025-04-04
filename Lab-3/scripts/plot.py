import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import os

def bandpass_filter(data, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def save_pulse_to_latex(file_name, measurements, theoretical_pulse):
    """
    Lagre målt puls i en tekstfil i LaTeX-tabellformat for flere målinger.

    Parameters:
        file_name (str): Navnet på filen uten utvidelse.
        measurements (list of tuples): Liste med målinger, der hver måling er en tuple (måling_nr, r_bpm, g_bpm, b_bpm).
        theoretical_pulse (float): Teoretisk verdi for puls (BPM).
    """
    # Opprett filnavn for tekstfilen
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    latex_file = os.path.join(output_dir, f"{file_name}_puls.tex")

    # Escape `_` i LaTeX
    latex_safe_file_name = file_name.replace("_", r"\_")

    # Start LaTeX-tabellen
    latex_content = r"""
\begin{table}[H]
\centering
\caption{Målt puls for fil: %s}
\label{tab:%s}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Måling} & \textbf{Rød (BPM)} & \textbf{Grønn (BPM)} & \textbf{Blå (BPM)} \\ \hline
""" % (latex_safe_file_name, latex_safe_file_name)

    # Legg til målinger i tabellen
    for i, (measurement_nr, r_bpm, g_bpm, b_bpm) in enumerate(measurements, start=1):
        latex_content += f"{measurement_nr} & {r_bpm:.1f} & {g_bpm:.1f} & {b_bpm:.1f} \\\\ \\hline\n"

        # Hvis vi har skrevet 5 målinger, avslutt tabellen og start en ny
        if i % 5 == 0 and i != len(measurements):
            part = i // 5
            latex_content += r"""\end{tabular}
\caption{Målt puls for fil: %s (del %d)}
\label{tab:%s_part%d}
\end{table}

\begin{table}[H]
\centering
\caption{Målt puls for fil: %s (forts.)}
\label{tab:%s_part%d}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Måling} & \textbf{Rød [BPM]} & \textbf{Grønn [BPM]} & \textbf{Blå [BPM]} \\ \hline
""" % (latex_safe_file_name, part, latex_safe_file_name, part,
       latex_safe_file_name, latex_safe_file_name, part + 1)

    # Avslutt siste målingstabell riktig før oppsummeringen
    latex_content += r"""\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Oppsummering av pulsverdier for fil: %s}
\label{tab:%s_summary}
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Kategori} & \textbf{Rød [BPM]} & \textbf{Grønn [BPM]} & \textbf{Blå [BPM]} \\ \hline
""" % (latex_safe_file_name, latex_safe_file_name)

    # Beregn statistikk
    r_values = [r_bpm for _, r_bpm, _, _ in measurements]
    g_values = [g_bpm for _, _, g_bpm, _ in measurements]
    b_values = [b_bpm for _, _, _, b_bpm in measurements]

    r_mean = np.mean(r_values)
    g_mean = np.mean(g_values)
    b_mean = np.mean(b_values)

    r_std = np.std(r_values)
    g_std = np.std(g_values)
    b_std = np.std(b_values)

    r_diff_mean = np.mean([abs(r - theoretical_pulse) for r in r_values])
    g_diff_mean = np.mean([abs(g - theoretical_pulse) for g in g_values])
    b_diff_mean = np.mean([abs(b - theoretical_pulse) for b in b_values])

    # Legg inn rader i oppsummeringstabellen
    latex_content += (
        f"Teoretisk verdi & {theoretical_pulse:.1f} & {theoretical_pulse:.1f} & {theoretical_pulse:.1f} \\\\ \\hline\n"
        f"Gjennomsnitt & {r_mean:.1f} & {g_mean:.1f} & {b_mean:.1f} \\\\ \\hline\n"
        f"Gjennomsnittlig avvik & {r_diff_mean:.1f} & {g_diff_mean:.1f} & {b_diff_mean:.1f} \\\\ \\hline\n"
        f"Standardavvik & {r_std:.1f} & {g_std:.1f} & {b_std:.1f} \\\\ \\hline\n"
        r"\end{tabular}" "\n" r"\end{table}"
    )

    # Skriv til fil
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"LaTeX-tabell lagret i: {latex_file}")

def find_peak_values(rs, gs, bs, fps, file_name):
    # Compute FFT for each component
    min = 0.5
    max = 4
    r = bandpass_filter(rs, min, max, fps, 10)
    g = bandpass_filter(gs, min, max, fps, 10)
    b = bandpass_filter(bs, min, max, fps, 10)

    # Compute FFT for each component
    n_fft = 2**20  # Length of FFT
    r_fft = np.fft.fft(r, n=n_fft)
    g_fft = np.fft.fft(g, n=n_fft)
    b_fft = np.fft.fft(b, n=n_fft)
    freq = np.fft.fftfreq(n_fft, 1/fps)  # Use n_fft for frequency axis

    # Filter FFT to show only from 0.5Hz to 4Hz
    mask = (freq >= min) & (freq <= max)

    # Find the peak frequency and convert to BPM
    r_peak_idx = np.argmax(np.abs(r_fft)[mask])
    g_peak_idx = np.argmax(np.abs(g_fft)[mask])
    b_peak_idx = np.argmax(np.abs(b_fft)[mask])

    r_peak_freq = freq[mask][r_peak_idx]
    g_peak_freq = freq[mask][g_peak_idx]
    b_peak_freq = freq[mask][b_peak_idx]

    r_bpm = r_peak_freq * 60
    g_bpm = g_peak_freq * 60
    b_bpm = b_peak_freq * 60

    return r_bpm, g_bpm, b_bpm


def plot_rgb_and_fft(rs, gs, bs, start_time, stop_time, number, fps, file_name):

    min = 0.5
    max = 4
    scale = 1
    r = bandpass_filter(rs, min, max, fps, 10)
    g = bandpass_filter(gs, min, max, fps, 10)
    b = bandpass_filter(bs, min, max, fps, 10)

    # Lagre plott i en mappe
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Create a time axis
    time = np.linspace(0, (stop_time - start_time) / 5, len(r))

    # Create a figure for all time-domain plots
    plt.figure(figsize=(12, 5))

    # Plot for the red component
    plt.subplot(3, 1, 1)
    plt.plot(time, rs, color = "r")
    # plt.xlabel('Tid [s]')
    # plt.ylabel('Amplitude')
    plt.title('Rød komponent')
    plt.grid(True)

    # Plot for the green component
    plt.subplot(3, 1, 2)
    plt.plot(time, gs, color = "g")
    # plt.xlabel('Tid [s]')
    plt.ylabel('Amplitude')
    plt.title(f'Grønn komponent')
    plt.grid(True)

    # Plot for the blue component
    plt.subplot(3, 1, 3)
    plt.plot(time, bs, color = "b")
    plt.xlabel('Tid [s]')
    # plt.ylabel('Amplitude')
    plt.title('Blå komponent')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_tidsplot_{number}.png"))
    plt.close()

    # Create a figure for all time-domain plots
    plt.figure(figsize=(12, 5))

    # Plot for the red component
    plt.subplot(3, 1, 1)
    plt.plot(time, r, color='red')
    # plt.xlabel('Tid [s]')
    # plt.ylabel('Amplitude')
    plt.title('Rød komponent')
    plt.grid(True)

    # Plot for the green component
    plt.subplot(3, 1, 2)
    plt.plot(time, g, color='green')
    # plt.xlabel('Tid [s]')
    plt.ylabel('Amplitude')
    plt.title('Grønn komponent')
    plt.grid(True)

    # Plot for the blue component
    plt.subplot(3, 1, 3)
    plt.plot(time, b, color='blue')
    plt.xlabel('Tid [s]')
    # plt.ylabel('Amplitude')
    plt.title('Blå komponent')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_båndpassfiltrert tidsplott_{number}.png"))
    plt.close()

    # Apply Hann window
    hann_window = np.hanning(len(r))
    r = r * hann_window
    g = g * hann_window
    b = b * hann_window

    # Compute FFT for each component
    n_fft = 2**16  # Length of FFT
    r_fft = np.fft.fft(r, n=n_fft)
    g_fft = np.fft.fft(g, n=n_fft)
    b_fft = np.fft.fft(b, n=n_fft)
    freq = np.fft.fftfreq(n_fft, 1/fps)  # Use n_fft for frequency axis

    # Filter FFT to show only from 0.5Hz to 4Hz
    mask = (freq >= min) & (freq <= max)

    # Find the peak frequency and convert to BPM
    r_peak_idx = np.argmax(np.abs(r_fft)[mask])
    g_peak_idx = np.argmax(np.abs(g_fft)[mask])
    b_peak_idx = np.argmax(np.abs(b_fft)[mask])

    r_peak_freq = freq[mask][r_peak_idx]
    g_peak_freq = freq[mask][g_peak_idx]
    b_peak_freq = freq[mask][b_peak_idx]

    r_bpm = r_peak_freq * 60
    g_bpm = g_peak_freq * 60
    b_bpm = b_peak_freq * 60

    # Create a figure for all FFT plots
    plt.figure(figsize=(12, 5))

    # Plot FFT for the red component
    plt.subplot(3, 1, 1)
    plt.plot(freq[mask]*scale, np.abs(r_fft)[mask])
    plt.axvline(r_peak_freq, color='r', linestyle='--', label=f'Målt puls: {r_bpm:.1f} BPM')
    # if scale == 60:
    #     plt.xlabel('Slag per minutt [BPM]')
    # else:
    #     plt.xlabel('Frekvens [Hz]')
    # plt.ylabel('Amplitude')
    plt.title('FFT av rød komponent')
    plt.xlim([min*scale, max*scale])
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot FFT for the green component
    plt.subplot(3, 1, 2)
    plt.plot(freq[mask]*scale, np.abs(g_fft)[mask])
    plt.axvline(g_peak_freq, color='g', linestyle='--', label=f'Målt puls: {g_bpm:.1f} BPM')
    # if scale == 60:
    #     plt.xlabel('Slag per minutt [BPM]')
    # else:
    #     plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Amplitude')
    plt.title('FFT av grønn komponent')
    plt.xlim([min*scale, max*scale])
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot FFT for the blue component
    plt.subplot(3, 1, 3)
    plt.plot(freq[mask]*scale, np.abs(b_fft)[mask])
    plt.axvline(b_peak_freq, color='b', linestyle='--', label=f'Målt puls: {b_bpm:.1f} BPM')
    if scale == 60:
        plt.xlabel('Slag per minutt [BPM]')
    else:
        plt.xlabel('Frekvens [Hz]')
    # plt.ylabel('Amplitude')
    plt.title('FFT av blå komponent')
    plt.xlim([min*scale, max*scale])
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_fft_{number}.png"))
    plt.close()

    r_spectral = np.abs(r_fft)[mask]**2
    g_spectral = np.abs(g_fft)[mask]**2
    b_spectral = np.abs(b_fft)[mask]**2

    plt.figure(figsize=(12, 5))

    # Plot spectral density for the red component
    log_scale = 10*np.log10(r_spectral/np.max(r_spectral))
    log_mean = np.mean(log_scale)
    plt.subplot(3, 1, 1)
    plt.plot(freq[mask], log_scale)
    plt.axhline(log_mean, color='r', linestyle='--', label=f'Støygulv ved {log_mean:.1f} dB')
    # plt.xlabel('Frekvens [Hz]')
    # plt.ylabel('Normalisert effekt [dB]')
    plt.title('Effektspekter av rød komponent')
    plt.xlim([min, max])
    plt.legend(loc='upper right')
    plt.grid(True)

    # Plot spectral density for the green component
    log_scale = 10*np.log10(g_spectral/np.max(g_spectral))
    log_mean = np.mean(log_scale)
    plt.subplot(3, 1, 2)
    plt.plot(freq[mask], log_scale)
    plt.axhline(log_mean, color='g', linestyle='--', label=f'Støygulv ved {log_mean:.1f} dB')
    # plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Normalisert effekt [dB]')
    plt.title('Effektspekter av grønn komponent')
    plt.legend(loc='upper right')
    plt.xlim([min, max])
    plt.grid(True)
    
    # Plot spectral density for the blue component
    log_scale = 10*np.log10(b_spectral/np.max(b_spectral))
    log_mean = np.mean(log_scale)
    plt.subplot(3, 1, 3)
    plt.plot(freq[mask], log_scale)
    plt.axhline(log_mean, color='b', linestyle='--', label=f'Støygulv ved {log_mean:.1f} dB')
    plt.xlabel('Frekvens [Hz]')
    # plt.ylabel('Normalisert effekt [dB]')
    plt.title('Effektspekter av blå komponent')
    plt.xlim([min, max])
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_name}_spectral_{number}.png"))
    plt.close()
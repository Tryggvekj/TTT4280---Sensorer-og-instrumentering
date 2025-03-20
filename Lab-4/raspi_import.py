import numpy as np
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def raspi_import(path: str, channels: int =5) -> None:
    """
    Import data produced using adc_sampler.c.

    Returns sample period and a (`samples`, `channels`) `float64` array of
    sampled data from all `channels` channels.

    Example (requires a recording named `foo.bin`):
    ```
    >>> from raspi_import import raspi_import
    >>> sample_period, data = raspi_import('foo.bin')
    >>> print(data.shape)
    (31250, 5)
    >>> print(sample_period)
    3.2e-05

    ```
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data

def compile(output: str) -> None:
    print("Compiling adc_sampler.c...")
    subprocess.run(['gcc', 'adc_sampler.c', '-lpigpio', '-lpthread', '-lm', '-o', 'adc_sampler'])

def run_sampler(output: str) -> None:
    print("Running adc_sampler...")
    subprocess.run(['sudo', './' + 'adc_sampler', '31250'])

def write_data(sample_period: str, data: str, output: str) -> None:
    with open(output, 'w') as f:
        f.write(str(sample_period) + '\n')
        f.write("ADC1,ADC2,ADC3,ADC4,ADC5\n")
        for row in data:
            f.write(','.join([str(x) for x in row]) + '\n')

def write_volts(sample_period: str, data: str, output: str) -> None:
    with open(output, 'w') as f:
        f.write(str(sample_period) + '\n')
        f.write("ADC1,ADC2,ADC3,ADC4,ADC5\n")
        for row in data:
            f.write(','.join([str(x) for x in row * 3.3 / 4095]) + '\n')

def convert(sample_period: str, data: str, output: str) -> None:
    print("Converting data...")

    write_data(sample_period, data, output + "data.txt")
    write_volts(sample_period, data, output + "volts.txt")

def get_adc_data(ADC: str, file_path: str):
    # Read the file using pandas
    df = pd.read_csv(file_path, skiprows=1)  # Skip the first line (sample period)

    # Check if the specified ADC column exists
    if ADC not in df.columns:
        raise ValueError(f"Column {ADC} not found in the file.")

    # Return the specified ADC column
    return df[ADC]

def fft(output: str, sample_rate: int, file_names = ["volts.txt"], folder_names = [""]) -> None:
    for i in folder_names:
        for j in file_names:
            # Les I- og Q-signalene fra filen
            I = get_adc_data("ADC5", output + i + j) - 1.65
            Q = get_adc_data("ADC1", output + i + j) - 1.65

            # Kombiner I- og Q-signalene til et komplekst signal
            complex_signal = I + 1j * Q

            # Beregn Fourier-transformasjonen
            spectrum = np.fft.fft(complex_signal)
            freqs = np.fft.fftfreq(len(complex_signal), d=1.0 * sample_rate)  # Beregn frekvensaksen

            # Skift nullfrekvensen til midten av spekteret
            spectrum = np.fft.fftshift(spectrum)
            freqs = np.fft.fftshift(freqs)

            # Finn toppfrekvensen
            max_idx = np.argmax(np.abs(spectrum))  # Indeksen til maksimal amplitude
            max_freq = freqs[max_idx]  # Toppfrekvensen
            max_amplitude = 10 * np.log10(np.abs(spectrum[max_idx]) / np.max(np.abs(spectrum)))

            # Plot spekteret
            plt.figure(figsize=(10, 6))
            plt.plot(freqs, 10*np.log10(np.abs(spectrum)/np.max(np.abs(spectrum))))
            plt.title("Frekvensspekter av I + jQ")
            plt.xlabel("Frekvens [Hz]")
            plt.ylabel("Normalisert amplitude [dB]")
            plt.xlim(- 1 / (sample_rate * 8), 1 / (sample_rate * 8))
            plt.grid()
            plt.legend()

             # Legg til en etikett for toppfrekvensen
            plt.annotate(
                f"Top: {max_freq:.2f} Hz",
                xy=(max_freq, max_amplitude),
                xytext=(max_freq, max_amplitude + 5),
                arrowprops=dict(facecolor='red', arrowstyle="->"),
                fontsize=10,
                color="red"
            )
            
            # Lagre plottet som et bilde
            plt.savefig(output + i + f"{j}_fft.png")
            plt.close()  # Lukk figuren for å frigjøre minne

def plot(output: str, sample_period: int, file_names = ["volts.txt"], folder_names = [""]):
    for i in folder_names:
        for j in file_names:
            # Les I- og Q-signalene fra filen
            I = get_adc_data("ADC5", output + i + j)
            Q = get_adc_data("ADC1", output + i + j)

            time = np.arange(len(I)) * sample_period

            # Plot det komplekse signalet
            plt.figure(figsize=(10, 6))
            plt.plot(time, I, label="I")
            plt.plot(time, Q, label="Q")
            plt.title("I- og Q-signal")
            plt.xlabel("Tid [s]")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.legend()

            # Lagre plottet som et bilde
            plt.savefig(output + i + f"{j}_I_Q.png")
            plt.close()  # Lukk figuren for å frigjøre minne

def spectral_analysis(output: str, adc: str) -> None:
    pass

def var_and_mean():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADC sampler with optional flags.")


    parser.add_argument('--compile', action='store_true', help="Compile the adc_sampler.c code.")
    parser.add_argument('--sample', action='store_true', help="Sample data.")
    parser.add_argument('--transform_data', action='store_true', help="Transform the data from output.bin to readable data.")
    parser.add_argument('--sample_and_run', action='store_true', help="Sample data and run.")
    parser.add_argument('--force', action='store_true', help="Force the script to compile and sample again if alreay compiled.")
    parser.add_argument('--plot', type=str, default="", help="Plot ever plot fpr the measured signal.")
    parser.add_argument('--do_all', type=str, default="", help="Run all the steps.")
    parser.add_argument('--lab4', action='store_true', help="")

    parser.add_argument('--channels', type=int, default=5, help="Number of channels to read from.")
    args = parser.parse_args()

    data_folder = "Data-to-keep/"

    if args.compile:
        compile(data_folder)

    if args.sample:
        run_sampler(data_folder)
    
    if args.transform_data:
        sample_period, data = raspi_import("output.bin", channels=args.channels)
        convert(sample_period, data, data_folder)
    
    if args.sample_and_run:
        run_sampler()
        sample_period, data = raspi_import("output.bin", channels=args.channels)
        convert(sample_period, data, data_folder)

    if args.plot:
        pass

    if args.lab4:
        if not os.path.exists("adc_sampler") or (args.force):
            compile(data_folder)

        if not os.path.exists("output.bin") or (args.force):
            run_sampler(data_folder)

        sample_period, data = raspi_import("output.bin", channels=args.channels)

        convert(sample_period, data, data_folder)

        folder_names = ["/1/", "/2/", "/3/"]
        file_names = ["volts.txt"]

        fft(data_folder, sample_period)

        plot(data_folder, sample_period)


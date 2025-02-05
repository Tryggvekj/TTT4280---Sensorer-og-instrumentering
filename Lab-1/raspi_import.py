import numpy as np
import sys
import subprocess
import time
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

def fft(output: str, adc: str) -> None:
    print("Taking the FFT of the measured signal...")

    number_of_samples = 31250
    file_name = output + "volts.txt"

    # Read the sampling frequency from the first line
    with open(file_name, 'r') as file:
        time_interval = float(file.readline().strip())

    # Read the data from volts.txt, skipping the first line
    if (adc == "ALL"):
        fig, ax = plt.subplots(5, 1, figsize=(10, 15))
        for i in range(5):
            data = pd.read_csv(file_name, skiprows=1)
            fft = np.fft.fft(data["ADC" + str(i + 1)])
            fft = np.abs(fft)

            t = np.linspace(0, 1 / time_interval, number_of_samples)

            ax[i].plot(t, fft)
            ax[i].set_xlabel('Frequency [Hz]')
            ax[i].set_ylabel('Magnitude')
            ax[i].set_title('FFT of measured signal for ADC' + str(i + 1))
    else:
        data = pd.read_csv(file_name, skiprows=1)
        fft = np.fft.fft(data[adc])
        fft = np.abs(fft)

        t = np.linspace(0, 1 / time_interval, number_of_samples)

        plt.plot(t, fft)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.title('FFT of measured signal')

    plt.tight_layout()
    plt.savefig(output + 'FFT.png')
    plt.close()

def plot(output: str, adc: str):
    print("Plotting the measured signal...")
    # Read the sampling frequency from the first line
    with open(output + 'volts.txt', 'r') as file:
        time_interval = float(file.readline().strip())

    # Read the data from volts.txt, skipping the first line
    data = pd.read_csv(output + 'volts.txt', skiprows=1)

    # Scaling factor
    scaling = 1000

    # Calculate the time axis
    time = [i * scaling * time_interval for i in range(len(data["ADC1"]))]

    if (adc == "ALL"):
        fig, ax = plt.subplots(5, 1, figsize=(10, 15))
        for i in range(5):
            ax[i].plot(time, data["ADC" + str(i + 1)])
            ax[i].set_xlabel('Time [ms]')
            ax[i].set_ylabel('Voltage [V]')
            ax[i].set_title('Voltage Readings for ADC' + str(i + 1))
            ax[i].grid(True)
            ax[i].set_xlim(0.005*scaling, 0.01*scaling)
            ax[i].set_ylim(-0.1, 2*1.1)
    else:
        # Extract the data for the specified ADC
        adc_sel = data[adc]
        # Plot every 100th data point to reduce clutter
        plt.plot(time, adc_sel)
        plt.xlabel('Time [ms]')
        plt.ylabel('Voltage [V]')
        plt.title('Voltage Readings')
        plt.grid(True)
        plt.set_xlim(0.005*scaling, 0.01*scaling)

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(output + 'plot.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADC sampler with optional flags.")

    # Arguments
    parser.add_argument('--compile', action='store_true', help="Compile the adc_sampler.c code.")
    parser.add_argument('--sample', action='store_true', help="Sample data.")
    parser.add_argument('--transform_data', action='store_true', help="Transform the data from output.bin to readable data.")
    parser.add_argument('--sample_and_run', action='store_true', help="Sample data and run.")
    parser.add_argument('--force', action='store_true', help="Force the script to compile and sample again if alreay compiled.")
    parser.add_argument('--fft', type=str, default="", help="Take the FFT of the measured signal.")
    parser.add_argument('--plot', type=str, default="", help="Plot the measured signal.")
    parser.add_argument('--do_all', type=str, default="", help="Run all the steps.")
    # Unneccecary arguments
    parser.add_argument('--channels', type=int, default=5, help="Number of channels to read from.")
    args = parser.parse_args()

    data_folder = "Temporary-data/"

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
    
    if args.fft:
        fft(data_folder, args.fft)

    if args.plot:
        plot(data_folder, args.plot)

    if args.do_all:
        if not os.path.exists("adc_sampler") or (args.force):
            compile(data_folder)

        if not os.path.exists("output.bin") or (args.force):
            run_sampler(data_folder)

        sample_period, data = raspi_import("output.bin", channels=args.channels)
        convert(sample_period, data, data_folder)
        fft(data_folder, args.do_all)
        plot(data_folder, args.do_all)
import numpy as np
import sys
import subprocess
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.signal as signal

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
    number_of_samples = 31250
    padding = 1000000
    file_name = output + "volts.txt"
    
    with open(file_name, 'r') as file:
        time_interval = float(file.readline().strip())

    for j in range(3):
        if j == 0:
            print("Taking the FFT of the measured signal...")
        elif j == 1:
            print("Taking the FFT of the measured signal with hanning window...")
        elif j == 2:
            print("Taking the FFT of the measured signal with hamming window...")

        if (adc == "ALL"):
            fig, ax = plt.subplots(5, 1, figsize=(10, 15))
            for i in range(5):
                data = pd.read_csv(file_name, skiprows=1)
                adc_data_org = data["ADC" + str(i + 1)]

                if j == 0:
                    adc_data = adc_data_org
                elif j == 1:
                    adc_data = adc_data_org * np.hanning(len(adc_data_org))
                elif j == 2:
                    adc_data = adc_data_org * np.hamming(len(adc_data_org))

                adc_data = np.pad(adc_data, (0, padding), 'constant')
                adc_data_org = np.pad(adc_data_org, (0, padding), 'constant')
                fft = np.fft.fft(adc_data)
                fft_org = np.fft.fft(adc_data_org)
                fft = np.abs(fft)
                fft_org = np.abs(fft_org)
                fft_db = 20 * np.log10(fft/np.max(fft))
                fft_db_org = 20 * np.log10(fft_org/np.max(fft_org))

                t = np.linspace(0, 1 / time_interval, number_of_samples + padding)

                ax[i].plot(t, fft_db_org, label="Uten vindu")
                if j != 0:
                    ax[i].plot(t, fft_db, label="Med Hanning-vindu" if j == 1 else "Med Hamming-vindu" if j == 2 else None)
                ax[i].set_xlabel('Frekvens [Hz]')
                ax[i].set_ylabel('Normalisert amplitude [dB]')
                ax[i].set_xlim(950, 1050)
                
                if j == 0:
                    ax[i].set_ylim(-90, 0)
                elif j == 1:
                    ax[i].set_ylim(-140, 0)
                elif j == 2:
                    ax[i].set_ylim(-140, 0)
                ax[i].set_title('FFT av målt signal for ADC' + str(i + 1))
                ax[i].legend()
        else:
            data = pd.read_csv(file_name, skiprows=1)
            adc_data_org = data[adc]

            if j == 0:
                adc_data = adc_data_org
            elif j == 1:
                adc_data = adc_data_org * np.hanning(len(adc_data_org))
            elif j == 2:
                adc_data = adc_data_org * np.hamming(len(adc_data_org))

            adc_data = np.pad(adc_data, (0, padding), 'constant')
            adc_data_org = np.pad(adc_data_org, (0, padding), 'constant')
            fft = np.fft.fft(adc_data)
            fft_org = np.fft.fft(adc_data_org)
            fft = np.abs(fft)
            fft_org = np.abs(fft_org)
            fft_db = 20 * np.log10(fft / np.max(fft)) 
            fft_db_org = 20 * np.log10(fft_org / np.max(fft_org))
            
            t = np.linspace(0, 1 / time_interval, number_of_samples + padding)

            plt.plot(t, fft_db_org, label="Uten vindu")
            if j != 0:
                plt.plot(t, fft_db, label="Med Hanning-vindu" if j == 1 else "Med Hamming-vindu" if j == 2 else None)
            plt.xlabel('Frekvens [Hz]')
            plt.ylabel('Normalisert amplitude [dB]')
            plt.xlim(0, 16000)
            if j == 0:
                plt.ylim(-90, 0)
            elif j == 1:
                plt.ylim(-140, 0)
            elif j == 2:
                plt.ylim(-140, 0)
            plt.title('FFT av målt signal for ' + adc)
            plt.legend()

        plt.tight_layout()
        if j == 0:
            plt.savefig(output + 'FFT.png')
        elif j == 1:
            plt.savefig(output + 'FFT_hanning.png')
        elif j == 2:
            plt.savefig(output + 'FFT_hamming.png')
    
        plt.close()

def spectral_analysis(output: str, adc: str) -> None:
    number_of_samples = 31250
    padding = 1000000
    file_name = output + "volts.txt"

    print("Performing spectral analysis of the measured signal...")
    
    with open(file_name, 'r') as file:
        time_interval = float(file.readline().strip())

    data = pd.read_csv(file_name, skiprows=1)
    adc_data_org = data[adc]

    adc_data = np.pad(adc_data_org, (0, padding), 'constant')

    fft = np.fft.fft(adc_data)
    fft = np.abs(fft)
    power_spectrum = np.square(fft) 

    power_spectrum_db = 20 * np.log10(power_spectrum / np.max(power_spectrum))

    t = np.linspace(0, 1 / time_interval, len(fft))

    plt.plot(t, power_spectrum_db)
    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Normalisert effekt [dB]')
    plt.title('Effektspekter av målt signal for ' + adc)
    plt.xlim(950, 1050)
    plt.ylim(-190, 0)

    plt.tight_layout()
    plt.savefig(output + 'Effektspekter' + '.png')
    plt.close()

def plot(output: str, adc: str):
    print("Plotting the measured signal...")

    with open(output + 'volts.txt', 'r') as file:
        time_interval = float(file.readline().strip())

    data = pd.read_csv(output + 'volts.txt', skiprows=1)

    scaling = 1000

    time = [i * scaling * time_interval for i in range(len(data["ADC1"]))]

    if (adc == "ALL"):
        fig, ax = plt.subplots(5, 1, figsize=(10, 15))
        for i in range(5):
            ax[i].plot(time, data["ADC" + str(i + 1)])
            ax[i].set_xlabel('Tid [ms]')
            ax[i].set_ylabel('Spenning [V]')
            ax[i].set_title('Spenningsmåling for ADC' + str(i + 1))
            ax[i].grid(True)
            # ax[i].set_xlim(0.005*scaling, 0.1*scaling)
            ax[i].set_ylim(-0.1, 2*1.1)
    else:
        adc_sel = data[adc]

        plt.plot(time, adc_sel)
        plt.xlabel('Tid [ms]')
        plt.ylabel('Spenning [V]')
        plt.title('Spenningsmåling for' + adc)
        plt.grid(True)
        plt.xlim(0.005*scaling, 0.01*scaling)


    plt.tight_layout()
    plt.savefig(output + 'plot.png')
    plt.close()

def corr(output: str):
    number_of_samples = 31250
    file_name = output + "volts.txt"
    
    with open(file_name, 'r') as file:
        time_interval = float(file.readline().strip())

    data = pd.read_csv(file_name, skiprows=1)

    adc2_data = data["ADC2"] - 1.65
    adc3_data = data["ADC4"] - 1.65
    adc4_data = data["ADC3"] - 1.65

    r_21 = signal.correlate(adc3_data, adc2_data)
    r_31 = signal.correlate(adc4_data, adc2_data)
    r_32 = signal.correlate(adc4_data, adc3_data)
    r_33 = signal.correlate(adc4_data, adc4_data)

    delay_21 = (np.argmax(np.abs(r_21)) - len(adc2_data) + 1 )
    delay_31 = (np.argmax(np.abs(r_31)) - len(adc2_data) + 1 )
    delay_32 = (np.argmax(np.abs(r_32)) - len(adc3_data) + 1 )

    print("n21", delay_21)
    print("n31", delay_31)
    print("n32", delay_32)

    plt.figure()
    plt.plot(r_21)
    plt.title('Cross-correlation between ADC3 and ADC2')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.savefig(output + 'cross_corr_21.png')
    plt.close()

    plt.figure()
    plt.plot(r_31)
    plt.title('Cross-correlation between ADC4 and ADC2')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.savefig(output + 'cross_corr_31.png')
    plt.close()

    plt.figure()
    plt.plot(r_32)
    plt.title('Cross-correlation between ADC4 and ADC3')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.savefig(output + 'cross_corr_32.png')
    plt.close()

    plt.figure()
    plt.plot(r_33)
    plt.title('Cross-correlation between ADC4 and ADC3')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.savefig(output + 'cross_corr_33.png')
    plt.close()
    print(np.argmax(r_33))

    angle = np.arctan2(np.sqrt(3)*(delay_31 + delay_21), (delay_31 - delay_21 + 2*delay_32))*(180/np.pi)

    print(angle)

def var_and_mean():
    with open("Data-to-keep/angles.txt", "r") as f:
        angles = f.readlines()
        angles = [float(angle) for angle in angles]
        for i in range(len(angles)):
            if angles[i] < 0:
                angles[i] += 360
        mean = np.mean(angles)
        var = np.var(angles)
        print("Mean 1: ", mean)
        print("Standard dev 1: ", np.sqrt(var))

    with open("Data-to-keep/angles2.txt", "r") as f:
        angles = f.readlines()
        angles = [float(angle) for angle in angles]
        for i in range(len(angles)):
            if angles[i] < 0:
                angles[i] += 360
        mean = np.mean(angles)
        var = np.var(angles)
        print("Mean 2: ", mean)
        print("Standard dev 2: ", np.sqrt(var))
    
    with open("Data-to-keep/angles3.txt", "r") as f:
        angles = f.readlines()
        angles = [float(angle) for angle in angles]
        for i in range(len(angles)):
            if angles[i] < 0:
                angles[i] += 360
        mean = np.mean(angles)
        var = np.var(angles)
        print("Mean 3: ", mean)
        print("Standard dev 3: ", np.sqrt(var))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADC sampler with optional flags.")


    parser.add_argument('--compile', action='store_true', help="Compile the adc_sampler.c code.")
    parser.add_argument('--sample', action='store_true', help="Sample data.")
    parser.add_argument('--transform_data', action='store_true', help="Transform the data from output.bin to readable data.")
    parser.add_argument('--sample_and_run', action='store_true', help="Sample data and run.")
    parser.add_argument('--force', action='store_true', help="Force the script to compile and sample again if alreay compiled.")
    parser.add_argument('--plot', type=str, default="", help="Plot ever plot fpr the measured signal.")
    parser.add_argument('--do_all', type=str, default="", help="Run all the steps.")
    parser.add_argument('--angle', action='store_true', default=False, help="")
    parser.add_argument('--lab2', action='store_true', help="")

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
        plot(data_folder, args.plot)
        # fft(data_folder, args.plot)
        # spectral_analysis(data_folder, args.plot)

    if args.angle:
        corr(data_folder)

    if args.do_all:
        if not os.path.exists("adc_sampler") or (args.force):
            compile(data_folder)

        if not os.path.exists("output.bin") or (args.force):
            run_sampler(data_folder)

        sample_period, data = raspi_import("output.bin", channels=args.channels)
        convert(sample_period, data, data_folder)
        fft(data_folder, args.do_all)
        plot(data_folder, args.do_all)

    if args.lab2:
        var_and_mean()
        run_sampler(data_folder)
        sample_period, data = raspi_import("output.bin", channels=args.channels)
        convert(sample_period, data, data_folder)
        corr(data_folder)
        # plot(data_folder, "ALL")
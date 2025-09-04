
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


def plot_fft(signal: np.ndarray, fs: int):
    n = len(signal)
    f = np.fft.rfftfreq(n, d=1/fs)
    fft_values = np.fft.rfft(signal)
    magnitude = np.abs(fft_values) / n

    plt.figure()
    plt.plot(f, magnitude)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("FFT of the Signal")
    plt.grid()
    plt.savefig("group_work_1/plots/fft_plot.png", dpi=300)
    plt.show(block = False)


def plot_signal(t: np.ndarray, signal: np.ndarray):
    plt.figure()
    plt.plot(t, signal * 1e6)  # Convert to microvolts
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title("Signal in Time Domain")
    plt.grid()
    plt.savefig("group_work_1/plots/signal_plot.png", dpi=300)
    plt.show(block = False)


def main():
    filepath = Path("group_work_1/Signal1_2018.csv")
    fs = 100  # Sampling frequency

    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return
    
    signal = np.loadtxt(filepath, delimiter=",")
    t = np.arange(len(signal)) / fs  # Time vector

    plot_signal(t, signal)
    plot_fft(signal, fs)

if __name__ == "__main__":
    main()
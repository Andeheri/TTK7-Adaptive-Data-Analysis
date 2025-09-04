
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.signal import chirp
import numpy as np


def plot_fft(signal: np.ndarray, fs: int, tag: str | None = None):
    n = len(signal)
    f = np.fft.rfftfreq(n, d=1/fs)
    fft_values = np.fft.rfft(signal)
    magnitude = np.abs(fft_values) / n

    plt.figure()
    plt.plot(f, magnitude)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title(f"FFT of the Signal {f'with {tag}' if tag else ''}")
    plt.grid()
    plt.savefig(f"group_work_1/plots/fft_plot{'_' + '_'.join(tag.split(' ')) if tag else ''}.png", dpi=300)
    plt.show(block = False)


def plot_signal(t: np.ndarray, signal: np.ndarray, tag: str | None = None):
    plt.figure()
    plt.plot(t, signal * 1e6)  # Convert to microvolts
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title(f"Signal {f'with {tag}' if tag else ''} in Time Domain")
    plt.grid()
    plt.savefig(f"group_work_1/plots/signal_plot{'_' + '_'.join(tag.split(' ')) if tag else ''}.png", dpi=300)
    plt.show(block = False)


def main():
    filepath = Path("group_work_1/Signal1_2018.csv")
    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return
    
    fs = 100  # Sampling frequency

    signal = np.loadtxt(filepath, delimiter=",")
    t = np.arange(len(signal)) / fs  # Time vector

    # Offset parameters
    offset_amplitude = 1e-6  # 1 microvolt
    # Noise parameters
    mu = 0  # Mean
    sigma = 5e-6  # Standard deviation (5 microvolts)
    # Chirp parameters
    start_frequency = 0.1  # Hz
    end_frequency = 10  # Hz
    chirp_duration = t[-1]  # Duration of the chirp signal
    chirp_amplitude = 10e-6  # 10 microvolts
    
    chirp_signal = chirp_amplitude * chirp(t, f0=start_frequency, f1=end_frequency, t1=chirp_duration, method="linear")
    
    # Raw signal
    plot_signal(t, signal)
    plot_fft(signal, fs)

    # Offset analysis
    signal_with_offset = signal + offset_amplitude  # Adding DC offset
    plot_signal(t, signal_with_offset, tag="offset")
    plot_fft(signal_with_offset, fs, tag="offset")

    # White noise analysis
    signal_with_noise = signal + np.random.normal(mu, sigma, size=signal.shape)  # Adding white noise

    plot_signal(t, signal_with_noise, tag="white noise")
    plot_fft(signal_with_noise, fs, tag="white noise")

    # White noise + offset analysis
    signal_with_noise_and_offset = signal_with_noise + offset_amplitude  # Adding DC offset
    plot_signal(t, signal_with_noise_and_offset, tag="white noise and offset")
    plot_fft(signal_with_noise_and_offset, fs, tag="white noise and offset")

    # Chirp analysis
    signal_with_chirp = signal + chirp_signal  # Adding chirp signal
    plot_signal(t, signal_with_chirp, tag="chirp")
    plot_fft(signal_with_chirp, fs, tag="chirp")


if __name__ == "__main__":
    main()
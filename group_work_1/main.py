from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import chirp
from hilbert import perform_hilbert_transform, plot_hilbert_components
from wvt import perform_wvt, plot_wvt_components


def plot_fft(signal: np.ndarray, fs: int, save_path: str, tag: str | None = None):
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
    plt.savefig(f"{save_path}/fft_plot{'_' + '_'.join(tag.split(' ')) if tag else ''}.png", dpi=300)
    plt.show(block = False)


def plot_signal(t: np.ndarray, signal: np.ndarray, save_path: str, tag: str | None = None):
    plt.figure()
    plt.plot(t, signal * 1e6)  # Convert to microvolts
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title(f"Signal {f'with {tag}' if tag else ''} in Time Domain")
    plt.grid()
    plt.savefig(f"{save_path}/signal_plot{'_' + '_'.join(tag.split(' ')) if tag else ''}.png", dpi=300)
    plt.show(block = False)


def process_signal(signal, t, fs, folder_name, tag=None):
    """
    Process a signal with Hilbert transform, WVT, and FFT, saving plots to a specific folder
    
    Parameters:
    -----------
    signal : array_like
        The signal to process
    t : array_like
        Time vector
    fs : float
        Sampling frequency
    folder_name : str
        Folder name to save plots in
    tag : str, optional
        Description of the signal for plot titles
    """
    # Create folder if it doesn't exist
    save_path = f"group_work_1/plots/{folder_name}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Plot signal and FFT
    plot_signal(t, signal, save_path, tag)
    plot_fft(signal, fs, save_path, tag)
    
    # Perform Hilbert Transform with offset preservation
    hilbert_results = perform_hilbert_transform(signal, preserve_offset=True)
    plot_hilbert_components(t, signal, hilbert_results, save_path=save_path)
    
    # Perform WVT
    wvt_results = perform_wvt(signal, fs=fs, window_length=256, step=10)
    plot_wvt_components(t, signal, wvt_results, save_path=save_path, show=True)
    
    print(f"Finished processing signal {f'with {tag}' if tag else 'original'}")


def main():
    filepath = Path("group_work_1/Signal1_2018.csv")
    fs = 100  # Sampling frequency
    
    # Parameters for signal modifications
    offset_amplitude = 2e-6  # 2 microvolt - increased for visibility
    noise_mu = 0  # Mean of white noise
    noise_sigma = 5e-6  # Standard deviation of white noise (5 microvolts)
    
    # Chirp parameters
    chirp_f0 = 5  # Starting frequency (Hz)
    chirp_f1 = 20  # Ending frequency (Hz)
    chirp_amplitude = 10e-6  # 10 microvolts - increased for visibility

    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return
    
    # Load the original signal
    signal = np.loadtxt(filepath, delimiter=",")
    t = np.arange(len(signal)) / fs  # Time vector
    
    # 1. Original signal
    process_signal(signal, t, fs, "original")
    
    # 2. Signal with offset
    signal_with_offset = signal + offset_amplitude
    process_signal(signal_with_offset, t, fs, "offset", "offset")
    
    # 3. Signal with white noise
    signal_with_noise = signal + np.random.normal(noise_mu, noise_sigma, size=signal.shape)
    process_signal(signal_with_noise, t, fs, "noise", "white noise")
    
    # 4. Signal with both offset and white noise
    signal_with_both = signal + offset_amplitude + np.random.normal(noise_mu, noise_sigma, size=signal.shape)
    process_signal(signal_with_both, t, fs, "offset_and_noise", "offset and white noise")
    
    # 5. Signal with chirp component
    chirp_signal = chirp_amplitude * chirp(t, chirp_f0, t[-1], chirp_f1)
    
    # Plot chirp component separately for visualization
    chirp_save_path = "group_work_1/plots/chirp_component"
    Path(chirp_save_path).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, chirp_signal * 1e6)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title("Chirp Signal Component")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    n = len(chirp_signal)
    chirp_f = np.fft.rfftfreq(n, d=1/fs)
    chirp_fft = np.fft.rfft(chirp_signal)
    chirp_magnitude = np.abs(chirp_fft) / n
    plt.plot(chirp_f, chirp_magnitude)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("FFT of Chirp Signal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{chirp_save_path}/chirp_analysis.png", dpi=300)
    
    # Combine with original signal
    signal_with_chirp = signal + chirp_signal
    process_signal(signal_with_chirp, t, fs, "chirp", "chirp")


if __name__ == "__main__":
    main()
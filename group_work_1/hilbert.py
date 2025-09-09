from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def perform_hilbert_transform(signal_data, preserve_offset=False):
    """
    Performs Hilbert transform on input signal and returns components
    
    Parameters:
    -----------
    signal_data : array_like
        Input signal
    preserve_offset : bool, optional
        If True, preserves DC offset in the analysis by adding it back to the real part
        
    Returns:
    --------
    dict
        Dictionary containing Hilbert transform components
    """
    # Calculate mean (DC offset) if we want to preserve it
    dc_offset = np.mean(signal_data) if preserve_offset else 0
    
    # Perform Hilbert transform - use asarray to handle type checking
    analytic_signal = np.asarray(signal.hilbert(signal_data))
    
    # Extract real and imaginary parts with array indexing
    real_part = analytic_signal.view(float)[::2]  # Get even-indexed elements
    imag_part = analytic_signal.view(float)[1::2]  # Get odd-indexed elements
    
    # Add DC offset back to real part if requested
    if preserve_offset:
        real_part = real_part + dc_offset
    
    # Calculate derived properties
    amplitude_envelope = np.sqrt(real_part**2 + imag_part**2)
    instantaneous_phase = np.arctan2(imag_part, real_part)
    instantaneous_phase = np.unwrap(instantaneous_phase)
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)  # Normalized to sampling frequency
    
    return {
        'real_part': real_part,
        'imag_part': imag_part,
        'amplitude_envelope': amplitude_envelope,
        'instantaneous_phase': instantaneous_phase,
        'instantaneous_frequency': instantaneous_frequency,
        'dc_offset': dc_offset
    }

def plot_hilbert_components(t, signal_data, hilbert_results, show=True, save_path=None):
    """
    Plot all Hilbert transform components and visualizations
    """
    real_part = hilbert_results['real_part']
    imag_part = hilbert_results['imag_part']
    amplitude_envelope = hilbert_results['amplitude_envelope']
    instantaneous_frequency = hilbert_results['instantaneous_frequency']
    fs = 1 / (t[1] - t[0])  # Calculate sampling frequency from time vector
    
    # Plot original signal, amplitude envelope, and instantaneous frequency
    fig1 = plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(t, signal_data * 1e6)  # Convert to microvolts
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title("Original Signal")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, amplitude_envelope * 1e6, 'r')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title("Amplitude Envelope")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(t[1:], instantaneous_frequency * fs, 'g')  # Scale by fs to get Hz
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Instantaneous Frequency")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        fig1.savefig(f"{save_path}/hilbert_components.png", dpi=300)
    
    # Plot the real and imaginary parts of the analytic signal
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(t, signal_data * 1e6, label='Original Signal')
    plt.plot(t, real_part * 1e6, '--', label='Real Part')
    plt.plot(t, imag_part * 1e6, ':', label='Imaginary Part')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title("Signal and its Hilbert Transform")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        fig2.savefig(f"{save_path}/hilbert_real_imag.png", dpi=300)
    
    # Add phase space plot (real vs imaginary components)
    fig3 = plt.figure(figsize=(8, 8))
    plt.plot(real_part * 1e6, imag_part * 1e6)
    plt.plot(0, 0, 'r+', markersize=10)  # Mark the origin
    
    # Plot a few points to show the direction
    n_points = 20
    indices = np.linspace(0, len(signal_data)-1, n_points, dtype=int)
    plt.plot(real_part[indices] * 1e6, 
             imag_part[indices] * 1e6, 
             'ro', markersize=5)
    
    # If DC offset is present, mark it on the plot
    if 'dc_offset' in hilbert_results and abs(hilbert_results['dc_offset']) > 1e-10:
        dc_offset = hilbert_results['dc_offset'] * 1e6  # Convert to microvolts
        plt.plot(dc_offset, 0, 'bx', markersize=12, label=f'DC Offset: {dc_offset:.2f} µV')
        plt.legend()
        
        # Add an arrow from origin to DC offset
        plt.annotate('', xy=(dc_offset, 0), xytext=(0, 0),
                    arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8))
    
    plt.xlabel("Real Part [$\\mu V$]")
    plt.ylabel("Imaginary Part [$\\mu V$]")
    plt.title("Phase Space Plot (Hilbert Transform)")
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.grid(True)
    
    if save_path:
        fig3.savefig(f"{save_path}/hilbert_phase_space.png", dpi=300)
    
    # Add a time-colored phase space plot
    fig4 = plt.figure(figsize=(8, 8))
    
    # Color the points based on time
    scatter = plt.scatter(real_part * 1e6, 
                         imag_part * 1e6,
                         c=t, 
                         cmap='viridis', 
                         s=5,
                         alpha=0.7)
    
    plt.colorbar(scatter, label="Time [s]")
    plt.plot(0, 0, 'r+', markersize=10)  # Mark the origin
    plt.xlabel("Real Part [$\\mu V$]")
    plt.ylabel("Imaginary Part [$\\mu V$]")
    plt.title("Time-Colored Phase Space Plot")
    plt.axis('equal')
    plt.grid(True)
    
    if save_path:
        fig4.savefig(f"{save_path}/hilbert_time_colored.png", dpi=300)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.close('all')
    
    return [fig1, fig2, fig3, fig4]  # Return figure handles

def main():
    filepath = Path("group_work_1/Signal1_2018.csv")
    fs = 100  # Sampling frequency

    if not filepath.exists():
        print(f"File {filepath} does not exist.")
        return
    
    signal_data = np.loadtxt(filepath, delimiter=",")
    t = np.arange(len(signal_data)) / fs  # Time vector
    
    hilbert_results = perform_hilbert_transform(signal_data)
    plot_hilbert_components(t, signal_data, hilbert_results)

if __name__ == "__main__":
    main()
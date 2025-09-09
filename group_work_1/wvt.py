from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def perform_wvt(signal_data, fs=100, window_length=None, step=1):
    """
    Performs Wigner-Ville Transform on input signal
    
    Parameters:
    -----------
    signal_data : array_like
        Input signal
    fs : float, optional
        Sampling frequency (default: 100 Hz)
    window_length : int, optional
        Length of analysis window (default: N/10 where N is signal length)
    step : int, optional
        Step size between windows (default: 1)
        
    Returns:
    --------
    dict
        Dictionary containing WVT results and parameters
    """
    N = len(signal_data)
    
    # Default window length if not specified
    if window_length is None:
        window_length = N // 10
        
    # Ensure window length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    half_win = window_length // 2
    
    # Create analytic signal (to eliminate cross-terms from negative frequencies)
    analytic_signal = signal.hilbert(signal_data)
    
    # Compute time and frequency vectors
    t = np.arange(N) / fs
    freq = np.fft.fftfreq(window_length, d=1/fs)
    freq = np.fft.fftshift(freq)
    
    # Initialize Wigner-Ville distribution
    time_points = np.arange(0, N, step)
    wvd = np.zeros((len(freq), len(time_points)))
    
    # For each time point
    for i, center in enumerate(time_points):
        # Get window of data centered at current point
        
        # Handle edge cases with zero padding
        if center < half_win:
            left_pad = half_win - center
            start = 0
            end = center + half_win
            if end > N:
                end = N
            window = np.concatenate((np.zeros(left_pad), analytic_signal[start:end]))
        elif center + half_win >= N:
            right_pad = (center + half_win) - N + 1
            start = center - half_win
            if start < 0:
                start = 0
            end = N
            window = np.concatenate((analytic_signal[start:end], np.zeros(right_pad)))
        else:
            start = center - half_win
            end = center + half_win + 1
            if end > N:
                end = N
            window = analytic_signal[start:end]
        
        # Ensure window has correct length
        if len(window) != window_length:
            # Pad or truncate to correct length
            if len(window) < window_length:
                window = np.pad(window, (0, window_length - len(window)))
            else:
                window = window[:window_length]
        
        # Calculate auto-correlation
        acf = np.zeros(window_length, dtype=complex)
        for tau in range(window_length):
            k = tau - half_win
            if abs(k) <= half_win:
                idx1 = half_win + k
                idx2 = half_win - k
                if 0 <= idx1 < window_length and 0 <= idx2 < window_length:
                    acf[tau] = window[idx1] * np.conjugate(window[idx2])
        
        # FFT of the auto-correlation gives the WVD slice at this time
        wvd_slice = np.fft.fftshift(np.fft.fft(acf))
        wvd[:, i] = np.real(wvd_slice)  # WVD is real-valued for real signals
    
    # Return results as dictionary
    return {
        'wvd': wvd,
        'time': t[time_points],
        'freq': freq,
        'fs': fs,
        'window_length': window_length,
        'step': step
    }


def plot_wvt_components(t, signal_data, wvt_results, show=True, save_path=None):
    """
    Plot Wigner-Ville Transform components and visualizations
    
    Parameters:
    -----------
    t : array_like
        Time vector for the original signal
    signal_data : array_like
        Original signal data
    wvt_results : dict
        Dictionary containing WVT results from perform_wvt function
    show : bool, optional
        Whether to display the plots (default: True)
    save_path : str, optional
        Path to save figures (default: None, no saving)
    
    Returns:
    --------
    list
        List of figure handles
    """
    # Extract results
    wvd = wvt_results['wvd']
    wvt_time = wvt_results['time']
    freq = wvt_results['freq']
    # fs is not used, so we don't need to extract it
    
    # Plot original signal
    fig1 = plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t, signal_data * 1e6)  # Convert to microvolts
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [$\\mu V$]")
    plt.title("Original Signal")
    plt.grid(True)
    
    # Plot Wigner-Ville distribution as a color map (time-frequency representation)
    plt.subplot(2, 1, 2)
    # Use log scale for better visualization
    # Add small constant to avoid log(0)
    vmin = np.max(np.abs(wvd)) / 1e5 if np.max(np.abs(wvd)) > 0 else 1e-10  # Set minimum to a small fraction of maximum for better contrast
    wvd_plot = np.log10(np.abs(wvd) + vmin)
    
    plt.imshow(wvd_plot, aspect='auto', origin='lower', 
               extent=(wvt_time[0], wvt_time[-1], freq[0], freq[-1]),
               cmap='viridis')
    
    plt.colorbar(label='log10(Energy)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Wigner-Ville Distribution')
    plt.tight_layout()
    
    if save_path:
        save_file = f"{save_path}/wvt_components.png"
        print(f"Saving Wigner-Ville components plot to {save_file}")
        fig1.savefig(save_file, dpi=300)
    
    # Plot 3D surface of WVD
    fig2 = plt.figure(figsize=(12, 9))
    ax = fig2.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D plot
    time_mesh, freq_mesh = np.meshgrid(wvt_time, freq)
    
    # Plot the surface
    surf = ax.plot_surface(time_mesh, freq_mesh, wvd_plot, 
                           cmap='viridis', linewidth=0, antialiased=True)
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_zlabel('log10(Energy)')
    ax.set_title('3D Visualization of Wigner-Ville Distribution')
    fig2.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='log10(Energy)')
    plt.tight_layout()
    
    if save_path:
        save_file = f"{save_path}/wvt_3d.png"
        print(f"Saving 3D Wigner-Ville plot to {save_file}")
        fig2.savefig(save_file, dpi=300)
    
    # Plot frequency slices at different times
    fig3 = plt.figure(figsize=(12, 8))
    num_slices = 5
    slice_indices = np.linspace(0, len(wvt_time)-1, num_slices, dtype=int)
    
    for i, idx in enumerate(slice_indices):
        plt.plot(freq, wvd_plot[:, idx], label=f"t = {wvt_time[idx]:.2f}s")
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('log10(Energy)')
    plt.title('Frequency Content at Different Time Slices')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        save_file = f"{save_path}/wvt_freq_slices.png"
        print(f"Saving Wigner-Ville frequency slices plot to {save_file}")
        fig3.savefig(save_file, dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close('all')
    
    return [fig1, fig2, fig3]
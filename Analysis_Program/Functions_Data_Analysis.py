
     #╔════════════════════════════════════════════════════════════════════════════╗#
     #║                         IMPORTING PROJECT MODULES                          ║#
     #╚════════════════════════════════════════════════════════════════════════════╝#


import sys; sys.path.append(r"C:\Users\pm\OneDrive\Bureau\PROJET\Python projet\Turbulence_Analysis\Analysis_Program")
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt
from spacepy import pycdf
from scipy.signal import welch, windows
from Inputs import nperseg, noverlap, rotation


     #╔════════════════════════════════════════════════════════════════════════════╗#
     #║                           FUNCTIONS DEFINITIONS                            ║#
     #╚════════════════════════════════════════════════════════════════════════════╝#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#      Matrix of rotation      #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def calculate_rotation_matrix(B0):
    """
    ┌─────────────────────────────────────────┐
    │ Computes a rotation matrix to align a   │
    │ reference frame with the mean magnetic  │
    │ field direction using Gram-Schmidt      │
    │ orthogonalization.                      │
    │                                         │
    │ Parameters:                             │
    │ ▔▔▔▔▔▔                             │
    │ B0 : numpy.ndarray                      │
    │     Mean magnetic field vector [Bx,     │
    │     By, Bz].                            │
    │                                         │
    │ Returns:                                │
    │ ▔▔▔▔▔                               │
    │ numpy.ndarray                           │
    │     3x3 orthogonal rotation matrix      │
    │     where:                              │
    │     - First row is B0 direction (e1)    │
    │     - Second row is perpendicular to    │
    │       [0,1,0] and e1 (e2)               │
    │     - Third row completes the right-    │
    │       handed system (e3)                │
    └─────────────────────────────────────────┘
    """
    B0_mag = np.linalg.norm(B0)
    e1 = B0 / B0_mag  # Normalize B0 to obtain the first basis vector
    e2 = np.cross([0, 1, 0], e1)  # Compute a vector orthogonal to e1
    e2 = e2 / np.linalg.norm(e2)  # Normalize the second vector
    e3 = np.cross(e1, e2)  # Compute the third orthogonal vector
    return np.array([e1, e2, e3]).T

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     Smooth Data Function     #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def smooth_data(data, window_size):
    """
    ┌─────────────────────────────────────────┐
    │ Smooths data using a moving average     │
    │ filter with edge reflection padding     │
    │                                         │
    │ Parameters:                             │
    │ ▔▔▔▔▔▔                             │
    │ data : numpy.ndarray                    │
    │     Input array to be smoothed          │
    │                                         │
    │ window_size : int                       │
    │     Size of the moving average window.  │
    │     If <= 1, returns original data      │
    │                                         │
    │ Returns:                                │
    │ ▔▔▔▔▔                               │
    │ numpy.ndarray                           │
    │     Smoothed array using convolution    │
    │     with reflected edges. Same length   │
    │     as input, padded if necessary       │
    └─────────────────────────────────────────┘
    """
    if window_size <= 1:
        return data
    
    # Calculate padding size
    pad_size = window_size // 2
    
    # Pad data with reflection at edges
    padded_data = np.pad(data, pad_size, mode='reflect')
    
    # Create moving average window
    window = np.ones(window_size) / window_size
    
    # Apply convolution
    smoothed = np.convolve(padded_data, window, mode='valid')
    
    # Adjust output size if necessary
    if len(smoothed) > len(data):
        smoothed = smoothed[:len(data)]
    elif len(smoothed) < len(data):
        smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode='edge')
    
    return smoothed

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#     Spectrogram Function     #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def calculate_spectrogram_data(frequencies, data_segments, time_array, is_psd=False, window_size=1):
    """
    ┌─────────────────────────────────────────┐
    │ Calculate spectrogram data without      │
    │ plotting, with frequency filtering      │
    │ and optional PSD compensation           │
    │                                         │
    │ Parameters:                             │
    │ ▔▔▔▔▔▔                             │
    │ frequencies : numpy.ndarray             │
    │     Array of frequency values           │
    │                                         │
    │ data_segments : numpy.ndarray           │
    │     2D array of segments data           │
    │                                         │
    │ time_array : numpy.ndarray              │
    │     Array of timestamp values           │
    │                                         │
    │ is_psd : bool, optional                 │
    │     If True, applies frequency-based    │
    │     compensation, log10 transform,      │
    │     and normalization (default: False)  │
    │                                         │
    │ window_size : int, optional             │
    │     Smoothing window size for moving    │
    │     average filter (default: 1)         │
    │                                         │
    │ Returns:                                │
    │ ▔▔▔▔▔                               │
    │ dict                                    │
    │     Dictionary containing:              │
    │     - frequencies: frequencies filtered │
    │       to 0-132 Hz range                 │
    │     - data: processed and smoothed      │
    │       spectrogram data                  │
    │     - segment_edges: time boundaries    │
    │       for plotting                      │
    │     - freq_edges: logarithmic frequency │
    │       boundaries for plotting           │
    └─────────────────────────────────────────┘
    """
    # Filter frequencies up to 132 Hz
    mask = (frequencies > 0) & (frequencies <= 132)
    frequencies_reduced = frequencies[mask]
    data_reduced = data_segments[:, mask]
    
    # Compensation for PSD
    if is_psd:
        # Softer compensation with lower exponent
        freq_compensation = frequencies_reduced**2  # or frequencies_reduced**1.5
        data_reduced = np.log10(data_reduced * freq_compensation[np.newaxis, :])
        
        # Normalization by frequency band
        for i in range(len(frequencies_reduced)):
            data_reduced[:, i] = (data_reduced[:, i] - np.mean(data_reduced[:, i])) / np.std(data_reduced[:, i])
        
        # Additional normalization to increase contrast
        data_reduced = (data_reduced - np.mean(data_reduced)) * 1.5 + np.mean(data_reduced)
    
    # Smooth data
    smoothed_data = np.zeros_like(data_reduced)
    for i in range(data_reduced.shape[0]):
        smoothed_data[i] = smooth_data(data_reduced[i], window_size)
    
    # Calculate temporal parameters
    time_num = plt.matplotlib.dates.date2num(time_array)
    n_segments = smoothed_data.shape[0]
    segment_duration = (time_num[-1] - time_num[0]) / n_segments
    segment_times = np.linspace(time_num[0], time_num[-1], n_segments)
    
    # Create edges for plotting
    segment_edges = np.append(segment_times - segment_duration/2,
                            segment_times[-1] + segment_duration/2)
    
    # Calculate frequency edges
    log_freq = np.log10(frequencies_reduced)
    d_log_freq = np.diff(log_freq).mean()
    freq_edges = np.append(frequencies_reduced,
                          10**(np.log10(frequencies_reduced[-1]) + d_log_freq))
    
    return {
        'frequencies': frequencies_reduced,
        'data': smoothed_data,
        'segment_edges': segment_edges,
        'freq_edges': freq_edges
    }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#      Segment Analysis Function (PSD and Helicity)       #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def compute_segment_analysis(B_R, B_T, B_N, fs, rotation, time_index):
    """
    ┌─────────────────────────────────────────┐
    │ Compute Power Spectral Density (PSD)    │
    │ and magnetic helicity for data segments │
    │ using Welch's method.                   │
    │                                         │
    │ Parameters:                             │
    │ ▔▔▔▔▔▔                             │
    │ B_R : numpy.ndarray                     │
    │     Radial component of magnetic field  │
    │                                         │
    │ B_T : numpy.ndarray                     │
    │     Tangential component of magnetic    │
    │     field                               │
    │                                         │
    │ B_N : numpy.ndarray                     │
    │     Normal component of magnetic field  │
    │                                         │
    │ fs : float                              │
    │     Sampling frequency in Hz            │
    │                                         │
    │ rotation : str                          │
    │     'yes' for field-aligned coords      │
    │     'no' for RTN coordinates            │
    │                                         │
    │ time_index : numpy.ndarray              │
    │     Time points for the data segments   │
    │                                         │
    │ Returns:                                │
    │ ▔▔▔▔▔                               │
    │ tuple:                                  │
    │     frequencies : numpy.ndarray         │
    │         Frequency array from Welch      │
    │                                         │
    │     psd_segments : numpy.ndarray        │
    │         Total PSD (R+T+N) for each      │
    │         segment using Welch's method    │
    │                                         │
    │     helicity_freq : numpy.ndarray       │
    │         Frequency array for helicity    │
    │         from FFT                        │
    │                                         │
    │     helicity_segments : numpy.ndarray   │
    │         Normalized magnetic helicity    │
    │         for each segment                │
    │                                         │
    │     theta_BR : numpy.ndarray            │
    │         Angle between mean field and    │
    │         radial direction (radians)      │
    │                                         │
    │     segment_times : numpy.ndarray       │
    │         Center times for each segment   │
    └─────────────────────────────────────────┘
    """

    step = int(nperseg - noverlap)
    n_segments = int((len(B_R) - noverlap) // step)
    
    print("\nSegment division information:")
    print(f"Segment size (nperseg): {nperseg} points")
    print(f"Overlap (noverlap): {noverlap} points")
    print(f"Step between segments: {step} points")
    print(f"Total number of segments: {n_segments}")
    print(f"Total data length: {len(B_R)} points")
    print(f"Field-aligned rotation: {rotation}\n")
    
    psd_segments = []  #    --- Power Spectral Density Segments ---
    helicity_segments = []  #    --- Magnetic Helicity Segments ---
    theta_BR = []  #    --- Store angles between mean field and radial direction ---
    frequencies = None  #    --- Frequency array ---
    
    B_data = np.array([B_R, B_T, B_N]).T   #    --- Magnetic field data ---
    
    for i in range(n_segments):   #    --- Loop over segments ---
        start = int(i * step)
        end = int(start + nperseg)
        
        segment = B_data[start:end]
        
        # Calculate mean magnetic field components for this segment
        B0 = np.mean(segment, axis=0)
        
        # Calculate angle between mean field and radial direction
        B_magnitude = np.sqrt(np.sum(B0**2))
        if B_magnitude > 0:
            angle = np.arccos(B0[0] / B_magnitude)
        else:
            angle = np.nan
        theta_BR.append(angle)
        
        if rotation == 'yes':   #    --- Field-aligned rotation selected in Inputs ---
            # Calculate and apply rotation
            R = calculate_rotation_matrix(B0)
            segment_rotated = np.dot(segment, R)
            
            B_x = segment_rotated[:, 0]
            B_y = segment_rotated[:, 1]
            B_z = segment_rotated[:, 2]
        else:
            B_x = segment[:, 0] 
            B_y = segment[:, 1]  
            B_z = segment[:, 2] 

        # Apply Hann window
        window = np.hanning(len(B_y))
        B_y_windowed = B_y * window
        B_z_windowed = B_z * window

        # Compute PSD using Welch
        if frequencies is None:
            frequencies, psd_R = welch(B_x, fs, window='hann', nperseg=nperseg)
            _, psd_T = welch(B_y, fs, window='hann', nperseg=nperseg)
            _, psd_N = welch(B_z, fs, window='hann', nperseg=nperseg)
        else:
            _, psd_R = welch(B_x, fs, window='hann', nperseg=nperseg)
            _, psd_T = welch(B_y, fs, window='hann', nperseg=nperseg)
            _, psd_N = welch(B_z, fs, window='hann', nperseg=nperseg)

        # Total PSD for segment
        psd_segments.append(psd_R + psd_T + psd_N)

        # Compute FFT for helicity
        fft_y = np.fft.rfft(B_y_windowed)
        fft_z = np.fft.rfft(B_z_windowed)

        # Helicity calculation
        trace = np.abs(fft_y)**2 + np.abs(fft_z)**2
        numerator = 2 * np.imag(np.conj(fft_y) * fft_z)
        sigma_m = np.zeros_like(trace)
        mask = trace > 0
        sigma_m[mask] = numerator[mask] / trace[mask]
        
        helicity_segments.append(sigma_m)
    
    # Calculate segment times (center of each segment)
    segment_times = [
        time_index[int(start + (nperseg // 2))] for start in range(0, len(time_index) - nperseg + 1, step)
    ]

    # Frequency array for helicity
    helicity_freq = np.fft.rfftfreq(len(B_y_windowed), 1/fs)
    
    return (
        frequencies,
        np.array(psd_segments),
        helicity_freq,
        np.array(helicity_segments),
        np.array(theta_BR),
        np.array(segment_times)
    )
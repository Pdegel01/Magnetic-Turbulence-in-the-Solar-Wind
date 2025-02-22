
 # Ne pas oublier d'enlever les titres effacées à la fin
 # Rajouter l'affichage des informations importantes (print)
 # Corriger le texte 0.2 ou 0.25 dans le dernier graphique
 
     #╔════════════════════════════════════════════════════════════════════════════╗#
     #║                         IMPORTING PROJECT MODULES                          ║#
     #╚════════════════════════════════════════════════════════════════════════════╝#


import sys; sys.path.append(r"C:\Users\pm\OneDrive\Bureau\PROJET\Python projet\Turbulence_Analysis\Analysis_Program")
import numpy as np, pandas as pd, os, matplotlib.pyplot as plt, matplotlib.dates as mdates, matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import num2date
from matplotlib.ticker import MultipleLocator
from spacepy import pycdf
from scipy import integrate
from scipy.signal import welch, windows, savgol_filter
from scipy.ndimage import gaussian_filter1d
from Functions_Data_Analysis import compute_segment_analysis, smooth_data, calculate_spectrogram_data
from Inputs import Data_path, Data_name, start_time, end_time, rotation, low_limit_cyclo, high_limit_cyclo, low_limit_w, high_limit_w, low_filtred_limit, high_filtred_limit


     #╔════════════════════════════════════════════════════════════════════════════╗#
     #║                            DATA INITIALISATION                             ║#
     #╚════════════════════════════════════════════════════════════════════════════╝#


# Data Loading
file_path = os.path.join(Data_path, Data_name)
with pycdf.CDF(file_path) as cdfData_name:
    B_merg = cdfData_name["B_RTN"][...]
    freq = cdfData_name["SAMPLING_RATE"][...]
    epoch = cdfData_name["Epoch"][...]

# Data filtred function of defined time interval
df = pd.DataFrame({'B_R': B_merg[:, 0], 'B_T': B_merg[:, 1], 'B_N': B_merg[:, 2]}, 
                 index=pd.to_datetime(epoch))

# Filtration of data between the defined time interval
start_time_obj = pd.to_datetime(start_time, format='%H:%M:%S').time()
end_time_obj = pd.to_datetime(end_time, format='%H:%M:%S').time()
df_filtered = df.between_time(start_time_obj, end_time_obj)


#╔════════════════════════════════════════════════════════════════════════════╗#
#║                             NOISE PROCESSING                               ║#
#╚════════════════════════════════════════════════════════════════════════════╝#


# Data Loading
noise_data = pd.read_csv(r"C:\Users\pm\OneDrive\Bureau\PROJET\Documents Projet\Mergebnoise_256_VR3_PSDtot.csv", 
                         header=0, 
                         names=['frequency', 'psd'],
                         sep=',',
                         na_values=['******'])

# Data Cleaning
noise_data = noise_data.apply(pd.to_numeric, errors='coerce')
noise_data = noise_data.dropna()
noise_data = noise_data.sort_values(by='frequency')

def calculate_noise_level_direct(psd):
    """
    ┌─────────────────────────────────────────┐
    │ Calculate 3-sigma noise level by        │
    │ directly multiplying the PSD by 3.0     │
    │                                         │
    │ Parameters:                             │
    │ ▔▔▔▔▔▔                             │
    │ psd : numpy.ndarray                     │
    │     Power Spectral Density array        │
    │                                         │
    │ Returns:                                │
    │ ▔▔▔▔▔                               │
    │ numpy.ndarray                           │
    │     3-sigma noise level array           │
    └─────────────────────────────────────────┘
    """
    return 3.0 * psd

noise_3_sigma = calculate_noise_level_direct(noise_data['psd'].values) #   ---   Noise computation  ---


     #╔════════════════════════════════════════════════════════════════════════════╗#
     #║                   DATA PROCESSING AND VISUALIZATIONS                       ║#
     #╚════════════════════════════════════════════════════════════════════════════╝#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Magnetic Field Evolution   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def create_figure_1(df_filtered):
    """
    ┌─────────────────────────────────────────┐
    │ Generates a plot of the magnetic field  │
    │ components and their magnitude.         │
    │                                         │
    │ Parameters:                             │
    │ ▔▔▔▔▔▔                             │
    │ df_filtered : pandas.DataFrame          │
    │     DataFrame containing columns        │
    │     'B_R', 'B_T', 'B_N' and a time      │
    │     index.                              │
    │                                         │
    │ Returns:                                │
    │ ▔▔▔▔▔                               │
    │ None                                    │
    │     Displays a time series plot of      │
    │     the magnetic field components and   │
    │     their magnitude.                    │
    └─────────────────────────────────────────┘
    """
    df_filtered = df_filtered.copy()
    df_filtered["B"] = np.sqrt(df_filtered["B_R"]**2 +   #   ---  Compute the magnitude of the magnetic field  ---
                               df_filtered["B_T"]**2 + 
                               df_filtered["B_N"]**2)

    start_date = df_filtered.index[0].strftime("%B %d, %Y")  # Data Loading
    start_time = df_filtered.index[0].strftime("%H:%M")
    end_time = df_filtered.index[-1].strftime("%H:%M")

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    
    colors = {'B_R': '#ffb3b3', 'B_T': '#b3ffb3', 'B_N': '#b3b3ff', 'B': 'black'}
    for comp, color in colors.items():
        plt.plot(df_filtered.index, df_filtered[comp], label=comp, color=color)
    plt.title(f'Magnetic Field Evolution\n{start_date}\nfrom {start_time} → {end_time}')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.xlim(df_filtered.index[0], df_filtered.index[-1])
    plt.ylabel('Magnetic Field (nT)')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)  # Enables grid in the background
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Power Spectral Density    #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Compute PSD et hélicité
frequencies_psd, psd_segments, frequencies_helicity, helicity_segments, theta_BR, segment_times = compute_segment_analysis(
    df_filtered['B_R'].values,
    df_filtered['B_T'].values,
    df_filtered['B_N'].values,
    freq[0],
    rotation,
    df_filtered.index
)

def create_figure_2(frequencies, psd_segments, noise_3_sigma, low_filtred_limit, high_filtred_limit):
    """
    ┌─────────────────────────────────────────┐
    │ Plots PSD segments and identifies those │
    │ exceeding noise in a certain range.     │
    │                                         │
    │ Parameters:                             │
    │ ▔▔▔▔▔                               │
    │ frequencies : numpy.ndarray             │
    │     Array of frequency values [Hz].     │
    │                                         │
    │ psd_segments : list of numpy.ndarray    │
    │     List of PSD segments.               │
    │                                         │
    │ noise_3_sigma : numpy.ndarray           │
    │     3σ noise level for each frequency.  │
    │                                         │
    │ low_filtred_limit : float               │
    │     Lower frequency limit for filtering.│
    │                                         │
    │ high_filtred_limit : float              │
    │     Upper frequency limit for filtering.│
    │                                         │
    │ Returns:                                │
    │ ▔▔▔▔                                 │
    │ None                                    │
    │     Displays two log-log plots:         │
    │     - All PSD segments + noise level.   │
    │     - Only segments above noise in      │
    │       a certain range.                  │
    └─────────────────────────────────────────┘
    """
    # Font size settings - modifications ici pour augmenter la taille des valeurs sur les axes
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18
    
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # Augmentation de la taille des étiquettes des graduations sur l'axe X
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # Augmentation de la taille des étiquettes des graduations sur l'axe Y
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Ensure noise_3_sigma is interpolated to match the frequency size
    if len(frequencies) != len(noise_3_sigma):
        noise_3_sigma = np.interp(frequencies, noise_data['frequency'], noise_3_sigma)
    
    # ──────────────────────── All PSD segments ────────────────────────
    ax1 = fig.add_subplot(gs[0])
    
    # Smoothing and segments Representation
    for psd in psd_segments:
        smoothed_psd = smooth_data(psd, window_size=20)
        ax1.loglog(frequencies, smoothed_psd, color='gray', alpha=0.2, linewidth=0.5)
    
    # Mean PSD
    mean_psd = np.mean([smooth_data(psd, window_size=20) for psd in psd_segments], axis=0)
    ax1.loglog(frequencies, mean_psd, color='black', linewidth=2, label='Mean PSD')
    ax1.loglog(frequencies, noise_3_sigma, linestyle="-", color="lime", linewidth=2, label='3σ Noise Level')

    # Add slopes
    '''
    freq1 = np.logspace(-2, 0, 100)
    psd1 = 10 * freq1**(-5/3)
    ax1.loglog(freq1, psd1, label="$k^{-5/3}$", linestyle="--", color="red", linewidth=2)
    '''
    freq2 = np.logspace(np.log10(3), 1.5, 90)
    psd2 = 20 * freq2**(-8/3)
    ax1.loglog(freq2, psd2, label="$k^{-8/3}$", linestyle="--", color="blue", linewidth=2)
    
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('PSD [(nT)²/Hz]')
    ax1.set_title(f'Power Spectral Density - {len(psd_segments)} Segments')
    ax1.grid(True)
    ax1.set_xlim(0.2, 100)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # ──────────────────────── Filtred PSD segments ────────────────────────
    ax2 = fig.add_subplot(gs[1])
    
    mask_noise = (frequencies >= low_filtred_limit) & (frequencies <= high_filtred_limit)
    mean_noise_level = np.mean(noise_3_sigma[mask_noise])
    
    valid_segments = []
    for psd in psd_segments:
        smoothed_psd = smooth_data(psd, window_size=20)
        mask_interval = (frequencies >= low_filtred_limit) & (frequencies <= high_filtred_limit)
        mean_psd_level = np.mean(smoothed_psd[mask_interval])
        
        if mean_psd_level > mean_noise_level:
            valid_segments.append(smoothed_psd)
            ax2.loglog(frequencies, smoothed_psd, color='gray', alpha=0.2, linewidth=0.5)
    
    if valid_segments:
        mean_valid_psd = np.mean(valid_segments, axis=0)
        ax2.loglog(frequencies, mean_valid_psd, color='black', linewidth=2, label='Mean PSD')
    
    ax2.loglog(frequencies, noise_3_sigma, linestyle="-", color="lime", linewidth=2, label='3σ Noise Level')
    '''
    ax2.loglog(freq1, psd1, label="$k^{-5/3}$", linestyle="--", color="red", linewidth=2)
    '''
    ax2.loglog(freq2, psd2, label="$k^{-8/3}$", linestyle="--", color="blue", linewidth=2)
    
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('PSD [(nT)²/Hz]')
    ax2.set_title(f'Power Spectral Density Filtred - {len(valid_segments)} Segments')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.set_xlim(0.2, 100)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4) 
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    Spectrograms of PSD and Helicity    #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def create_figure_3(df_filtered, frequencies_psd, psd_segments, 
                    frequencies_helicity, helicity_segments, theta_BR, window_size=0):
    """
    ┌─────────────────────────────────────────────────────────┐
    │ Creates a figure with three subplots:                   │
    │ PSD, Helicity and θBR angle                             │
    │                                                         │
    │ Parameters:                                             │
    │ ▔▔▔▔▔▔                                             │
    │ df_filtered : pandas.DataFrame                          │
    │     Time-indexed magnetic field data.                   │
    │ frequencies_psd : np.ndarray                            │
    │     Frequencies for PSD calculation.                    │
    │ psd_segments : np.ndarray                               │
    │     PSD segment data.                                   │
    │ frequencies_helicity : np.ndarray                       │
    │     Frequencies for helicity calculation.               │
    │ helicity_segments : np.ndarray                          │
    │     Helicity segment data.                              │
    │ theta_BR : np.ndarray                                   │
    │     Angle between B field and radial direction.         │
    │ window_size : int, default=0                            │
    │     Window size for smoothing.                          │
    │                                                         │
    │ Returns:                                                │
    │ ▔▔▔▔                                                 │
    │ None                                                    │
    │     Displays spectrograms of PSD, helicity and angle.   │
    └─────────────────────────────────────────────────────────┘
    """
    # Set font sizes with increased values
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18  # Modified sizes
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # Increased x-tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # Increased y-tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)

    start_time = df_filtered.index[0].strftime("%H:%M")
    end_time = df_filtered.index[-1].strftime("%H:%M")

    # Create figure with constrained layout
    fig, axs = plt.subplots(3, 1, figsize=(15, 15), 
                            gridspec_kw={'height_ratios': [1, 1, 0.5]}, 
                            constrained_layout=True)

    # ──────────────────────── PSD Spectrogram ────────────────────────

    psd_data = calculate_spectrogram_data(frequencies_psd, psd_segments, 
                                          df_filtered.index, is_psd=True)
    
    im1 = axs[0].pcolormesh(psd_data['segment_edges'], psd_data['freq_edges'], 
                            psd_data['data'].T, cmap='viridis',
                            norm=plt.Normalize(vmin=-4, vmax=4), shading='auto')

    axs[0].set_yscale('log')
    axs[0].set_ylim(1e-1, 1e2)
    axs[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axs[0].set_ylabel('Frequency (Hz)')
    axs[0].set_title('Power Spectral Density Spectrogram')
    axs[0].tick_params(labelbottom=False)

    cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical', pad=0.02)
    cbar1.set_label('log₁₀(PSD) [(nT)²/Hz]', size=MEDIUM_SIZE)  # Explicit size

    # ──────────────────────── Helicity Spectrogram ────────────────────────

    helicity_data = calculate_spectrogram_data(frequencies_helicity, helicity_segments,
                                               df_filtered.index, window_size=window_size)
    
    colors = ['#ff0000', '#ffffff', '#0000ff']
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
    
    im2 = axs[1].pcolormesh(helicity_data['segment_edges'], helicity_data['freq_edges'],
                            helicity_data['data'].T, cmap=custom_cmap,
                            norm=plt.Normalize(vmin=-1, vmax=1), shading='auto')
    axs[1].set_yscale('log')
    axs[1].set_ylim(1e-1, 1e2)
    axs[1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axs[1].set_ylabel('Frequency (Hz)')
    axs[1].set_title('Magnetic Helicity Spectrogram')
    axs[1].tick_params(labelbottom=False)

    cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical', pad=0.02)
    cbar2.set_label('Helicity', size=MEDIUM_SIZE)  # Explicit size

    # ──────────────────────── θBR Angle Plot ────────────────────────

    segment_edges = psd_data['segment_edges']
    segment_centers = (segment_edges[:-1] + segment_edges[1:]) / 2
    segment_times = [num2date(t) for t in segment_centers]

    theta_BR_deg = np.degrees(theta_BR)
    axs[2].plot(segment_times, theta_BR_deg, 'k-', linewidth=1)

    axs[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axs[2].axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    axs[2].axhline(y=180, color='gray', linestyle='--', alpha=0.5)

    axs[2].set_ylim(0, 180)
    axs[2].set_ylabel('θBR (degrees)')
    axs[2].set_xlabel(f'Time ({start_time} → {end_time})')
    axs[2].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axs[2].set_title('Magnetic Field - Radial Direction Angle')
    axs[2].grid(True, alpha=0.3)

    axs[2].yaxis.set_major_locator(MultipleLocator(45))
    axs[2].yaxis.set_minor_locator(MultipleLocator(15))
    axs[2].tick_params(axis='x', rotation=45)

    # Synchronize x-axis limits
    axs[2].set_xlim(axs[0].get_xlim())

    plt.show()
    print(f'Min θBR: {theta_BR_deg.min()}°, Max θBR: {theta_BR_deg.max()}°')
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#        PSD and Helicity filtred        #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def create_figure_4(frequencies, psd_segments, helicity_freqs, helicity_segments, noise_3_sigma, low_limit_cyclo, high_limit_cyclo, low_limit_w, high_limit_w, low_filtred_limit, high_filtred_limit):
    """
    ┌───────────────────────────────────────────────┐
    │ Creates a figure with four subplots           │
    │ for PSD and Helicity filtering                │
    │                                               │                
    │ Parameters:                                   │               
    │ ─────────                                     │             
    │ frequencies : np.ndarray                      │               
    │     Frequency array for PSD analysis.         │               
    │ psd_segments : np.ndarray                     │               
    │     PSD segment data.                         │               
    │ helicity_freqs : np.ndarray                   │              
    │     Frequency array for helicity analysis.    │               
    │ helicity_segments : np.ndarray                │              
    │     Helicity segment data.                    │              
    │ noise_3_sigma : np.ndarray                    │             
    │     Noise threshold for filtering.            │              
    │ low_limit_cyclo : float                       │
    │     Cyclotron band lower limit [Hz].          │
    │ high_limit_cyclo : float                      │
    │     Cyclotron band upper limit [Hz].          │
    │ low_limit_w : float                           │
    │     Whistler band lower limit [Hz].           │
    │ high_limit_w : float                          │
    │     Whistler band upper limit [Hz].           │
    │ low_filtred_limit : float                     │
    │     Noise filter lower limit [Hz].            │
    │ high_filtred_limit : float                    │
    │     Noise filter upper limit [Hz].            │
    │                                               │                
    │ Returns:                                      │               
    │ ───────                                       │           
    │ None                                          │              
    │     Displays filtered spectrograms            │
    │     for PSD and helicity.                     │
    └───────────────────────────────────────────────┘
    """
    # Set font sizes with increased values
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18  # Modified sizes
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # Increased x-tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # Increased y-tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)

    print('low_limit_cyclo = ' + str(low_limit_cyclo))
    print('high_limit_cyclo =' + str(high_limit_cyclo))
    print('low_limit_w = ' + str(low_limit_w))
    print('high_limit_w = ' + str(high_limit_w))
    # Create frequency array corresponding to noise data
    noise_freqs = np.linspace(frequencies.min(), frequencies.max(), len(noise_3_sigma))

    if len(frequencies) != len(noise_3_sigma):
        noise_3_sigma = np.interp(frequencies, noise_data['frequency'], noise_3_sigma)

    # Create the main figure with a 2x2 grid layout
    fig = plt.figure(figsize=(24, 12))
    gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # === Compute helicity-based filtering (cyclo) ===
    mask_band = (helicity_freqs >= low_limit_cyclo) & (helicity_freqs <= high_limit_cyclo)
    mean_helicity = np.mean(helicity_segments[:, mask_band], axis=1)
    
    original_high_helicity = mean_helicity > 0.2  # Store original condition
    
    # === Compute helicity-based filtering (whistler) ===
    mask_band_1 = (helicity_freqs >= low_limit_w) & (helicity_freqs <= high_limit_w)
    mean_helicity_1 = np.mean(helicity_segments[:, mask_band_1], axis=1)
    
    original_high_helicity_1 = mean_helicity_1 < -0.2  # Store original condition

    # Make high_helicity and high_helicity_1 mutually exclusive
    high_helicity = original_high_helicity & ~original_high_helicity_1
    high_helicity_1 = original_high_helicity_1 & ~original_high_helicity

    # Combine masks
    filtered_segments = high_helicity | high_helicity_1
    remaining_segments = ~filtered_segments

    # Informations
    print("\nfirst case (without filtre):")
    print(f"blue segments (cyclotron): {np.sum(high_helicity)}")
    print(f"red segments (whistler): {np.sum(high_helicity_1)}")
    print(f"remaining segments: {np.sum(remaining_segments)}")

    # ──────────────────────── Left Side (All Segments) ────────────────────────

    ax1 = fig.add_subplot(gs[0, 0])  # PSD Plot
    
    # Find index corresponding to à 0.1 Hz
    f_ref_idx = np.argmin(np.abs(frequencies - 0.1))

    # Compute mean PSD
    mean_psd_high = np.mean(psd_segments[high_helicity], axis=0)
    mean_psd_high_1 = np.mean(psd_segments[high_helicity_1], axis=0)
    mean_psd_remaining = np.mean(psd_segments[remaining_segments], axis=0)

    # Normalized Normalize with respect to the value of remaining_segments at 0.1 Hz  
    norm_factor = mean_psd_remaining[f_ref_idx]
    mean_psd_high_norm = mean_psd_high * (norm_factor / mean_psd_high[f_ref_idx])
    mean_psd_high_1_norm = mean_psd_high_1 * (norm_factor / mean_psd_high_1[f_ref_idx])

    # Plot normalized PSDs  
    ax1.loglog(frequencies, mean_psd_high_norm, 'b-', linewidth=2, 
               label=f'σₘ > 0.2 on [{low_limit_cyclo}-{high_limit_cyclo}Hz] → {np.sum(high_helicity)} segments')
    ax1.loglog(frequencies, mean_psd_high_1_norm, 'r-', linewidth=2, 
               label=f'σₘ < -0.2 on [{low_limit_w}-{high_limit_w}Hz] → {np.sum(high_helicity_1)} segments')
    ax1.loglog(frequencies, mean_psd_remaining, 'k-', linewidth=2, 
               label=f'Remaining segments → {np.sum(remaining_segments)}')
    
    # Noise threshold line
    ax1.loglog(frequencies, noise_3_sigma, linestyle="-", color="lime", linewidth=2, label='3σ Noise Level')
    
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('PSD [(nT)²/Hz]')
    ax1.set_title('All Segments - Power Spectral Density')
    ax1.grid(True)
    ax1.set_xlim(1e-1, 100)

    # Second subplot on the left: Helicity
    ax2 = fig.add_subplot(gs[1, 0])

    # Extract high and low helicity segments
    hel_high = helicity_segments[high_helicity]
    hel_high_1 = helicity_segments[high_helicity_1]
    hel_remaining = helicity_segments[remaining_segments]

    # Calculate mean helicity for each group
    mean_hel_high = np.mean(hel_high, axis=0)
    mean_hel_high_1 = np.mean(hel_high_1, axis=0)
    mean_hel_remaining = np.mean(hel_remaining, axis=0)

    # Plot the mean helicity curves for each group
    ax2.semilogx(helicity_freqs, mean_hel_high, 'b-', linewidth=2,
                 label=f'σₘ > 0.2 on [{low_limit_cyclo}-{high_limit_cyclo}Hz] → {np.sum(high_helicity)} segments')
    ax2.semilogx(helicity_freqs, mean_hel_high_1, 'r-', linewidth=2,
                 label=f'σₘ < -0.2 on [{low_limit_w}-{high_limit_w}Hz] → {np.sum(high_helicity_1)} segments')
    ax2.semilogx(helicity_freqs, mean_hel_remaining, 'k-', linewidth=2,
                 label=f'Remaining segments → {np.sum(remaining_segments)}')

    # Set labels and title for the left subplot
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Magnetic Helicity')
    ax2.set_title('All Segments - Magnetic Helicity')
    ax2.grid(True)
    ax2.set_xlim(1e-1, 100)
    ax2.set_ylim(-1, 1)

    # ──────────────────────── Right Side (Above Noise Segments) ────────────────────────
 
    # Additional filtering based on noise level
    above_noise_mask = []

    # Mean PSD Level Comparison Against Noise Level
    for psd in psd_segments:
        mask_interval = (frequencies >= low_filtred_limit) & (frequencies <= high_filtred_limit)
        mean_psd_level = np.mean(psd[mask_interval])
        noise_level = np.mean(noise_3_sigma[mask_interval])
        above_noise_mask.append(mean_psd_level > noise_level)

    above_noise_mask = np.array(above_noise_mask)

    # Apply noise filtering to segments already filtered by helicity
    high_helicity_noise = high_helicity & above_noise_mask
    high_helicity_noise_1 = high_helicity_1 & above_noise_mask

    # Create a combined mask for noise
    filtered_segments_noise = high_helicity_noise | high_helicity_noise_1
    remaining_segments_noise = above_noise_mask & (~filtered_segments_noise)

    # Informations
    print("\n2nd case (with filter):")
    print(f"blue segments (cyclotron): {np.sum(high_helicity_noise)}")
    print(f"red segments (whistler): {np.sum(high_helicity_noise_1)}")
    print(f"remaining segments: {np.sum(remaining_segments_noise)}")

    # First right subplot: Filtered PSD
    ax3 = fig.add_subplot(gs[0, 1])

    # Calculate means for PSD with noise
    mean_psd_high_noise = np.mean(psd_segments[high_helicity_noise], axis=0)
    mean_psd_high_1_noise = np.mean(psd_segments[high_helicity_noise_1], axis=0)
    mean_psd_remaining_noise = np.mean(psd_segments[remaining_segments_noise], axis=0)

    # Normalize by the value of remaining_segments_noise at 0.1 Hz
    norm_factor_noise = mean_psd_remaining_noise[f_ref_idx]
    mean_psd_high_noise_norm = mean_psd_high_noise * (norm_factor_noise / mean_psd_high_noise[f_ref_idx])
    mean_psd_high_1_noise_norm = mean_psd_high_1_noise * (norm_factor_noise / mean_psd_high_1_noise[f_ref_idx])

    ax3.loglog(frequencies, mean_psd_high_noise_norm, 'b-', linewidth=2,
               label=f'σₘ > 0.2 on [{low_limit_cyclo}-{high_limit_cyclo}Hz] → {n_high_noise} segments')
    ax3.loglog(frequencies, mean_psd_high_1_noise_norm, 'r-', linewidth=2,
               label=f'σₘ < -0.2 on [{low_limit_w}-{high_limit_w}Hz] → {n_high_noise_1} segments')
    ax3.loglog(frequencies, mean_psd_remaining_noise, 'k-', linewidth=2,
               label=f'Remaining segments → {n_remaining_noise}')
    
    # Plot the noise level
    ax3.loglog(frequencies, noise_3_sigma, linestyle="-", color="lime",
               linewidth=2, label='3σ Noise Level')

    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('PSD [(nT)²/Hz]')
    ax3.set_title('Above Noise Segments - Power Spectral Density')
    ax3.grid(True)
    ax3.set_xlim(1e-1, 100)

    # Second right subplot: Filtered Helicity
    ax4 = fig.add_subplot(gs[1, 1])

    hel_high_noise = helicity_segments[high_helicity_noise]
    hel_high_noise_1 = helicity_segments[high_helicity_noise_1]
    hel_remaining_noise = helicity_segments[remaining_segments_noise]

    mean_hel_high_noise = np.mean(hel_high_noise, axis=0)
    mean_hel_high_noise_1 = np.mean(hel_high_noise_1, axis=0)
    mean_hel_remaining_noise = np.mean(hel_remaining_noise, axis=0)

    ax4.semilogx(helicity_freqs, mean_hel_high_noise, 'b-', linewidth=2,
                 label=f'σₘ > 0.2 on [{low_limit_cyclo}-{high_limit_cyclo}Hz] → {n_high_noise} segments')
    ax4.semilogx(helicity_freqs, mean_hel_high_noise_1, 'r-', linewidth=2,
                 label=f'σₘ < -0.2 on [{low_limit_w}-{high_limit_w}Hz] → {n_high_noise_1} segments')
    ax4.semilogx(helicity_freqs, mean_hel_remaining_noise, 'k-', linewidth=2,
                 label=f'Remaining segments → {n_remaining_noise}')

    ax4.set_xlabel('Frequency [Hz]')
    ax4.set_ylabel('Magnetic Helicity')
    ax4.set_title('Above Noise Segments - Magnetic Helicity')
    ax4.grid(True)
    ax4.set_xlim(1e-1, 100)
    ax4.set_ylim(-1, 1)

    # Adjust layout and show the plot
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout(pad=2.0)
    
    # Print statistics
    print("\nAnalysis Statistics:")
    print(f"Total segments: {len(psd_segments)}")
    print(f"Segments above noise: {np.sum(above_noise_mask)}")
    
    if np.any(mask_interval):
        mean_psd = np.nanmean([np.nanmean(psd[mask_interval]) for psd in psd_segments])
        mean_noise = np.nanmean(noise_3_sigma[mask_interval])
        print(f"Mean PSD in the 3-4 Hz band: {mean_psd:.2e}")
        print(f"Mean noise level in the 3-4 Hz band: {mean_noise:.2e}")

    # Fine adjustment of margins
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.15, wspace=0.3, hspace=0.4)

    # Example of adjustment on a sub-plot (repeat for each sub-plot)
    ax1.yaxis.set_label_coords(-0.15, 0.5)  # More significant movement
    ax3.yaxis.set_label_coords(-0.15, 0.5)

    # Moving the x titles
    ax1.xaxis.set_label_coords(0.5, -0.2) 
    ax3.xaxis.set_label_coords(0.5, -0.2)
    ax2.xaxis.set_label_coords(0.5, -0.2)
    ax4.xaxis.set_label_coords(0.5, -0.2)
    
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#       Plot Mean Helicity and PSD over time        #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def create_figure_5(frequencies_helicity, helicity_segments, segment_times, frequencies_psd, psd_segments, low_limit_cyclo, high_limit_cyclo, low_limit_w, high_limit_w):
    """
    ┌──────────────────────────────────────────────────┐
    │ Plot the mean helicity in defined frequency bands│
    │ as a function of time (one point per segment).   │
    │                                                  │
    │ Parameters:                                      │
    │ ▔▔▔▔▔▔                                      │
    │ frequencies_helicity : numpy.ndarray             │
    │     Frequencies for helicity data.               │
    │ helicity_segments : numpy.ndarray                │
    │     Magnetic helicity per segment.               │
    │ segment_times : list or numpy.ndarray            │
    │     Segment center timestamps.                   │
    │ frequencies_psd : numpy.ndarray                  │
    │     Frequencies for PSD data.                    │
    │ psd_segments : numpy.ndarray                     │
    │     PSD values per segment.                      │
    │ low_limit_cyclo : float                          │
    │     Cyclotron band lower limit [Hz].             │
    │ high_limit_cyclo : float                         │
    │     Cyclotron band upper limit [Hz].             │
    │ low_limit_w : float                              │
    │     Whistler band lower limit [Hz].              │
    │ high_limit_w : float                             │
    │     Whistler band upper limit [Hz].              │
    │                                                  │
    │ Returns:                                         │
    │ ▔▔▔▔                                          │
    │ None                                             │
    │     Displays two subplots:                       │
    │     - Mean magnetic helicity vs time.            │
    │     - Power Spectral Density vs time.            │
    └──────────────────────────────────────────────────┘
    """
    # Set font sizes with increased values
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18  # Modified sizes
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # Increased x-tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # Increased y-tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)

    bands = [(low_limit_cyclo, high_limit_cyclo), (low_limit_w, high_limit_w)]  # Define frequency bands

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # First subplot: Mean Magnetic Helicity
    for f_min, f_max in bands:
        band_indices = np.where((frequencies_helicity >= f_min) & (frequencies_helicity <= f_max))[0]
        mean_helicity = np.mean(helicity_segments[:, band_indices], axis=1)
        axes[0].plot(segment_times[:len(mean_helicity)], mean_helicity, label=f'{f_min}-{f_max} Hz')
    
    axes[0].set_ylabel('Mean Magnetic Helicity', fontsize=MEDIUM_SIZE)
    axes[0].set_title('Magnetic Helicity in Frequency Bands vs Time')
    axes[0].legend()
    axes[0].grid(True)
    
    # Second subplot: Power Spectral Density (PSD)
    for f_min, f_max in bands:
        band_indices = np.where((frequencies_psd >= f_min) & (frequencies_psd <= f_max))[0]
        mean_psd = np.mean(psd_segments[:, band_indices], axis=1)
        axes[1].plot(segment_times[:len(mean_psd)], mean_psd, label=f'{f_min}-{f_max} Hz')
    
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Time', fontsize=MEDIUM_SIZE)
    axes[1].set_ylabel('Power Spectral Density (PSD)', fontsize=MEDIUM_SIZE)
    axes[1].legend()
    axes[1].grid(True)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


     #╔════════════════════════════════════════════════════════════════════════════╗#
     #║                           FONCTIONS SELECTION                              ║#
     #╚════════════════════════════════════════════════════════════════════════════╝#


  #    ---  Figure of Magnetic Field Evolution  ---
create_figure_1(df_filtered)

  #    ---  Figure of PSD  ---
create_figure_2(frequencies_psd, psd_segments, noise_3_sigma, low_filtred_limit, high_filtred_limit)

  #    ---  Spectrograms of PSD and Helicity  ---
create_figure_3(df_filtered, frequencies_psd, psd_segments, 
               frequencies_helicity, helicity_segments, theta_BR, window_size=30)

  #    ---  Figure of Helicity with filtered segments  ---
create_figure_4(frequencies_psd, psd_segments, frequencies_helicity, helicity_segments, noise_3_sigma, low_limit_cyclo, high_limit_cyclo, low_limit_w, high_limit_w, low_filtred_limit, high_filtred_limit)

  #    ---  Figure of mean Helicity and PSD vs Time  ---
create_figure_5(frequencies_helicity, helicity_segments, segment_times, frequencies_psd, psd_segments, low_limit_cyclo, high_limit_cyclo, low_limit_w, high_limit_w)

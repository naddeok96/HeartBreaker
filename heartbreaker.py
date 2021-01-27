# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, argrelmin
from scipy import signal as scisig
import math
import csv
from patient import Patient
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import obspy.signal.filter
from scipy.signal import filtfilt, butter
import xlwt 
from xlwt import Workbook 
import random
from peaks import Peaks
import pickle
import os

# Heartbreaker Functions
#----------------------------------#
def plot_signal(time, 
                signal,
                title = "", 
                xlabel = 'Time (s)', 
                ylabel = "",
                show_mean = False,
                intervals = None):
    '''
    Plot a signal
    '''
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show_mean == True:
        plt.plot(time, [np.mean(signal)] * len(time))

    if intervals is not None:
        for interval in intervals:
            if intervals[interval][1] == "end":
                intervals[interval][1] = np.max(time)

            facecolor = 'g'
            if interval == 7:
                facecolor = 'r'
            plt.axvspan(int(intervals[interval][0]), int(intervals[interval][1]), facecolor=facecolor, alpha=0.25)

    plt.show()

def get_fft(time, 
                    signal, 
                    plot = False,
                    save = False,
                    patient_name = None):
    '''
    Performs the fourier transform on the signal
    '''

    # Frequency Domain
    time_step = (np.max(time) - np.min(time))/ (len(time) - 1)
    freqs = np.fft.fftfreq(len(signal), d = time_step)
    amps  = np.fft.fft(signal)


    # Save
    if save == True:
        with open(patient_name + '_ecg_freq.csv', 'w') as f:
            for freq, amp in zip(freqs,amps):
                f.write("%f,%f\n"%(freq, np.abs(amp)))

    # Display
    if plot == True:    
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.plot(time, signal)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Signal amplitude')
        ax1.set_xlim(min(time), max(time))
        ax1.set_ylim(min(signal) - (0.2 * abs(min(signal))), max(signal) * 1.2)

        ax2.stem(freqs, np.abs(amps), use_line_collection = True)
        ax2.set_xlabel('Frequency in Hertz [Hz]')
        ax2.set_ylabel('Frequency Domain (Spectrum) Magnitude')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-1, max(np.abs(amps)))
        plt.show()
    
    return freqs, amps

def get_spectrum(time,
                        signal):
    '''
    Plot the spectrogram of the signal
    '''
    # Calculate the frequency
    time_step = (np.max(time) - np.min(time))/ (len(time) - 1)
    frequency = 1 / time_step

    # Define figure plots
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Plot signal
    ax1.plot(signal)
    ax1.set_ylabel('ECG')

    # Plot spectrogram
    NFFT     = 2**10
    noverlap = 2**6
    Pxx, freqs, bins, im = ax2.specgram(x = signal,
                                        Fs=frequency,
                                        NFFT=NFFT,
                                        noverlap = noverlap)
    ax2.set_ylim((0,100))
    
    plt.show()

def load_file_data(files, folder_name, dosage, file_number):

    # Initalize patient and interval
    # Change directory
    wd = 'data/' + folder_name + '/files_of_interest'

    # Load TDMS file into Patient object
    file_name = files[folder_name][dosage][file_number]["file_name"]
    patient = Patient(wd, file_name)

    # Declare time and signal
    time = patient.times
    signal = patient.ecg
    seis1 = patient.seis1
    seis2 = patient.seis2
    phono1 = patient.phono1
    phono2 = patient.phono2
    os.chdir('../..')
        
    return time, signal, seis1, seis2, phono1, phono2

def load_intervals(folder_name):

    # Load file name
    load_filename = "data/Derived/intervals/Interval_Dict_" + folder_name
    assert os.path.isfile(load_filename + '.pkl'), "Folder does not have preevaluated data. Please make use_intervals = FALSE."

    with open(load_filename + '.pkl', 'rb') as input:
        return pickle.load(input)


def save_peaks_to_excel(file_name, time, peaks):
    '''
    Save PQRST, S''max and T''max peaks in time to excel
    '''
    # Excel Workbook Object is created 
    wb = Workbook() 
    
    # Create sheet
    sheet = wb.add_sheet('Peaks') 

    # Write out each peak and data
    for i, peak in enumerate(peaks):
        sheet.write(0, i, peak)
        for j, value in enumerate(peaks[peak]):
            time_instant = "N/A" if value == 0 else time[value]
            sheet.write(j + 1, i, time_instant)

    wb.save(file_name + '.xls') 
    
def bandpass_filter(time, 
                            signal, 
                            freqmin, 
                            freqmax): 
    '''
    Removes frequencies in an interval of frequencies
    '''
    # Calculate the frequency
    time_step = (np.max(time) - np.min(time))/ (len(time) - 1)
    frequency = 1 / time_step
    return obspy.signal.filter.bandstop(signal,
                                        freqmin,
                                        freqmax,
                                        frequency)

def lowpass_filter(signal, cutoff_freq, time = None):
    '''
    Lowpass filter of the signal to remove frequencies above the cutoff frequency
    '''
    if time is None:
        frequency = 4000
    else:
        # Calculate frequency
        time_step = (np.max(time) - np.min(time))/ (len(time) - 1)
        frequency = 1 / time_step

    # Lowpass filter
    nyq = 0.5 * frequency
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(5 , normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def moving_average(signal, pt) :
    '''
    Calculate moving point average of the signal
    '''
    return np.convolve(signal, np.ones((pt,))/pt, mode='valid')

def normalize(signal):
    '''
    Normalize a signal 
    '''
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std

def peaks_between(start, stop, peaks_dict):
    """Determine if current interval of interest contains 
    the Q and QM peak from a given dictionary 

    Args:
        start (float): start of interval of interest
        stop ([float): end of interval of interest
        peaks_dict (dict): dictionary with two keys ("q" and "qm_seis1")
                            which contain list of equal lengths of corresponding
                            times of peaks

    Returns:
        [tuple(float, float)]: a tuple of the Q and QM peak from a given dictionary
                                contained in the interval of interest
    """
    # Check if "q" peak list has any elements in interval of interest
    peaks_between = [start <= x <= stop for x in peaks_dict["q"]]

    if sum(peaks_between) > 0:
        # Find the index of the peak in the interval in the list 
        peak = np.where(peaks_between)[0][0]

        # Return the time of the Q and QM peak
        return peaks_dict["q"][peak], peaks_dict["qm_seis1"][peak]

    else:
        return 0, 0

def get_derivatives(signal,
                            plot = False): 
    '''
    Finds the first and second derivative of a signal
    '''
    # Calculate derivatives
    signal = normalize(signal)
    first  = normalize(np.gradient(signal))
    second = normalize(np.gradient(first))

    # Display
    if plot == True:    
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        ax1.plot(signal)
        ax1.set_ylabel('ECG')

        ax2.plot(first)
        ax2.set_ylabel('First Derivative')

        ax3.plot(second)
        ax3.set_ylabel('Second Derivative')
        
        plt.show()

    return first, second
    
def get_segments(time, signal, max_time_segment):
    '''
    Takes large signal and breaks it into segments
    '''
    # Calculate signal propertys
    initial_time = np.min(time)
    final_time = np.max(time)
    total_time = final_time - initial_time
    data_points = len(time) - 1
    
    time_step = total_time / data_points
    frequency = 1 / time_step    


    # Calculate number of segments
    num_segments = int(np.ceil(total_time / max_time_segment))

    # Find segments
    time_segments = []
    signal_segments = []
    for i in range(num_segments):
        # Determine start and end time of segment
        start_time = int((i * max_time_segment) * frequency)
        end_time   = int((i + 1) * max_time_segment * frequency) if i != num_segments - 1 else len(time)
        print(start_time, end_time)
        
        time_segments.append(time[range(start_time, end_time)])
        signal_segments.append(signal[range(start_time, end_time)])

    return time_segments, signal_segments

def get_peaks_for_composites(time,
                        signal,
                        dosage,

                        random_sample_size = 0,

                        r_max_to_mean_ratio  = 0.5,  ## 0.0 -> mean ## 1.0 -> max
                        qm_max_to_mean_ratio = 0.4,
                        tm_max_to_mean_ratio = 0.4, 

                        r_window_ratio = 0.3,       ## r_window_ratio == 150 beats per min
                        q_window_ratio = 0.08,      ## 
                        s_window_ratio = 0.07,      ## 
                        p_window_ratio = 0.30,      ## 
                        t_window_ratio = 0.45,      ## 
                        
                        qm_bounds    = { 0: [0.2500, 0.3333],
                                        10: [0.1000, 0.1111],
                                        20: [0.0714, 0.0769],
                                        30: [0.0625, 0.0667],
                                        40: [0.0488, 0.0513],
                                        42: [0.0488, 0.0513]},
                        
                        peaks_dict = None,

                        plot                         = False,
                        plot_windows                 = False,
                        plot_derivatives             = False,
                        plot_segmentation_decisons   = False,
                        plot_st_segments             = False,
                        
                        seis1  = None,
                        seis2  = None,
                        phono1 = None,
                        phono2 = None):
    '''
    Finds the P, Q, R, S, T and S-T segments of each beat in a signal
    '''
    # Normalize Signal
    signal = normalize(signal)

    # Normalize Seis
    if (seis1 is not None) and (seis2 is not None):
        seis1  = 0.75 * normalize(seis1)
        seis2  = 0.75 * normalize(seis2)
        seis   = (seis1 + seis2) / 2
    else:
        seis = seis1 if seis1 is not None else seis2
    seis = normalize(seis)
    
    # Normalize Phono
    if (phono1 is not None) and (phono2 is not None):
        phono1 = 0.75 * normalize(phono1)
        phono2 = 0.75 * normalize(phono2)
        phono = (phono1 + phono2) / 2
    else:
        phono = phono1 if phono1 is not None else phono2

    # Initalize Peaks
    peaks = Peaks()
    peaks.time   = time
    peaks.signal = signal
    peaks.seis1  = seis1
    peaks.seis2  = seis2
    peaks.seis   = seis
    peaks.phono1 = phono1
    peaks.phono2 = phono2
    peaks.phono  = phono

    # Calculate frequency    
    frequency =  4000 # (len(time) - 1) / (np.max(time) - np.min(time))

    # Convert to time steps
    qm_bound_indices = [x * frequency for x in qm_bounds[dosage]]    

    # Define Signal and Time range
    first, second = get_derivatives(signal)
    signal_max_to_mean = np.max(signal) - np.mean(signal)

    # Find R Peaks in Interval
    peaks.get_R_peaks(second, r_max_to_mean_ratio, signal_max_to_mean, signal, r_window_ratio, frequency)
    
    # Determine what cutoff freq to use
    cutoff_freq = 15 if np.mean(np.diff(peaks.R.data)) > 2500 else 10

    # Pass through a Low pass
    smoothed_signal = lowpass_filter(time = time, 
                                    signal = signal,
                                    cutoff_freq = cutoff_freq)

    # Calculate second derivative
    smoothed_first, smoothed_second = get_derivatives(smoothed_signal)

    # Generate random list of peaks for trouble shooting
    randomlist = random.sample(range(len(peaks.R.data)), random_sample_size)
     # Use R peak to find other peaks
    for i in range(len(peaks.R.data)):
        peaks.add_Q_peak(    i, q_window_ratio, signal)
        peaks.add_P_peak(    i, p_window_ratio, signal)
        peaks.add_S_peak(    i, s_window_ratio, signal)
        peaks.add_T_peak(    i, t_window_ratio, signal)

        peaks.add_ST_segment(i, second, smoothed_first, signal)
        peaks.add_ddT_peak(  i, smoothed_second)

    if plot:
        peaks.plot(time, signal, seis1, seis2, phono1, phono2)

    return peaks

def temp_get_ecg_peaks(time,
                        signal,
                        dosage,

                        random_sample_size = 0,

                        r_max_to_mean_ratio  = 0.5,  ## 0.0 -> mean ## 1.0 -> max
                        qm_max_to_mean_ratio = 0.4,
                        tm_max_to_mean_ratio = 0.4, 

                        r_window_ratio = 0.3,       ## r_window_ratio == 150 beats per min
                        q_window_ratio = 0.08,      ## 
                        s_window_ratio = 0.07,      ## 
                        p_window_ratio = 0.30,      ## 
                        t_window_ratio = 0.45,      ## 
                        
                        qm_bounds    = { 0: [0.2500, 0.3333],
                                        10: [0.1000, 0.1111],
                                        20: [0.0714, 0.0769],
                                        30: [0.0625, 0.0667],
                                        40: [0.0488, 0.0513]},
                        
                        peaks_dict = None,

                        plot                         = False,
                        plot_windows                 = False,
                        plot_derivatives             = False,
                        plot_segmentation_decisons   = False,
                        plot_st_segments             = False,
                        
                        seis1  = None,
                        seis2  = None,
                        phono1 = None,
                        phono2 = None):
    '''
    Finds the P, Q, R, S, T and S-T segments of each beat in a signal
    '''
    # Normalize
    signal = normalize(signal)
    seis1  = 0.75 * normalize(seis1)
    seis2  = 0.75 * normalize(seis2)
    seis   = (seis1 + seis2) / 2
    phono1 = 0.75 * normalize(phono1)
    phono2 = 0.75 * normalize(phono2)
    phono = (phono1 + phono2) / 2
    # seis1_first, seis1_second = get_derivatives(seis1)

    # Initalize Peaks
    peaks = Peaks()
    peaks.time = time
    peaks.signal = signal
    peaks.seis1 = seis1
    peaks.seis2 = seis2
    peaks.seis  = seis
    peaks.phono1 = phono1
    peaks.phono2 = phono2
    peaks.phono  = phono

    # Calculate frequency    
    frequency =  (len(time) - 1) / (np.max(time) - np.min(time))

    # Convert to time steps
    qm_bound_indices = [x * frequency for x in qm_bounds[dosage]]    

    # Define Signal and Time range
    first, second = get_derivatives(signal)
    signal_max_to_mean = np.max(signal) - np.mean(signal)

    # Find R Peaks in Interval
    peaks.get_R_peaks(second, r_max_to_mean_ratio, signal_max_to_mean, signal, r_window_ratio, frequency)
    
    # Determine what cutoff freq to use
    cutoff_freq = 15 if np.mean(np.diff(peaks.R.data)) > 2500 else 10

    # Pass through a Low pass
    smoothed_signal = lowpass_filter(time = time, 
                                    signal = signal,
                                    cutoff_freq = cutoff_freq)

    # Calculate second derivative
    smoothed_first, smoothed_second = get_derivatives(smoothed_signal)

    # Generate random list of peaks for trouble shooting
    randomlist = random.sample(range(len(peaks.R.data)), random_sample_size)
     # Use R peak to find other peaks
    for i in range(len(peaks.R.data)):
        peaks.add_Q_peak(       i, q_window_ratio, signal)
        peaks.add_P_peak(       i, p_window_ratio, signal)
        peaks.add_S_peak(       i, s_window_ratio, signal)
        peaks.add_T_peak(       i, t_window_ratio, signal)

        peaks.add_ST_segment(   i, second, smoothed_first, signal)
        peaks.add_ddT_max(      i, smoothed_second)

        peaks.add_QM_peak(      i, seis1, qm_max_to_mean_ratio, qm_bound_indices) 
        peaks.add_TM_peak(      i, seis1, tm_max_to_mean_ratio)

        if plot_segmentation_decisons:
            peaks.plot_segmentation_decisons(i, randomlist, time, seis1, signal, qm_bounds, dosage, frequency)
                    
    if plot:
        peaks.plot(time, signal, seis1, seis2, phono1, phono2)

    return peaks

def get_ecg_peaks(  time,
                    signal,
                    dosage,

                    random_sample_size = None,

                    r_max_to_mean_ratio  = 0.5,  ## 0.0 -> mean ## 1.0 -> max
                    qm_max_to_mean_ratio = 0.4,
                    tm_max_to_mean_ratio = 0.4, 

                    r_window_ratio = 0.3,       ## r_window_ratio == 150 beats per min
                    q_window_ratio = 0.08,      ## 
                    s_window_ratio = 0.07,      ## 
                    p_window_ratio = 0.30,      ## 
                    t_window_ratio = 0.45,      ## 
                    
                    qm_bounds    = { 0: [0.2500, 0.3333],
                                    10: [0.1000, 0.1111],
                                    20: [0.0714, 0.0769],
                                    30: [0.0625, 0.0667],
                                    40: [0.0488, 0.0513]},
                    
                    peaks_dict = None,

                    plot                         = False,
                    plot_windows                 = False,
                    plot_derivatives             = False,
                    plot_segmentation_decisons   = False,
                    plot_st_segments             = False,
                    
                    seis1  = None,
                    seis2  = None,
                    phono1 = None,
                    phono2 = None):
    '''
    Finds the P, Q, R, S, T and S-T segments of each beat in a signal
    '''

    # Normalize
    signal = normalize(signal)
    seis1  = 0.5 * normalize(seis1)
    seis1_first, seis1_second = get_derivatives(seis1)

    # Calculate frequency
    initial_time = np.min(time)
    final_time = np.max(time)
    total_time = final_time - initial_time
    data_points = len(time) - 1
    
    time_step = total_time / data_points
    frequency = 1 / time_step    

    # Convert to time steps
    qm_bound_indices = [x * frequency for x in qm_bounds[dosage]]    

    # Define Signal and Time range
    signal_mean = np.mean(signal)
    signal_max  = np.max(signal)
    signal_min  = np.min(signal)

    signal_max_to_mean = signal_max - signal_mean

    # Find R Peaks in Interval
    first, second = get_derivatives(signal)
    r_peaks = find_peaks(-second,
                            height = (r_max_to_mean_ratio * signal_max_to_mean) + signal_mean,
                            distance = r_window_ratio * frequency)[0] # 0.3 r_window_ratio == 150 beats per min
    
    # Determine what cutoff freq to use
    if np.mean(np.diff(r_peaks)) > 2500:
        cutoff_freq = 15
    else:
        cutoff_freq = 10

    # Pass through a 10Hz low pass
    smoothed_signal = lowpass_filter(time = time, 
                                            signal = signal,
                                            cutoff_freq = cutoff_freq)

    # Calculate second derivative
    smoothed_first, smoothed_second = get_derivatives(smoothed_signal)

    # Initalize other peaks to be the size of R_peaks
    def fill_w_zeros():
        return np.zeros(len(r_peaks)).astype(int)

    q_peaks, s_peaks, p_peaks, t_peaks = fill_w_zeros(), fill_w_zeros(), fill_w_zeros(), fill_w_zeros()
    st_starts, st_ends, t_ddots, qm_seis1_peaks, tm_seis1_peaks = [], [], [], [], []

    # Initalize boundaries
    q_windows, s_windows, p_windows, t_windows, tm_seis1_windows = {}, {}, {}, {}, {}

    # Use R peak to find other peaks
    randomlist = []
    if random_sample_size is not None:
        randomlist = np.sort(random.sample(range(len(r_peaks)), random_sample_size))
    
    for i in range(len(r_peaks)):
        # Do not calculate Q or P peaks if there is no previous R peak
        if i != 0:
            # Find distance between this r peak and the last
            last_r_peak_distance = abs(r_peaks[i] - r_peaks[i - 1])

            # Find lower bound of q and p windows and define windows
            q_lower_bound = r_peaks[i] - int(q_window_ratio * last_r_peak_distance)
            q_windows[i] = list(range(q_lower_bound, r_peaks[i]))
            q_peaks[i] = r_peaks[i] - len(q_windows[i]) + np.argmin(signal[q_windows[i]])
            
            p_lower_bound = q_peaks[i] - int(p_window_ratio * last_r_peak_distance)
            p_windows[i] = list(range(p_lower_bound, q_peaks[i])) 
            p_peaks[i] = int(q_peaks[i] - len(p_windows[i]) + np.argmax(signal[p_windows[i]]))

        # Do not calculate S or T peaks if there is no next R peak
        if i != (len(r_peaks) - 1):
            # Find distance between this r peak and the next r peak
            next_r_peak_distance = abs(r_peaks[i + 1] - r_peaks[i])

            # Find upper bound of s and t peaks and define windows
            s_upper_bound = r_peaks[i] + int(s_window_ratio * next_r_peak_distance)
            s_windows[i] = list(range(r_peaks[i], s_upper_bound))
            if len(find_peaks(-signal[s_windows[i]], distance = len(s_windows[i])/2)[0]) == 0:
                s_peaks[i] = int(np.argmax(-signal[s_windows[i]]) + r_peaks[i])
            else:
                s_peaks[i] = int(find_peaks(-signal[s_windows[i]], distance = len(s_windows[i])/2)[0][0] + r_peaks[i])

            t_upper_bound = s_peaks[i] + int(t_window_ratio * next_r_peak_distance)
            t_windows[i] = list(range(s_peaks[i], t_upper_bound))
            t_peaks[i] = int(np.argmax(signal[t_windows[i]]) + s_peaks[i])

            # Find S-T segment
            #------------------------------------------#
            # Look at interval between s and t peak
            s_t_interval = range(s_peaks[i], t_peaks[i])

            # Find start of S-T segment
            st_start = np.argmin(second[s_t_interval[:int(0.15 * len(s_t_interval))]])

            # Look at interval between st_start and t peak
            st_start_t_interval = range(s_peaks[i] + st_start, t_peaks[i])

            # Find end of S-T segment
            smoothed_first_peaks = find_peaks(smoothed_first[st_start_t_interval], distance = len(st_start_t_interval)/6)[0]
            st_end = int(smoothed_first_peaks[-2]) if len(smoothed_first_peaks) != 1 else smoothed_first_peaks[0]

            # Locate t''max
            smoothed_second_peaks = find_peaks(smoothed_second[st_start_t_interval], distance = len(st_start_t_interval)/6)[0]
            t_ddot = int(smoothed_second_peaks[-1])
            
            # Locate Q-M and T-M peaks in Seis1
            if seis1 is not None:
                # Q-M
                if i == 0:
                    qm_seis1_peak = 0
                else:
                    q_t_interval = range(q_peaks[i], t_peaks[i])

                    qm_seis1_peaks_temp = find_peaks(seis1[q_t_interval],height   = (qm_max_to_mean_ratio * (np.max(seis1[q_t_interval]) - np.mean(seis1[q_t_interval]))) + np.mean(seis1[q_t_interval]),  distance = len(q_t_interval)/8)[0] #,
                    if len(qm_seis1_peaks_temp) == 0:
                        qm_seis1_peaks_temp = find_peaks(seis1[q_t_interval], distance = len(q_t_interval)/8)[0]

                    distances = []
                    for qm_seis1_peak_temp in qm_seis1_peaks_temp:
                        distances.append(np.min([abs(qm_bound_index - qm_seis1_peak_temp) for qm_bound_index in qm_bound_indices]))
                    
                    qm_seis1_peak =  qm_seis1_peaks_temp[np.argmin(distances)]
                        
                # T-M
                tm_seis1_window = range(t_peaks[i], r_peaks[i + 1])
                tm_seis1_peaks_temp = find_peaks(seis1[tm_seis1_window], 
                                                height   = (tm_max_to_mean_ratio * (np.max(seis1[tm_seis1_window]) - np.mean(seis1[tm_seis1_window]))) + np.mean(seis1[tm_seis1_window]),
                                                distance = len(tm_seis1_window)/8)[0]
                tm_seis1_peak = tm_seis1_peaks_temp[0] # if len(tm_seis1_peaks_temp) == 1 else tm_seis1_peaks_temp[1]
                    
            
            # Display decison areas for segmentation
            peak_dict_start, peak_dict_stop = 0, 0
            if peaks_dict is not None:
                peak_dict_start, peak_dict_stop = peaks_between(time[r_peaks[i-1]], time[r_peaks[i+1]], peaks_dict)

            if plot_segmentation_decisons == True and ((i != 0 and i in randomlist) or (peak_dict_start != 0)):
                print("heartbeat: ", i)
                print(peak_dict_start, peak_dict_stop)

                # Plot Signals
                plt.plot(time[r_peaks[i-1]:r_peaks[i+1]], signal[r_peaks[i-1]:r_peaks[i+1]], label='Signal')
                # plt.plot(range(r_peaks[i-1],r_peaks[i+1]), smoothed_first[r_peaks[i-1]:r_peaks[i+1]], label="Smoothed First Derivative")
                # plt.plot(range(r_peaks[i-1],r_peaks[i+1]), smoothed_second[r_peaks[i-1]:r_peaks[i+1]], label='Smoothed Second Derivative')
                if seis1 is not None:
                    plt.plot(time[r_peaks[i-1]: r_peaks[i+1]], seis1[r_peaks[i-1]:r_peaks[i+1]], label='Seis-I')
                    # plt.plot(range(r_peaks[i-1],r_peaks[i+1]), seis1_first[r_peaks[i-1]:r_peaks[i+1]], label ="Seis-I First Derivative")
                    # plt.plot(range(r_peaks[i-1],r_peaks[i+1]), seis1_second[r_peaks[i-1]:r_peaks[i+1]], label ="Seis-I Second Derivative")
                    # plt.plot(range(r_peaks[i-1],r_peaks[i+1]), np.ones(r_peaks[i+1] - r_peaks[i-1]) *  np.mean(seis1[q_t_interval]), label='Q')
                    # plt.plot(range(r_peaks[i-1],r_peaks[i+1]), np.ones(r_peaks[i+1] - r_peaks[i-1]) *  np.mean(seis1[tm_seis1_window]), label='Q')

                if peaks_dict is not None:
                    plt.axvline(peak_dict_start, c = 'r')
                    plt.axvline(peak_dict_stop, c = 'r')

                # Plot S peak
                plt.scatter(time[s_peaks[i]], signal[s_peaks[i]], c = "g")
                plt.text(time[s_peaks[i]], signal[s_peaks[i]] + 0.2, "S", fontsize=9, horizontalalignment = 'center')

                # Plot S-T start
                # plt.axvline(s_peaks[i] + st_start)
                plt.scatter(time[s_peaks[i] + st_start], signal[s_t_interval][st_start], c = "c")
                plt.text(time[s_peaks[i] + st_start], signal[s_t_interval][st_start] + 0.2, "S-T Start", fontsize=9, horizontalalignment = 'center')

                # Plot S-T end
                # plt.axvline(s_peaks[i] + st_start + st_end)
                plt.scatter(time[s_peaks[i] + st_start + st_end], signal[st_start_t_interval][st_end], c = "m")
                plt.text(time[s_peaks[i] + st_start + st_end], signal[st_start_t_interval][st_end] + 0.2, "S-T End", fontsize=9, horizontalalignment = 'center')
                
                # Plot T''max
                # plt.axvline(s_peaks[i] + st_start + t_ddot)
                plt.scatter(time[s_peaks[i] + st_start + t_ddot], signal[st_start_t_interval][t_ddot], c = "r")
                plt.text(time[s_peaks[i] + st_start + t_ddot], signal[st_start_t_interval][t_ddot] + 0.2, "T''max", fontsize=9, horizontalalignment = 'center')

                # Plot Q-M and T-M Seis I
                if seis1 is not None:
                    # Q-M
                    plt.axvline(time[int(q_peaks[i] +  np.ceil(qm_bounds[dosage][0] * frequency))], c = "g")
                    plt.axvline(time[int(q_peaks[i] +  np.ceil(qm_bounds[dosage][1] * frequency))], c = "g")

                    # plt.axvline(q_peaks[i] +  qm_seis1_peak)
                    plt.scatter(time[int(q_peaks[i] + qm_seis1_peak)], seis1[q_t_interval][qm_seis1_peak], c = "g")
                    plt.text(time[int(q_peaks[i] + qm_seis1_peak)], seis1[q_t_interval][qm_seis1_peak] + 0.2, "Q-M-Seis I", fontsize=9, horizontalalignment = 'center')

                    # plt.axvline(t_peaks[i] +  tm_seis1_peak)
                    plt.scatter(time[int(t_peaks[i] + tm_seis1_peak)], seis1[tm_seis1_window][tm_seis1_peak], c = "g")
                    plt.text(time[int(t_peaks[i] + tm_seis1_peak)], seis1[tm_seis1_window][tm_seis1_peak] + 0.2, "T-M-Seis I", fontsize=9, horizontalalignment = 'center')

                plt.legend(loc = 'upper left')
                plt.show()
            
            # Reindex to entire signal not just interval
            st_starts.append(s_peaks[i] + st_start)
            st_ends.append(s_peaks[i] + st_start + st_end)
            t_ddots.append(s_peaks[i] + st_start + t_ddot)
            if seis1 is not None:
                qm_seis1_peaks.append(q_peaks[i] + qm_seis1_peak)
                tm_seis1_peaks.append(t_peaks[i] + tm_seis1_peak)

            # Add a zero at the end
            if i == (len(r_peaks) - 2):
                st_starts.append(0)
                st_ends.append(0)
                t_ddots.append(0)
                if seis1 is not None:
                    qm_seis1_peaks.append(0)
                    tm_seis1_peaks.append(0)

    # Display Results
    #----------------------------------------------------------------#
    if plot == True:
        # Declare figure dimensions
        if plot_derivatives == True:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        else:
            fig, ax1 = plt.subplots()

        # Plot Signal
        ax1.plot(time, signal, color='b', label = "Signal")
        ax1.plot()

        # Plot Seismocardiogram
        if seis1 is not None:
            ax1.plot(time, seis1, color='r', linewidth = 0.5, label = "Seis-I")
        if seis2 is not None:
            ax1.plot(time, seis2, color='r', linewidth=0.5, label = "Seis-II")
        if phono1 is not None:
            ax1.plot(time, 10 * phono1, color='g', linewidth=0.5, label = "Phono-I")
        if phono2 is not None:
            ax1.plot(time, 10 * phono2, color='g', linewidth=0.5, label = "Phono-II")
        ax1.legend(loc = 'upper left')

        # Plot Peaks
        for i in range(len(r_peaks)):

            # R Peaks
            ax1.scatter(time[r_peaks[i]], signal[r_peaks[i]], c='red', marker = "D")
            ax1.text(time[r_peaks[i]], 0.02 + signal[r_peaks[i]], "R", fontsize=9)

            if i != 0:
                # Q Peaks
                ax1.scatter(time[q_peaks[i]], signal[q_peaks[i]], c='green', marker = "D")
                ax1.text(time[q_peaks[i]], 0.02 + signal[q_peaks[i]], "Q", fontsize=9)

                # P Peaks
                ax1.scatter(time[p_peaks[i]], signal[p_peaks[i]], c='blue', marker = "D")
                ax1.text(time[p_peaks[i]], 0.02 + signal[p_peaks[i]], "P", fontsize=9)

                # Plot Windows
                if plot_windows == True:
                    ax1.axvspan(time[np.min(q_windows[i])], time[np.max(q_windows[i])], facecolor='g', alpha=0.25)
                    ax1.axvspan(time[np.min(p_windows[i])], time[np.max(p_windows[i])], facecolor='b', alpha=0.25)

            if i != (len(r_peaks) - 1):
                # S Peaks
                ax1.scatter(time[s_peaks[i]], signal[s_peaks[i]], c='yellow', marker = "D")
                ax1.text(time[s_peaks[i]],  0.02 + signal[s_peaks[i]], "S", fontsize=9)

                # T Peaks
                ax1.scatter(time[t_peaks[i]], signal[t_peaks[i]], c='magenta', marker = "D")
                ax1.text(time[t_peaks[i]], 0.02 + signal[t_peaks[i]], "T", fontsize=9)

                # T''max Peaks
                ax1.scatter(time[t_ddots[i]], signal[t_ddots[i]], c='cyan', marker = "D")
                ax1.text(time[t_ddots[i]], 0.02 + signal[t_ddots[i]], "T''max", fontsize=9)

                # Plot ST Segments
                if plot_st_segments == True:
                    ax1.axvspan(time[st_starts[i]], time[st_ends[i]], facecolor='r', alpha=0.25)
                    ax1.text(time[st_starts[i]],  1.1 * signal_max_to_mean, "S-T Segment", fontsize=9)
                    ax1.text(time[st_starts[i]],  0.5 * signal_max_to_mean, str((abs(st_ends[i] - st_starts[i]))/frequency) + "s", fontsize=9)

                # Plot Q-M and T-M Seis1
                if seis1 is not None:
                    if i != 0:
                        # Q-M
                        ax1.scatter(time[qm_seis1_peaks[i]], seis1[qm_seis1_peaks[i]], c='y', marker = "D")
                        ax1.text(time[qm_seis1_peaks[i]], 0.02 + seis1[qm_seis1_peaks[i]], "Q-M-Seis I", fontsize=9)

                    # T-M
                    ax1.scatter(time[tm_seis1_peaks[i]], seis1[tm_seis1_peaks[i]], c='y', marker = "D")
                    ax1.text(time[tm_seis1_peaks[i]], 0.02 + seis1[tm_seis1_peaks[i]], "T-M-Seis I", fontsize=9)
                
                # Plot Windows
                if plot_windows == True:
                    ax1.axvspan(time[np.min(s_windows[i])], time[np.max(s_windows[i])], facecolor='g', alpha=0.25)
                    ax1.axvspan(time[np.min(t_windows[i])], time[np.max(t_windows[i])], facecolor='b', alpha=0.25)

        # Plot Derivatives
        if plot_derivatives == True:
            first, second = get_derivatives(signal)

            ax1.set_ylabel('ECG')

            ax2.plot(time, first)
            ax2.set_ylabel('First Derivative')

            ax3.plot(time, second)
            ax3.set_ylabel('Second Derivative')
            
        plt.show()

    # Store peak data in a dictionary
    peaks = {"P" : p_peaks,
            "Q" : q_peaks,
            "R" : r_peaks,
            "S" : s_peaks,
            "T" : t_peaks,
            "S-T Start": st_starts,
            "S-T End": st_ends,
            "T''max": t_ddots,
            "Q-M Seis I": qm_seis1_peaks,
            "T-M Seis I": tm_seis1_peaks}

    return peaks

def get_phase_ratios_to_RR_intervals_BETA(peaks, file_name):

    # Distances
    peaks["R"][file_name]["statistics"]          = Variable(np.diff(peaks["R"][file_name]["data"]))
    peaks["S-T"][file_name]["statistics"]        = Variable(peaks["S-T End"][file_name]["data"][:-1]     - peaks["S-T Start"][file_name]["data"][:-1])
    peaks["Q-M Seis I"][file_name]["statistics"] = Variable(peaks["Q-M Seis I"][file_name]["data"][1:-1] - peaks["Q"][file_name]["data"][1:-1])
    peaks["T-M Seis I"][file_name]["statistics"] = Variable(peaks["T-M Seis I"][file_name]["data"][1:-1] - peaks["T"][file_name]["data"][1:-1])

    # Ratios
    for peak in ["P", "Q"]:
        interval = peaks[peak][file_name]["data"] - peaks["R"][file_name]["data"]

        if peak in ["P", "Q"]:
            peaks[peak][file_name]["statistics"] = interval[1:] / peaks["R"][file_name]["statistics"].data

        else:
            peaks[peak][file_name]["statistics"] = interval[:-1] / peaks["R"][file_name]["statistics"].data
    
    return peaks

def get_phase_ratios_to_RR_intervals(p, q, r, s, t, st_starts, st_ends, t_ddots, qm_seis1, tm_seis1):
    '''
    Takes the peaks for an interval and finds the average ratios from the r peaks

    For example rp_ratio is:
    ((time of p peak) - (time of current r peak)) / ((time of next r peak) - (time of current r peak))
    '''
    # Calculate peak intervals
    rp_intervals = p - r
    rq_intervals = q - r
    rr_intervals = np.diff(r)
    rs_intervals = s - r
    rt_intervals = t - r
    rtddot_intervals = t_ddots - r
    qm_seis1_intervals = qm_seis1[1:-1] - q[1:-1]
    tm_seis1_intervals = tm_seis1[1:-1] - t[1:-1]

    # Caluculate ratios
    rp_ratios = rp_intervals[1:]  / rr_intervals
    rq_ratios = rq_intervals[1:]  / rr_intervals
    rs_ratios = rs_intervals[:-1] / rr_intervals
    rt_ratios = rt_intervals[:-1] / rr_intervals
    rtddot_ratios = rtddot_intervals[:-1] / rr_intervals

    # Calculate st segment lengths
    st_seg = st_ends[:-1] - st_starts[:-1]

    # Calculate mean and std
    rr_mean, rr_std = np.mean(rr_intervals), np.std(rr_intervals)
    rp_mean, rp_std = np.mean(rp_ratios), np.std(rp_ratios)
    rq_mean, rq_std = np.mean(rq_ratios), np.std(rq_ratios)
    rs_mean, rs_std = np.mean(rs_ratios), np.std(rs_ratios)
    rt_mean, rt_std = np.mean(rt_ratios), np.std(rt_ratios)
    st_seg_mean, st_seg_std = np.mean(st_seg), np.std(st_seg)
    rtddot_mean, rtddot_std = np.mean(rtddot_ratios), np.std(rtddot_ratios)
    qm_seis1_mean, qm_seis1_std = np.mean(qm_seis1_intervals), np.std(qm_seis1_intervals)
    tm_seis1_mean, tm_seis1_std = np.mean(tm_seis1_intervals), np.std(tm_seis1_intervals)

    phase_ratios = (rp_mean, rp_std, rq_mean, rq_std, rr_mean, rr_std, rs_mean, rs_std, rt_mean, rt_std, st_seg_mean, st_seg_std, rtddot_mean, rtddot_std, qm_seis1_mean, qm_seis1_std, tm_seis1_mean, tm_seis1_std)

    return phase_ratios

def add_stats(mean1, std1, weight1, mean2, std2, weight2):
    '''
    Takes stats of two sets (assumed to be from the same distribution) and combines them
    Method from https://www.statstodo.com/CombineMeansSDs_Pgm.php
    '''
    # Calculate E[x] and E[x^2] of each
    sig_x1 = weight1 * mean1
    sig_x2 = weight2 * mean2

    sig_xx1 = ((std1 ** 2) * (weight1 - 1)) + (((sig_x1 ** 2) / weight1))
    sig_xx2 = ((std2 ** 2) * (weight2 - 1)) + (((sig_x2 ** 2) / weight2))

    # Calculate sums
    tn  = weight1 + weight2
    tx  = sig_x1  + sig_x2
    txx = sig_xx1 + sig_xx2

    # Calculate combined stats
    mean = tx / tn
    std = np.sqrt((txx - (tx**2)/tn) / (tn - 1))

    return mean, std, tn

def plot_phase_space(file_name,
                        rp_mean, rp_std,
                        rq_mean, rq_std,
                        rs_mean, rs_std,
                        rt_mean, rt_std):
    '''
    Plots the phase space of peaks wrt R-R intervals
    '''
    # Display Circle
    fig, ax = plt.subplots(1)
    plt.title(file_name)
    ax.axis('off')
    circle = plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None')
    ax.add_patch(circle)
    ax.set_aspect(1)
    plt.xlim(-1.25,1.25)
    plt.ylim(-1.25,1.25)

    # Add points
    ax.plot([0, 1], [0, 0], c='red')
    plt.text(1.05, 0, "R", fontsize=12)

    ax.plot([0, math.cos(np.pi * rp_mean)], [0, math.sin(np.pi * rp_mean)], c='blue', linewidth=2)
    ax.plot([0, math.cos(np.pi * (rp_mean + rp_std))], [0, math.sin(np.pi * (rp_mean + rp_std))], c='blue', linestyle='dashed', linewidth=1)
    ax.plot([0, math.cos(np.pi * (rp_mean - rp_std))], [0, math.sin(np.pi * (rp_mean - rp_std))], c='blue', linestyle='dashed', linewidth=1)
    plt.text(1.05 * math.cos(np.pi * rp_mean), 1.2 * math.sin(np.pi * rp_mean), "P: " + str(round(rp_mean,3)) + " +/- " + str(round(rp_std,3)), fontsize=12)

    ax.plot([0, math.cos(np.pi * rq_mean)], [0, math.sin(np.pi * rq_mean)], c='green', linewidth=2)
    ax.plot([0, math.cos(np.pi * (rq_mean + rq_std))], [0, math.sin(np.pi * (rq_mean + rq_std))], c='green', linestyle='dashed', linewidth=1)
    ax.plot([0, math.cos(np.pi * (rq_mean - rq_std))], [0, math.sin(np.pi * (rq_mean - rq_std))], c='green', linestyle='dashed', linewidth=1)
    plt.text(1.05 * math.cos(np.pi * rq_mean), 1.2 * math.sin(np.pi * rq_mean), "Q: " + str(round(rq_mean,3)) + " +/- " + str(round(rq_std,3)), fontsize=12)

    ax.plot([0, math.cos(np.pi * rs_mean)], [0, math.sin(np.pi * rs_mean)], c='yellow', linewidth=2)
    ax.plot([0, math.cos(np.pi * (rs_mean + rs_std))], [0, math.sin(np.pi * (rs_mean + rs_std))], c='yellow', linestyle='dashed', linewidth=1)
    ax.plot([0, math.cos(np.pi * (rs_mean - rs_std))], [0, math.sin(np.pi * (rs_mean - rs_std))], c='yellow', linestyle='dashed', linewidth=1)
    plt.text(1.05 * math.cos(np.pi * rs_mean), 1.2 * math.sin(np.pi * rs_mean), "S: " + str(round(rs_mean,3)) + " +/- " + str(round(rs_std,3)), fontsize=12)

    ax.plot([0, math.cos(np.pi * rt_mean)], [0, math.sin(np.pi * rt_mean)], c='magenta', linewidth=2)
    ax.plot([0, math.cos(np.pi * (rt_mean + rt_std))], [0, math.sin(np.pi * (rt_mean + rt_std))], c='magenta', linestyle='dashed', linewidth=1)
    ax.plot([0, math.cos(np.pi * (rt_mean - rt_std))], [0, math.sin(np.pi * (rt_mean - rt_std))], c='magenta', linestyle='dashed', linewidth=1)
    plt.text(1.05 * math.cos(np.pi * rt_mean), 1.1 * math.sin(np.pi * rt_mean), "T: " + str(round(rt_mean,3)) + " +/- " + str(round(rt_std,3)), fontsize=12)

    plt.show()

def plot_phase_space_for_dosages(folder_name, phase_data):
    '''
    Takes the phase ratios for each dosage and plots it
    '''
    dosages = ["0", "10", "20", "30", "40"]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(234)
    ax4 = fig.add_subplot(235)
    ax5 = fig.add_subplot(236)

    fig.suptitle(folder_name)
    plt.xlim(-1.25,1.25)
    plt.ylim(-1.25,1.25)
    count = 0
    for ax, phase_ratios in zip((ax1, ax2, ax3, ax4, ax5), phase_data):

        rp_mean, rp_std, rq_mean, rq_std, rr_mean, rr_std, rs_mean, rs_std, rt_mean, rt_std, st_mean, st_std, rtddot_mean, rtddot_std, qm_seis1_mean, qm_seis1_std, tm_seis1_mean, tm_seis1_std, num_data_points = phase_ratios

        ax.set_title(dosages[count] + ' mcg/kg')
        ax.text(-0.8, -0.15, "Derived\nfrom " + str(num_data_points) + "\nheartbeats", fontsize=10)
        font_size = 10
        ax.axis('off')
        ax.add_patch(plt.Circle((0, 0), radius=1, edgecolor='k', facecolor='None'))
        ax.set_aspect(1)
        ax.set_xlim(-1.25,1.25)
        ax.set_ylim(-1.25,1.25)

        # Add points
        # R
        ax.plot([0, 1], [0, 0], c='red')
        ax.text(1.05, 0, "R: " + str(round(60 / rr_mean - rr_std,3)) + " - " + str(round(60 / rr_mean + rr_std,3)) + " [bpm]", fontsize=font_size)

        # P
        ax.plot([0, math.cos(np.pi * rp_mean)], [0, math.sin(np.pi * rp_mean)], c='blue', linewidth=2)
        ax.plot([0, math.cos(np.pi * (rp_mean + rp_std))], [0, math.sin(np.pi * (rp_mean + rp_std))], c='blue', linestyle='dashed', linewidth=1)
        ax.plot([0, math.cos(np.pi * (rp_mean - rp_std))], [0, math.sin(np.pi * (rp_mean - rp_std))], c='blue', linestyle='dashed', linewidth=1)
        ax.text(1.05 * math.cos(np.pi * rp_mean), 1.2 * math.sin(np.pi * rp_mean), "P: " + str(round(rp_mean,3)) + " +/- " + str(round(rp_std,3)), fontsize=font_size)

        # Q
        ax.plot([0, math.cos(np.pi * rq_mean)], [0, math.sin(np.pi * rq_mean)], c='green', linewidth=2)
        ax.plot([0, math.cos(np.pi * (rq_mean + rq_std))], [0, math.sin(np.pi * (rq_mean + rq_std))], c='green', linestyle='dashed', linewidth=1)
        ax.plot([0, math.cos(np.pi * (rq_mean - rq_std))], [0, math.sin(np.pi * (rq_mean - rq_std))], c='green', linestyle='dashed', linewidth=1)
        ax.text(1.05 * math.cos(np.pi * rq_mean), 1.2 * math.sin(np.pi * rq_mean), "Q: " + str(round(rq_mean,3)) + " +/- " + str(round(rq_std,3)), fontsize=font_size)

        # S
        ax.plot([0, math.cos(np.pi * rs_mean)], [0, math.sin(np.pi * rs_mean)], c='yellow', linewidth=2)
        ax.plot([0, math.cos(np.pi * (rs_mean + rs_std))], [0, math.sin(np.pi * (rs_mean + rs_std))], c='yellow', linestyle='dashed', linewidth=1)
        ax.plot([0, math.cos(np.pi * (rs_mean - rs_std))], [0, math.sin(np.pi * (rs_mean - rs_std))], c='yellow', linestyle='dashed', linewidth=1)
        ax.text(1.05 * math.cos(np.pi * rs_mean), 1.2 * math.sin(np.pi * rs_mean), "S: " + str(round(rs_mean,3)) + " +/- " + str(round(rs_std,3)), fontsize=font_size)

        # T
        ax.plot([0, math.cos(np.pi * rt_mean)], [0, math.sin(np.pi * rt_mean)], c='magenta', linewidth=2)
        ax.plot([0, math.cos(np.pi * (rt_mean + rt_std))], [0, math.sin(np.pi * (rt_mean + rt_std))], c='magenta', linestyle='dashed', linewidth=1)
        ax.plot([0, math.cos(np.pi * (rt_mean - rt_std))], [0, math.sin(np.pi * (rt_mean - rt_std))], c='magenta', linestyle='dashed', linewidth=1)
        ax.text(1.05 * math.cos(np.pi * rt_mean), 1.1 * math.sin(np.pi * rt_mean), "T: " + str(round(rt_mean,3)) + " +/- " + str(round(rt_std,3)), fontsize=font_size)
        
        # S-T
        between_t_and_s = (((rt_mean - rt_std) - (rs_mean + rs_std))/2) + (rs_mean + rs_std)
        # ax.plot([0, math.cos(np.pi * between_t_and_s)], [0, math.sin(np.pi * between_t_and_s)], c='orange', linewidth=2)
        ax.text(1.05 * math.cos(np.pi * between_t_and_s), 0.8 * math.sin(np.pi * between_t_and_s), "ST Segment\n    Duration [s]: " + str(round(st_mean,3)) + " +/- " + str(round(st_std,3)), fontsize=font_size)
        
        # T''max
        ax.plot([0, math.cos(np.pi * rtddot_mean)], [0, math.sin(np.pi * rtddot_mean)], c='cyan', linewidth=2)
        ax.plot([0, math.cos(np.pi * (rtddot_mean + rtddot_std))], [0, math.sin(np.pi * (rtddot_mean + rtddot_std))], c='cyan', linestyle='dashed', linewidth=1)
        ax.plot([0, math.cos(np.pi * (rtddot_mean - rtddot_std))], [0, math.sin(np.pi * (rtddot_mean - rtddot_std))], c='cyan', linestyle='dashed', linewidth=1)
        ax.text(1.05 * math.cos(np.pi * rtddot_mean), 1.1 * math.sin(np.pi * rtddot_mean), "T''max: " + str(round(rtddot_mean,3)) + " +/- " + str(round(rtddot_std,3)), fontsize=font_size)
        
        # Q-M seis1
        ax.text(0, -1.1, "(E-M)ino Q-Seis I: " + str(round(qm_seis1_mean,3)) + " +/- " + str(round(qm_seis1_std,3)) + " [s]", fontsize=font_size, ha="center", va="center")
        
        # T-M seis1
        ax.text(0, -1.3, "(E-M)lusi T-Seis I: " + str(round(tm_seis1_mean,3)) + " +/- " + str(round(tm_seis1_std,3)) + " [s]", fontsize=font_size, ha="center", va="center")
        
        count += 1

    plt.show()

def plot_em_vs_strain_rate(dosages, qm_seises, tm_seises, ino_strains, lusi_strains):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Plot 1/(E-M)ino Q-Seis I vs Dosage
    ax1.scatter(1/np.asarray(qm_seises), dosages)
    # ax1.set_xticks(np.arange(2, 24, step=2))
    # ax1.minorticks_on()
    ax1.set_title("INO")
    ax1.set_xlabel("1/(E-M)ino Q-Seismo I")
    ax1.set_ylabel("Dobutamine Infusion (mcg/kg/min)")
    ax1.grid()

    # Plot 1/(E-M)lusi Q-Seis I vs Dosage
    ax2.scatter(1/np.asarray(tm_seises), dosages)
    # ax2.set_xticks(np.arange(2, 24, step=2))
    # ax2.minorticks_on()
    ax2.set_title("LUSI")
    ax2.set_xlabel("1/(E-M)lusi T-Seismo I")
    ax2.set_ylabel("Dobutamine Infusion (mcg/kg/min)")
    ax2.grid()

    plt.show()

def get_composite_dataset(time, signal, peaks, num_signals, slide_step_size):

    for T_peak in peaks.T:
        print(T_peak)

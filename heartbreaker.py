# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, argrelmin
from scipy import signal as scisig
import csv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import obspy.signal.filter
from scipy.signal import filtfilt, butter
import xlwt 
from xlwt import Workbook 

# Heartbreaker
class HeartBreak:
    '''
    Holds many functions to munipulate heart signals
    '''
    def __init__(self):
        super(HeartBreak, self).__init__

    def save_peaks_to_excel(self, file_name, time, peaks):
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

    def plot_signal(self, time, 
                          signal, 
                          title = "", 
                          xlabel = 'Time (s)', 
                          ylabel = ""):
        '''
        Plot a signal
        '''
        plt.plot(time, signal)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def get_fft(self, time, 
                      signal, 
                      plot = False,
                      save = False):
        '''
        Performs the fourier transform on the signal
        '''

        # Frequency Domain
        time_step = (np.max(time) - np.min(time))/ (len(time) - 1)
        freqs = np.fft.fftfreq(len(signal), d = time_step)
        amps  = np.fft.fft(signal)


        # Save
        if save == True:
            with open(self.patient_name + '_ecg_freq.csv', 'w') as f:
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

    def get_spectrum(self, time,
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
        

    def bandpass_filter(self, time, 
                              signal, 
                              freqmin, 
                              freqmax): 
        '''
        Removes frequencies in an interval of frequencies
        '''
        time_step = (np.max(time) - np.min(time))/ (len(time) - 1)
        frequency = 1 / time_step
        return obspy.signal.filter.bandstop(signal, 
                                            freqmin, 
                                            freqmax, 
                                            frequency)

    def lowpass_filter(self, time, signal, cutoff_freq):
        time_step = (np.max(time) - np.min(time))/ (len(time) - 1)
        frequency = 1 / time_step

        nyq = 0.5 * frequency
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(5 , normal_cutoff, btype='low', analog=False)
        
        return filtfilt(b, a, signal)

    def moving_average(self, signal, pt) :
        '''
        Calculate moving average of signal
        '''
        return np.convolve(signal, np.ones((pt,))/pt, mode='valid')

    def normalize(self, signal):
        '''
        Normalize a signal 
        '''
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / std

    def get_derivatives(self, signal,
                              plot = False): 
        '''
        Finds the first and second derivative of a signal
        '''
        # Calculate derivatives
        signal = self.normalize(signal)
        first  = self.normalize(np.gradient(signal))
        second = self.normalize(np.gradient(first))

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
        
    def get_segments(self, time, signal, max_time_segment):
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

    def get_ecg_peaks(self, time,
                            signal,

                            r_max_to_mean_ratio = 0.5,  ## 0.0 -> mean ## 1.0 -> max

                            r_window_ratio = 0.3,       ## 0.0 -> R peak ## 1.0 -> last or next R peak
                            q_window_ratio = 0.15,      ## 
                            s_window_ratio = 0.15,      ## 
                            p_window_ratio = 0.3,       ## 
                            t_window_ratio = 0.4,       ## 

                            plot             = False,
                            plot_windows     = False,
                            plot_derivatives = False,
                            plot_st_segments = False):
        '''
        Finds the P, Q, R, S, T and S-T segments of each beat in a signal
        '''
        # Normalize
        signal = self.normalize(signal)

        # Calculate frequency
        initial_time = np.min(time)
        final_time = np.max(time)
        total_time = final_time - initial_time
        data_points = len(time) - 1
        
        time_step = total_time / data_points
        frequency = 1 / time_step    

        # Define Signal and Time range
        signal_mean = np.mean(signal)
        signal_max  = np.max(signal)
        signal_min  = np.min(signal)

        signal_max_to_mean = signal_max - signal_mean

        # Find R Peaks in Interval
        r_peaks = find_peaks(signal,
                             height = (r_max_to_mean_ratio * signal_max_to_mean) + signal_mean,
                             distance = r_window_ratio * frequency)[0] # 0.3 r_window_ratio == 150 beats per min

        # Determine what cutoff freq to use
        if np.mean(np.diff(r_peaks)) > 2500:
            cutoff_freq = 15
        else:
            cutoff_freq = 10

        # Pass through a 10Hz low pass
        smoothed_signal = self.lowpass_filter(time = time, 
                                                signal = signal,
                                                cutoff_freq = cutoff_freq)

        # Calculate second derivative
        _, smoothed_second = self.get_derivatives(smoothed_signal)

        # Initalize other peaks to be the size of R_peaks
        def fill_w_zeros():
            return np.zeros(len(r_peaks)).astype(int)

        q_peaks = fill_w_zeros()
        s_peaks = fill_w_zeros()
        p_peaks = fill_w_zeros()
        t_peaks = fill_w_zeros()
        s_ddot = []
        t_ddot = []

        # Initalize boundaries
        q_windows = {} 
        s_windows = {}
        p_windows = {}
        t_windows = {}
        
        # Use R peak to find other peaks
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
                s_peaks[i] = int(np.argmin(signal[s_windows[i]]) + r_peaks[i])

                t_upper_bound = s_peaks[i] + int(t_window_ratio * next_r_peak_distance)
                t_windows[i] = list(range(s_peaks[i], t_upper_bound))
                t_peaks[i] = int(np.argmax(signal[t_windows[i]]) + s_peaks[i])

                # Find S-T segment

                # Look at interval between s and t peak
                interval = range(s_peaks[i],t_peaks[i])

                # Find s''max and t''max peaks
                st_peaks = find_peaks(smoothed_second[interval],distance = len(interval)/3)[0]

                # Calculate the values
                values = smoothed_second[st_peaks]

                # If there are more than two pick the two largest
                if len(values) > 2:
                    idx_peaks = np.argpartition(values, 2)
                    lower = values[idx_peaks[-2:]][0]
                    upper = values[idx_peaks[-2:]][1]
                # Else the biggest is s''max
                else:
                    lower = np.max(values)
                    upper = np.min(values)

                # Reindex to entire signal not just interval
                s_ddot.append(int(np.where(smoothed_second == lower)[0] + s_peaks[i]))
                t_ddot.append(int(np.where(smoothed_second == upper)[0] + s_peaks[i]))

                # Add a zero at the end
                if i == (len(r_peaks) - 2):
                    s_ddot.append(0)
                    t_ddot.append(0)

        # Display Results
        if plot == True:
            # Declare figure dimensions
            if plot_derivatives == True:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            else:
                fig, ax1 = plt.subplots()

            # Plot Peaks
            for i in range(len(r_peaks)):
                # Plot Signal
                ax1.plot(time, signal)
                ax1.plot()

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
                    ax1.scatter(time[s_peaks[i]], signal[s_peaks[i]], c='green', marker = "D")
                    ax1.text(time[s_peaks[i]],  0.02 + signal[s_peaks[i]], "S", fontsize=9)

                    # T Peaks
                    ax1.scatter(time[t_peaks[i]], signal[t_peaks[i]], c='blue', marker = "D")
                    ax1.text(time[t_peaks[i]], 0.02 + signal[t_peaks[i]], "T", fontsize=9)

                    # Plot ST Segments
                    if plot_st_segments == True:
                        ax1.axvspan(time[s_ddot[i]], time[t_ddot[i]], facecolor='r', alpha=0.25)
                        ax1.text(time[s_ddot[i]],  1.1 * signal_max_to_mean, "S-T Segment", fontsize=9)
                        ax1.text(time[s_ddot[i]],  0.5 * signal_max_to_mean, str((abs(t_ddot[i] - s_ddot[i]))/frequency) + "s", fontsize=9)
                    
                    # Plot Windows
                    if plot_windows == True:
                        ax1.axvspan(time[np.min(s_windows[i])], time[np.max(s_windows[i])], facecolor='g', alpha=0.25)
                        ax1.axvspan(time[np.min(t_windows[i])], time[np.max(t_windows[i])], facecolor='b', alpha=0.25)

            # Plot Derivatives
            if plot_derivatives == True:
                first, second = self.get_derivatives(signal)

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
                 "S''max": s_ddot,
                 "T''max": t_ddot}

        return peaks




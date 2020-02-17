'''
This class will break a ECG signal into heartbeats
'''
# Imports
from nptdms import TdmsFile as tdms
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import csv
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import obspy.signal.filter

# Heartbreaker
class HeartBreak:

    def __init__(self, working_directory,
                       file_name):
        
        super(HeartBreak, self).__init__

        os.chdir(working_directory)
        self.patient_name = file_name
        self.tdms_file = tdms(file_name + ".tdms")
        os.chdir("../Derived")

        self.times  = self.tdms_file.object('Data','Time(S)').data
        self.seis1  = self.tdms_file.object('Data','Seismocardiogram I').data
        self.seis2  = self.tdms_file.object('Data',' Seismocardiogram II').data
        self.phono1 = self.tdms_file.object('Data',' Precordial Phonocardiogram-I').data
        self.phono2 = self.tdms_file.object('Data',' Precordial Phonocardiogram-II').data
        self.ecg    = self.tdms_file.object('Data',' Direct ECG').data

        self.num_data_points = len(self.times)
        self.frequency = 2000

    def plot_seis1(self, interval = None):
        interval = range(len(self.times)) if interval == None else interval

        plt.plot(self.times[interval], self.seis1[interval])
        plt.title('Seismocardiogram I')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.show()

    def plot_seis2(self, interval = None):
        interval = range(len(self.times)) if interval == None else interval

        plt.plot(self.times[interval], self.seis2[interval])
        plt.title('Seismocardiogram II')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.show()

    def plot_phono1(self, interval = None):
        interval = range(len(self.times)) if interval == None else interval

        plt.plot(self.times[interval], self.phono1[interval])
        plt.title('Precordial Phonocardiogram-I')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement')
        plt.show()

    def plot_phono2(self, interval = None):
        interval = range(len(self.times)) if interval == None else interval

        plt.plot(self.times[interval], self.phono2[interval])
        plt.title('Precordial Phonocardiogram-II')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement')
        plt.show()

    def plot_ecg(self, interval = None):
        interval = range(len(self.times)) if interval == None else interval

        plt.plot(self.times[interval], self.ecg[interval])
        plt.title('Electrocardiogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Potential')
        plt.show()

    def plot_all(self, interval = None):
        '''
        Plots all signals within a given interval.
        '''
        interval = range(len(self.times)) if interval == None else interval

        plt.figure(figsize = (10, 10))

        plt.subplot(321)
        f1 = plt.plot(self.times[interval], self.seis1[interval])
        plt.title('Seismocardiogram I')
        plt.ylabel('Acceleration')
        plt.axis('off')

        plt.subplot(322)
        plt.plot(self.times[interval], self.seis2[interval])
        plt.title('Seismocardiogram II')
        plt.axis('off')

        plt.subplot(323)
        plt.plot(self.times[interval], self.phono1[interval])
        plt.title('Precordial Phonocardiogram-I')
        plt.ylabel('Displacement')
        plt.axis('off')

        plt.subplot(324)
        plt.plot(self.times[interval], self.phono2[interval])
        plt.title('Precordial Phonocardiogram-II')
        plt.axis('off')

        plt.subplot(313)
        plt.plot(self.times[interval], self.ecg[interval])
        plt.title('Electrocardiogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Potential')
        plt.axis('off')

        plt.show()

    def ecg_fft(self, interval,
                      plot = False,
                      save = True):
        # Initialize
        signal = self.ecg[interval]
        time   = self.times[interval]

        # Frequency Domain
        freqs = np.fft.fftfreq(len(interval), d = 1/self.frequency)
        amps  = np.fft.fft(signal)

        # Save
        if save == True:
            with open(self.patient_name + '_ecg_freq.csv', 'w') as f:
                for freq, amp in zip(freqs,amps):
                    f.write("%f,%f\n"%(freq, np.abs(amp)))

        # Display
        if plot == True:    
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(time, signal)
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Signal amplitude')
            ax1.set_xlim(min(time), max(time))
            ax1.set_ylim(min(signal) * 1.2, max(signal) * 1.2)

            ax2.stem(freqs, np.abs(amps), use_line_collection = True)
            ax2.set_xlabel('Frequency in Hertz [Hz]')
            ax2.set_ylabel('Frequency Domain (Spectrum) Magnitude')
            ax2.set_xlim(0, 100)
            ax2.set_ylim(-1, max(np.abs(amps)))
            plt.show()
        
        return freqs, amps

    def ecg_bandpass_filter(self, freqmin, freqmax): 
        return obspy.signal.filter.bandstop(self.ecg, 
                                            freqmin, 
                                            freqmax, 
                                            self.frequency)

    def ecg_derivatives(self, interval,
                              plot = False): 

        first = np.gradient(self.ecg[interval])
        second = np.gradient(first)

        # Display
        if plot == True:    
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

            ax1.plot(interval, self.ecg[interval])
            ax1.set_ylabel('ECG')

            ax2.plot(interval, first)
            ax2.set_ylabel('First Derivative')

            ax3.plot(interval, second)
            ax3.set_ylabel('Second Derivative')
            
            plt.show()

        return first, second

    def ecg_peaks(self, interval, 
                        plot = False):
        peaks = find_peaks(self.ecg[interval], 
                           height = 0.5,
                           distance = 0.4 * self.frequency)

        print(peaks)
        
        if plot == True:
            plt.scatter(self.times[interval][peaks[0]], self.ecg[interval][peaks[0]], c='green')
            plt.plot(self.times[interval], self.ecg[interval])
            plt.show()

        return peaks

        

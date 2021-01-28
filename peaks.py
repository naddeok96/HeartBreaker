# Imports
import pickle
import numpy as np
import heartbreaker as hb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from variable_statistics import Variable


class Peaks:

    def __init__(self, filename = None):

        super(Peaks, self).__init__()
        
        self.time   = None
        self.signal = None
        self.seis1  = None
        self.seis2  = None
        self.seis   = None
        self.phono1 = None
        self.phono2 = None
        self.phono  = None

        self.P = Variable([])
        self.Q = Variable([])
        self.R = Variable([])
        self.S = Variable([])
        self.T = Variable([])

        self.ST_start = Variable([])
        self.ST_end   = Variable([])

        self.ddT = Variable([])

        self.QM  = Variable([])
        self.QM_seis  = Variable([])
        self.QM_phono = Variable([])
        
        self.TM  = Variable([])
        self.TM_seis  = Variable([])
        self.TM_phono = Variable([])

        if filename is not None:
            self.load(filename)

    def get_R_peaks(self, second, r_max_to_mean_ratio, signal_max_to_mean, signal, r_window_ratio, frequency):

        self.R.data = find_peaks(-second,
                                 height = (r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal),
                                 distance = r_window_ratio * frequency)[0]

    def add_Q_peak(self, index, q_window_ratio, signal):
        # Do not calculate peaks if there is no previous R peak
        if index != 0:
            # Find distance between this r peak and the last
            last_r_peak_distance = abs(self.R.data[index] - self.R.data[index - 1])

            # Find lower bound of Q windows and define windows
            q_lower_bound = self.R.data[index] - int(q_window_ratio * last_r_peak_distance)
            q_window = list(range(q_lower_bound, self.R.data[index]))

            # Add Q peak
            self.Q.data.append(self.R.data[index] - len(q_window) + np.argmin(signal[q_window]))

            # if index == 10:
            #     fig, ax1 = plt.subplots()

            #     # Plot Signal
            #     ax1.plot(time[self.R.data[index-1] : self.R.data[index+1]], signal[self.R.data[index-1] : self.R.data[index+1]], color='b', label = "Signal")
            #     # ax1.plot(time, np.ones(len(time)) * ((r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal)), color='r', label = "Min Height")
                
            #     ax1.scatter(time[self.Q.data[index]], signal[self.Q.data[index]], c='red', marker = "D")
            #     ax1.text(time[self.Q.data[index]], 0.02 + signal[self.Q.data[index]], "Q", fontsize=9)

            #     ax1.axvspan(time[np.min(q_window)], time[np.max(q_window)], facecolor='g', alpha=0.25)
            #     plt.show()
            #     exit()

        else:
            self.Q.data.append(0)

    def add_P_peak(self, index, p_window_ratio, signal):
        # Do not calculate peaks if there is no previous R peak
        if index != 0:
            # Find distance between this r peak and the last
            last_r_peak_distance = abs(self.R.data[index] - self.R.data[index - 1])
            
            # Find lower bound of P windows and define windows
            p_lower_bound = self.R.data[index] - int(p_window_ratio * last_r_peak_distance)
            p_window  = list(range(p_lower_bound, self.Q.data[index])) 

            # Add P peak
            self.P.data.append(int(self.Q.data[index] - len(p_window) + np.argmax(signal[p_window])))

        else:
            self.P.data.append(0)

        # if index == 10:
        #     fig, ax1 = plt.subplots()

        #     # Plot Signal
        #     ax1.plot(time[self.R.data[index-1] : self.R.data[index+1]], signal[self.R.data[index-1] : self.R.data[index+1]], color='b', label = "Signal")
        #     # ax1.plot(time, np.ones(len(time)) * ((r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal)), color='r', label = "Min Height")
            
        #     ax1.scatter(time[self.P.data[index]], signal[self.P.data[index]], c='red', marker = "D")
        #     ax1.text(time[self.P.data[index]], 0.02 + signal[self.P.data[index]], "P", fontsize=9)

        #     ax1.axvspan(time[np.min(p_window)], time[np.max(p_window)], facecolor='g', alpha=0.25)
        #     plt.show()
        #     exit()

    def add_S_peak(self, index, s_window_ratio, signal):
        # Do not calculate peaks if there is no next R peak
        if index != (len(self.R.data) - 1):
            # Find distance between this r peak and the next r peak
            next_r_peak_distance = abs(self.R.data[index + 1] - self.R.data[index])

            # Find upper bound of S peak and define windows
            s_upper_bound = self.R.data[index] + int(s_window_ratio * next_r_peak_distance)
            s_window = list(range(self.R.data[index], s_upper_bound))

            # Look for defined peaks
            possible_defined_peaks = find_peaks(-signal[s_window], distance = len(s_window)/2)[0]

            # If there are no defined peaks just use the max otherwise use the first defined peak
            if len(possible_defined_peaks) == 0:
                self.S.data.append(int(np.argmax(-signal[s_window]) + self.R.data[index]))
            else:
                self.S.data.append(int(possible_defined_peaks[0] + self.R.data[index]))
        else:
            self.S.data.append(0)

    def add_T_peak(self, index, t_window_ratio, signal):
        # Do not calculate peaks if there is no next R peak
        if index != (len(self.R.data) - 1):

            # Find distance between this r peak and the next r peak
            next_r_peak_distance = abs(self.R.data[index + 1] - self.R.data[index])
            
            # Find upper bound of T peak and define windows
            t_upper_bound = self.R.data[index] + int(t_window_ratio * next_r_peak_distance)
            t_window = list(range(self.S.data[index], t_upper_bound))
            
            # Add T peak
            self.T.data.append(int(np.argmax(signal[t_window]) + self.S.data[index]))
        
        else:
            self.T.data.append(0)

    def add_ST_segment(self, index, second, smoothed_first, signal):
        # Do not calculate peaks if there is no next R peak
        if index != (len(self.R.data) - 1):
            # Look at interval between s and t peak
            s_t_interval = range(self.S.data[index], self.T.data[index])

            # Find start of S-T segment
            self.ST_start.data.append(self.S.data[index] + np.argmin(second[s_t_interval[ :int(0.15 * len(s_t_interval))]]))

            # Look at interval between st_start and t peak
            st_start_t_interval = range(self.ST_start.data[index], self.T.data[index])

            # Find end of S-T segment
            smoothed_first_peaks = find_peaks(smoothed_first[st_start_t_interval], distance = len(st_start_t_interval)/6)[0]
          
            if len(smoothed_first_peaks) > 1:
                st_end = int(smoothed_first_peaks[-2])
            elif len(smoothed_first_peaks) == 1:
                st_end = smoothed_first_peaks[0]
            else:
                st_end = ((self.T.data[index] - self.ST_start.data[index])/2) + self.ST_start.data[index]
        
            self.ST_end.data.append(self.ST_start.data[index] + st_end)

        else:
            self.ST_start.data.append(0)
            self.ST_end.data.append(0)

    def add_ddT_peak(self, index, smoothed_second):
        # Do not calculate peaks if there is no next R peak
        if index != (len(self.R.data) - 1):
            # Look at interval between st_start and t peak
            st_start_t_interval = range(self.ST_start.data[index], self.T.data[index])

            # Locate T''max
            smoothed_second_peaks = find_peaks(smoothed_second[st_start_t_interval], distance = len(st_start_t_interval)/6)[0]
            self.ddT.data.append(self.ST_start.data[index] + int(smoothed_second_peaks[-1]))

        else:
            self.ddT.data.append(0)

    def add_QM_peak(self, index, seis, qm_max_to_mean_ratio, qm_bound_indices):
        # Do not calculate peaks if there is no next or previous R peak
        if (index != 0) and (index != (len(self.R.data) - 1)) and (seis is not None):
            # Look at interval between Q and T Peak
            q_t_interval = range(self.Q.data[index], self.T.data[index])

            # Look for peaks at a certain height relative to the mean and max of signal
            q_t_interval_max_to_mean = np.max(seis[q_t_interval]) - np.mean(seis[q_t_interval])
            qm_seis_peaks_temp = find_peaks(seis[q_t_interval], 
                                             height = (qm_max_to_mean_ratio * q_t_interval_max_to_mean) + np.mean(seis[q_t_interval]),
                                             distance = len(q_t_interval)/8)[0]

            # Check if a peak exist, if not drop the height requirement
            if len(qm_seis_peaks_temp) == 0:
                qm_seis_peaks_temp = find_peaks(seis[q_t_interval], distance = len(q_t_interval)/8)[0]

            # Find distance from each peak to bounds set by Bob's labeled data
            distances = []
            for qm_seis_peak_temp in qm_seis_peaks_temp:
                distances.append(np.min([abs(qm_bound_index - qm_seis_peak_temp) for qm_bound_index in qm_bound_indices]))
            
            # Pick the closest
            self.QM.data.append(self.Q.data[index] + qm_seis_peaks_temp[np.argmin(distances)])

        else:
            self.QM.data.append(0)

    def add_TM_peak(self, index, seis, tm_max_to_mean_ratio):
        # Do not calculate peaks if there is no next or previous R peak
        if (index != 0) and (index != (len(self.R.data) - 1)) and (seis is not None):      
            # Look between current T peak and next R peak
            tm_seis_window = range(self.T.data[index], self.R.data[index + 1])

            # Find all defined peaks in the window with a certain height and width
            tm_seis_peaks_temp = find_peaks(seis[tm_seis_window], 
                                            height   = (tm_max_to_mean_ratio * (np.max(seis[tm_seis_window]) - np.mean(seis[tm_seis_window]))) + np.mean(seis[tm_seis_window]),
                                            distance = len(tm_seis_window)/8)[0]

            # Pick the first such peak
            self.TM.data.append(self.T.data[index] + tm_seis_peaks_temp[0])

        else:
            self.TM.data.append(0)

    def plot_segmentation_decisons(self, index, randomlist, time, seis, signal, qm_bounds, dosage, frequency):

        if (index != 0) and (index in randomlist):
            # Display Heartbeat
            print("Heartbeat Number: ", index)

            # Plot Time Signal
            plt.plot(time[self.R.data[index-1]:self.R.data[index+1]], signal[self.R.data[index-1]:self.R.data[index+1]], label='Signal')
            # plt.plot(range(self.R.data[index-1],self.R.data[index+1]), smoothed_first[self.R.data[index-1]:self.R.data[index+1]], label="Smoothed First Derivative")
            # plt.plot(range(self.R.data[index-1],self.R.data[index+1]), smoothed_second[self.R.data[index-1]:self.R.data[index+1]], label='Smoothed Second Derivative')

            # Plot Seis1 Signal
            if seis is not None:
                plt.plot(time[self.R.data[index-1]: self.R.data[index+1]], seis[self.R.data[index-1]:self.R.data[index+1]], label='Seis-I')
                # plt.plot(range(self.R.data[index-1],self.R.data[index+1]), seis_first[self.R.data[index-1]:self.R.data[index+1]], label ="Seis-I First Derivative")
                # plt.plot(range(self.R.data[index-1],self.R.data[index+1]), seis_second[self.R.data[index-1]:self.R.data[index+1]], label ="Seis-I Second Derivative")

            # Plot S peak
            plt.scatter(time[self.S.data[index]], signal[self.S.data[index]], c = "g")
            plt.text(time[self.S.data[index]], signal[self.S.data[index]] + 0.2, "S", fontsize=9, horizontalalignment = 'center')

            # Plot S-T start
            # plt.axvline(self.ST_start.data[index])
            plt.scatter(time[self.ST_start.data[index]], signal[self.ST_start.data[index]], c = "c")
            plt.text(time[self.ST_start.data[index]], signal[self.ST_start.data[index]] + 0.2, "S-T Start", fontsize=9, horizontalalignment = 'center')

            # Plot S-T end
            # plt.axvline(self.ST_end.data[index])
            plt.scatter(time[self.ST_end.data[index]], signal[self.ST_end.data[index]], c = "m")
            plt.text(time[self.ST_end.data[index]], signal[self.ST_end.data[index]] + 0.2, "S-T End", fontsize=9, horizontalalignment = 'center')
            
            # Plot T''max
            # plt.axvline(self.ddT.data[index])
            plt.scatter(time[self.ddT.data[index]], signal[self.ddT.data[index]], c = "r")
            plt.text(time[self.ddT.data[index]], signal[self.ddT.data[index]] + 0.2, "T''max", fontsize=9, horizontalalignment = 'center')

            # Plot Q-M and T-M Seis I
            if seis is not None:
                # Plot Q-M Bounds
                plt.axvline(time[int(self.Q.data[index] +  np.ceil(qm_bounds[dosage][0] * frequency))], c = "g")
                plt.axvline(time[int(self.Q.data[index] +  np.ceil(qm_bounds[dosage][1] * frequency))], c = "g")

                # Plot Q-M
                # plt.axvline(q_peaks[i] +  qm_seis_peak)
                plt.scatter(time[self.QM.data[index]], seis[self.QM.data[index]], c = "g")
                plt.text(time[self.QM.data[index]], seis[self.QM.data[index]] + 0.2, "Q-M-Seis I", fontsize=9, horizontalalignment = 'center')

                # Plot T-M
                # plt.axvline(t_peaks[i] +  tm_seis_peak)
                plt.scatter(time[self.TM.data[index]], seis[self.TM.data[index]], c = "g")
                plt.text(time[self.TM.data[index]], seis[self.TM.data[index]] + 0.2, "T-M-Seis I", fontsize=9, horizontalalignment = 'center')

            plt.legend(loc = 'upper left')
            plt.show()

    def plot(self, time, signal, smoothed_second, seis, seis2, phono1, phono2):

        fig, ax1 = plt.subplots()

        # Plot Signal
        ax1.plot(time, signal, '--', color='b', label = "Signal", linewidth = 0.75)
        ax1.plot(time, 0.8 * smoothed_second, '--', color='r', label = "Signal", linewidth = 0.5)
        ax1.set_ylim(min(signal) - abs(0.2*min(signal)), max(signal) + abs(0.2*max(signal)))

        # Plot Sies1
        # if seis is not None:
            # ax1.plot(time, seis, color='r', linewidth = 0.5, label = "Seis-I")

        for i in range(len(self.R.data)):
            # R Peaks
            ax1.scatter(time[self.R.data[i]], signal[self.R.data[i]], c='red', marker = "D")
            ax1.text(time[self.R.data[i]], 0.02 + signal[self.R.data[i]], "R", fontsize=9)

            # Q Peaks
            ax1.scatter(time[self.Q.data[i]], signal[self.Q.data[i]], c='green', marker = "D")
            ax1.text(time[self.Q.data[i]], 0.02 + signal[self.Q.data[i]], "Q", fontsize=9)

            # P Peaks
            ax1.scatter(time[self.P.data[i]], signal[self.P.data[i]], c='blue', marker = "D")
            ax1.text(time[self.P.data[i]], 0.02 + signal[self.P.data[i]], "P", fontsize=9)

            # S Peaks
            ax1.scatter(time[self.S.data[i]], signal[self.S.data[i]], c='yellow', marker = "D")
            ax1.text(time[self.S.data[i]],  0.02 + signal[self.S.data[i]], "S", fontsize=9)

            # T Peaks
            ax1.scatter(time[self.T.data[i]], signal[self.T.data[i]], c='magenta', marker = "D")
            ax1.text(time[self.T.data[i]], 0.02 + signal[self.T.data[i]], "T", fontsize=9)

            # T''max Peaks
            ax1.scatter(time[self.ddT.data[i]], signal[self.ddT.data[i]], c='cyan', marker = "D")
            ax1.text(time[self.ddT.data[i]], 0.02 + signal[self.ddT.data[i]], "T''max", fontsize=9)

            # # Plot Seismocardiogram
            # if seis is not None:                
            #     # Q-M
            #     ax1.scatter(time[self.QM.data[i]], seis[self.QM.data[i]], c='y', marker = "D")
            #     ax1.text(time[self.QM.data[i]], 0.02 + seis[self.QM.data[i]], "Q-M-Seis I", fontsize=9)

            #     # T-M
            #     ax1.scatter(time[self.TM.data[i]], seis[self.TM.data[i]], c='y', marker = "D")
            #     ax1.text(time[self.TM.data[i]], 0.02 + seis[self.TM.data[i]], "T-M-Seis I", fontsize=9)

        ax1.legend(loc = 'upper left')
        plt.show()

    def save(self, filename):
        with open(filename + '.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename + '.pkl', 'rb') as input:
            peaks = pickle.load(input)

        # Load Data
        self.time   = peaks.time
        self.signal = peaks.signal

        self.seis1  = peaks.seis1
        self.seis2  = peaks.seis2
        self.seis  = peaks.seis1
        
        self.phono1  = peaks.phono1
        self.phono2  = peaks.phono2
        self.phono  = peaks.phono
        
        self.P = peaks.P
        self.Q = peaks.Q
        self.R = peaks.R
        self.S = peaks.S
        self.T = peaks.T

        self.ST_start = peaks.ST_start
        self.ST_end   = peaks.ST_end

        self.ddT = peaks.ddT

        self.QM  = peaks.QM
        self.TM  = peaks.TM 

        self.get_inital_statistics()

    def get_inital_statistics(self):
        # Get stats    
        self.P._get_inital_statistics()
        self.Q._get_inital_statistics()
        self.R._get_inital_statistics()
        self.S._get_inital_statistics()
        self.T._get_inital_statistics()

        self.ST_start._get_inital_statistics()
        self.ST_end._get_inital_statistics()

        self.ddT._get_inital_statistics()

        self.QM._get_inital_statistics()
        self.TM._get_inital_statistics()


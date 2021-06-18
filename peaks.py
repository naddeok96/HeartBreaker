# Imports
import pickle
import numpy as np
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

        
        self.dT = Variable([])
        self.ddT = Variable([])
        self.ddQ = Variable([])

        self.QM  = Variable([])
        self.QM_seis  = Variable([])
        self.QM_phono = Variable([])
        
        self.TM  = Variable([])
        self.TM_seis  = Variable([])
        self.TM_phono = Variable([])

        if filename is not None:
            self.load(filename)

    def get_R_peaks(self, second, r_max_to_mean_ratio, signal_max_to_mean, signal, r_window_ratio, frequency, display_windowing = False):

        self.R.data = find_peaks(-second,
                                 height = (r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal),
                                 distance = r_window_ratio * frequency)[0]

        # Display
        if display_windowing:
            plt.plot(signal, label = "ECG", color = 'b', linewidth=2)
            plt.plot(-second, '-.', label = "ECG Neg. 2nd Derv.", color = 'g', linewidth=1)
            for i in range(len(self.R.data)):
                # # R Peaks
                plt.scatter(self.R.data[i], signal[self.R.data[i]], c='red', marker = "D")
                plt.scatter(self.R.data[i], -second[self.R.data[i]], c='red', marker = "o")
                plt.text(self.R.data[i] + 0.03, 0.03 + signal[self.R.data[i]], "R", fontsize=9)
                if i ==0:
                    plt.axvline(self.R.data[i] + r_window_ratio * frequency, linestyle=':', color = 'r', linewidth=1, label = "Distance Threshold")
                else:
                    plt.axvline(self.R.data[i] + r_window_ratio * frequency, linestyle='--', color = 'r', linewidth=1)

            plt.plot((r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal) *np.ones(len(signal)), linestyle='--', color = 'k', linewidth=1, label = "Height Threshold")
            plt.legend(loc = "upper right", prop={"size":20})
            plt.axis('off')
            plt.show()

    def add_Q_peak(self, index, q_window_ratio, signal, second, display_windowing = False):
        # Find distance between this r peak and the last
        last_r_peak_distance = abs(self.R.data[index] - self.R.data[index - 1]) if index != 0 else abs(self.R.data[index + 1] - self.R.data[index])

        # Find lower bound of Q windows and define windows
        q_lower_bound = self.R.data[index] - int(q_window_ratio * last_r_peak_distance)
        q_lower_bound = q_lower_bound if q_lower_bound > 0 else 0
        q_window = range(q_lower_bound, self.R.data[index])

        # Add Q peak
        q_peak = find_peaks(second[q_window])[0][-1]


        self.Q.data.append(self.R.data[index] - len(q_window) + q_peak) #np.argmin(signal[q_window]))

        if display_windowing and index != 0:
            # Plot Signal
            plt.plot(range(self.R.data[index-1], self.R.data[index+1]), signal[self.R.data[index-1] : self.R.data[index+1]], label = "ECG", color = 'b', linewidth=2)
            plt.plot(range(self.R.data[index-1], self.R.data[index+1]), second[self.R.data[index-1] : self.R.data[index+1]], '-.', label = "ECG 2nd Derv.", color = 'g', linewidth=1)
            # ax1.plot(time, np.ones(len(time)) * ((r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal)), color='r', label = "Min Height")
            
            plt.scatter(self.Q.data[index], signal[self.Q.data[index]], c='red', marker = "D")
            # plt.scatter(self.R.data[index], signal[self.R.data[index]], c='red', marker = "D")
            plt.scatter(self.Q.data[index], second[self.Q.data[index]], c='red', marker = "o")
            plt.text(self.Q.data[index], 0.02 + signal[self.Q.data[index]], "Q", fontsize=9)

            plt.axvspan(q_lower_bound, self.R.data[index], facecolor='y', alpha=0.2, label= "Window")
            plt.legend(loc = "upper right", prop={"size":12})
            plt.axis('off')
            plt.show()

    def add_ddQ_peak(self, index, signal, second):
        if index == 0:
            # Add ddQ peak
            self.ddQ.data.append(np.argmax(second[:self.R.data[index]]))
            
        else:
            ddq_window = list(range(self.P.data[index], self.R.data[index]))
            self.ddQ.data.append(np.argmax(second[ddq_window]) + self.P.data[index])
     
    def add_P_peak(self, index, p_window_ratio, signal, display_windowing = False):
        # Do not calculate peaks if there is no previous R peak
        if index != 0:
            # Find distance between this r peak and the last
            last_r_peak_distance = abs(self.R.data[index] - self.R.data[index - 1])
            
            # Find lower bound of P windows and define windows
            p_lower_bound = self.R.data[index] - int(p_window_ratio * last_r_peak_distance)
            p_window  = list(range(p_lower_bound, self.Q.data[index])) 

            # Add P peak
            self.P.data.append(int(self.Q.data[index] - len(p_window) + np.argmax(signal[p_window])))

            if display_windowing and index != 0:
                # Plot Signal
                plt.plot(range(self.R.data[index-1], self.R.data[index+1]), signal[self.R.data[index-1] : self.R.data[index+1]], label = "ECG", color = 'b', linewidth=2)
                # plt.plot(range(self.R.data[index-1], self.R.data[index+1]), second[self.R.data[index-1] : self.R.data[index+1]], '-.', label = "ECG 2nd Derv.", color = 'g', linewidth=1)
                # ax1.plot(time, np.ones(len(time)) * ((r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal)), color='r', label = "Min Height")
                
                plt.scatter(self.P.data[index], signal[self.P.data[index]], c='red', marker = "D")
                # plt.scatter(self.R.data[index], signal[self.R.data[index]], c='red', marker = "D")
                # plt.scatter(self.Q.data[index], signal[self.Q.data[index]], c='red', marker = "o")
                plt.text(self.P.data[index], 0.02 + signal[self.P.data[index]], "P", fontsize=9)

                plt.axvspan(p_lower_bound, self.Q.data[index], facecolor='y', alpha=0.2, label= "Window")
                plt.legend(loc = "upper right", prop={"size":15})
                plt.axis('off')
                plt.show()

        else:
            self.P.data.append(0)

        
    def add_S_peak(self, index, s_window_ratio, signal, display_windowing = False):
        # Find distance between this r peak and the next r peak
        next_r_peak_distance = abs(self.R.data[index + 1] - self.R.data[index]) if index != (len(self.R.data) - 1) else abs(self.R.data[index] - self.R.data[index -1])

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

        if display_windowing and index != 0:
            # Plot Signal
            plt.plot(range(self.R.data[index-1], self.R.data[index+1]), signal[self.R.data[index-1] : self.R.data[index+1]], label = "ECG", color = 'b', linewidth=2)
            # plt.plot(range(self.R.data[index-1], self.R.data[index+1]), second[self.R.data[index-1] : self.R.data[index+1]], '-.', label = "ECG 2nd Derv.", color = 'g', linewidth=1)
            # ax1.plot(time, np.ones(len(time)) * ((r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal)), color='r', label = "Min Height")
            
            plt.scatter(self.S.data[index], signal[self.S.data[index]], c='red', marker = "D")
            # plt.scatter(self.R.data[index], signal[self.R.data[index]], c='red', marker = "D")
            # plt.scatter(self.Q.data[index], signal[self.Q.data[index]], c='red', marker = "o")
            plt.text(self.S.data[index], 0.02 + signal[self.S.data[index]], "S", fontsize=9)

            plt.axvspan(self.R.data[index], s_upper_bound, facecolor='y', alpha=0.2, label= "Window")
            plt.legend(loc = "upper right", prop={"size":15})
            plt.axis('off')
            plt.show()

    def add_T_peak(self, index, t_window_ratio, signal, display_windowing = False):
        # Find distance between this r peak and the next r peak
        next_r_peak_distance = abs(self.R.data[index + 1] - self.R.data[index]) if index != (len(self.R.data) - 1) else abs(self.R.data[index] - self.R.data[index - 1])
        
        # Find upper bound of T peak and define windows
        t_upper_bound = self.R.data[index] + int(t_window_ratio * next_r_peak_distance) 
        t_window = list(range(self.S.data[index], t_upper_bound))
        
        # Add T peak
        self.T.data.append(int(np.argmax(signal[t_window]) + self.S.data[index]))

        if display_windowing and index != 0:
            # Plot Signal
            plt.plot(range(self.R.data[index-1], self.R.data[index+1]), signal[self.R.data[index-1] : self.R.data[index+1]], label = "ECG", color = 'b', linewidth=2)
            # plt.plot(range(self.R.data[index-1], self.R.data[index+1]), second[self.R.data[index-1] : self.R.data[index+1]], '-.', label = "ECG 2nd Derv.", color = 'g', linewidth=1)
            # ax1.plot(time, np.ones(len(time)) * ((r_max_to_mean_ratio * signal_max_to_mean) + np.mean(signal)), color='r', label = "Min Height")
            
            plt.scatter(self.T.data[index], signal[self.T.data[index]], c='red', marker = "D")
            # plt.scatter(self.R.data[index], signal[self.R.data[index]], c='red', marker = "D")
            # plt.scatter(self.Q.data[index], signal[self.Q.data[index]], c='red', marker = "o")
            plt.text(self.T.data[index], 0.02 + signal[self.T.data[index]], "T", fontsize=9)

            plt.axvspan(self.S.data[index], t_upper_bound, facecolor='y', alpha=0.2, label= "Window")
            plt.legend(loc = "upper right", prop={"size":15})
            plt.axis('off')
            plt.show()
        
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

    def add_dT_peak(self, index, smoothed_first):
        # Do not calculate peaks if there is no next R peak
        if index != (len(self.R.data) - 1):
            # Look at interval between st_start and t peak
            st_start_t_interval = range(self.ST_start.data[index], self.T.data[index])

            # Locate T'max
            smoothed_first_peaks = find_peaks(smoothed_first[st_start_t_interval], distance = len(st_start_t_interval)/6)[0]
            self.dT.data.append(self.ST_start.data[index] + int(smoothed_first_peaks[-1]))

        else:
            self.dT.data.append(0)

    def add_ddT_peak(self, index, second):
        # Look at interval between S and T peak
        ddTmin_window = range(self.S.data[index], int(np.mean((self.T.data[index], self.R.data[index + 1])))) if index != (len(self.R.data) - 1) else range(self.S.data[index], len(second))

        # Locate T''min
        ddTmin_peak = np.argmin(second[ddTmin_window]) + self.S.data[index]
        
        # Look at interval between S and ddTmin peak
        ddTmax_window = range(self.S.data[index], ddTmin_peak)
        
        # Locate T''max
        ddTmax_peak = find_peaks(second[ddTmax_window]) # , distance = len(ddTmax_window)/6)[0][-1]

        # print(ddTmax_peak)
        # exit()
        plt.plot(second)
        # plt.scatter(self.S.data[index] + ddTmax_peak)
        plt.show()
        exit()
        
        print(self.S.data[index], ddTmax_peak)
        exit()
        self.ddT.data.append(self.S.data[index] + int(ddTmax_peak))

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

    def add_QM_peak_v2(self, index, seis, qm_max_to_mean_ratio, time = None, alt_time = None, signal = None):
        
        if time is None:
            # Set window to look in
            q_t_interval = range(int(np.mean((self.R.data[index], self.S.data[index]))), self.T.data[index])

            # Look for peaks at a certain height relative to the mean and max of signal
            q_t_interval_max_to_mean = np.max(seis[q_t_interval]) - np.mean(seis[q_t_interval])
            qm_seis_peaks = find_peaks(seis[q_t_interval], 
                                                height = (qm_max_to_mean_ratio * q_t_interval_max_to_mean) + np.mean(seis[q_t_interval]),
                                                distance = len(q_t_interval)/8)[0]

            # Save the First Peak
            self.QM.data.append(int(np.mean((self.R.data[index], self.S.data[index]))) + qm_seis_peaks[0])

        else:
            # Set window to look in
            q_t_interval = range((np.abs(alt_time - time[int(np.mean((self.R.data[index], self.S.data[index])))])).argmin(), (np.abs(alt_time - time[self.T.data[index]])).argmin())
            
            # Look for peaks at a certain height relative to the mean and max of signal
            q_t_interval_max_to_mean = np.max(seis[q_t_interval]) - np.mean(seis[q_t_interval])
            qm_seis_peaks = find_peaks(seis[q_t_interval], 
                                                height = (qm_max_to_mean_ratio * q_t_interval_max_to_mean) + np.mean(seis[q_t_interval]),
                                                distance = len(q_t_interval)/8)[0]

            # Save the First Peak
            self.QM.data.append(q_t_interval[0] + qm_seis_peaks[0])

            # plt.plot(alt_time, seis)
            # plt.plot(time, signal)
            # plt.plot(alt_time[q_t_interval], seis[q_t_interval])
            # plt.scatter(alt_time[q_t_interval[0] + qm_seis_peaks[0]], seis[q_t_interval[0] + qm_seis_peaks[0]])
            # plt.show()
            # exit()

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

    def add_TM_peak_v2(self, index, seis, tm_max_to_mean_ratio, time = None, alt_time = None, signal = None, second = None, dosage = None):
        if time is None:
            tm_seis_window = range(self.T.data[index], self.R.data[index + 1])

            # Find all defined peaks in the window with a certain height and width
            tm_seis_peaks_temp = find_peaks(seis[tm_seis_window], 
                                            height   = (tm_max_to_mean_ratio * (np.max(seis[tm_seis_window]) - np.mean(seis[tm_seis_window]))) + np.mean(seis[tm_seis_window]),
                                            distance = len(tm_seis_window)/8)[0]

            # Pick the first such peak
            self.TM.data.append(self.T.data[index] + tm_seis_peaks_temp[0])

        else:
            # Set window to look in
            if dosage < 30:
                if index != (len(self.R.data) - 1):
                    tm_second_window = range(self.T.data[index], self.R.data[index + 1])
                    # Find all defined peaks in the window with a certain height and width
                    tm_second_peaks = find_peaks(second[tm_second_window], 
                                                distance = len(tm_second_window)/8)[0]

                    tm_seis_window = range((np.abs(alt_time - time[self.T.data[index] + tm_second_peaks[0]])).argmin(), (np.abs(alt_time - time[self.R.data[index  + 1]])).argmin())
                else:
                    tm_second_window = range(self.T.data[index], len(second) - 1)

                    # Find all defined peaks in the window with a certain height and width
                    tm_second_peaks = find_peaks(second[tm_second_window], 
                                                distance = len(tm_second_window)/8)[0]

                    tm_seis_window = range((np.abs(alt_time - time[self.T.data[index] + tm_second_peaks[0]])).argmin(), len(seis) - 1)
                    
            else:
                if index != (len(self.R.data) - 1):
                    tm_seis_window = range((np.abs(alt_time - time[self.T.data[index]])).argmin(), (np.abs(alt_time - time[self.R.data[index  + 1]])).argmin())

                else:
                    tm_seis_window = range((np.abs(alt_time - time[self.T.data[index]])).argmin(), len(seis) - 1)
            
            # Find all defined peaks in the window with a certain height and width
            tm_seis_peaks = find_peaks(seis[tm_seis_window], 
                                            height   = (tm_max_to_mean_ratio * (np.max(seis[tm_seis_window]) - np.mean(seis[tm_seis_window]))) + np.mean(seis[tm_seis_window]),
                                            distance = len(tm_seis_window)/8)[0]

            # Pick the first such peak
            self.TM.data.append(tm_seis_window[0] + tm_seis_peaks[0])

            # Plotting for trouble shooting
            # plt.plot(alt_time, seis, label = "seis")
            # plt.plot(time, signal, label = "signal")
            # plt.plot(time, second, label = "second")
            # plt.scatter(time[tm_second_window[0] + tm_second_peaks], second[tm_second_window[0] + tm_second_peaks])
            # plt.plot(alt_time[tm_seis_window], seis[tm_seis_window], label = "seis window")
            # plt.scatter(alt_time[tm_seis_window[0] + tm_seis_peaks], seis[tm_seis_window[0] + tm_seis_peaks])
            # print(alt_time[tm_seis_window[0] + tm_seis_peaks])
            # plt.legend()
            # plt.show()

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

    def plot(self, time, signal, first = None, second = None, seis = None, alt_time = None):

        _, ax1 = plt.subplots()

        # Plot Signal
        ax1.plot(time, signal, '--', color='b', label = "Signal", linewidth = 0.75)
        # # ax1.plot(time, 0.8 * first, '--', color='r', label = "1st", linewidth = 0.5)
        # ax1.plot(time, second, '--', color='r', label = "2nd", linewidth = 0.5)
        ax1.set_ylim(min(signal) - abs(0.2*min(signal)), max(signal) + abs(0.2*max(signal)))

        # Plot Sies1
        if seis is not None:
            ax1.plot(alt_time, seis, color='r', linewidth = 0.5, label = "Seis-I")

        # print("ddQ\t", time[self.ddQ.data])
        # print("QM\t", alt_time[self.QM.data])
        # print("ddT\t", time[self.ddT.data])
        # print("TM\t", alt_time[self.TM.data])
        for i in range(len(self.R.data)):
            # # R Peaks
            ax1.scatter(time[self.R.data[i]], signal[self.R.data[i]], c='red', marker = "D")
            ax1.text(time[self.R.data[i]], 0.02 + signal[self.R.data[i]], "R", fontsize=9)

            # Q Peaks
            ax1.scatter(time[self.Q.data[i]], signal[self.Q.data[i]], c='green', marker = "D")
            ax1.text(time[self.Q.data[i]], 0.02 + signal[self.Q.data[i]], "Q", fontsize=9)

            # ddQ Peaks
            # ax1.scatter(time[self.ddQ.data[i]], second[self.ddQ.data[i]], c='orange', marker = "D")
            # ax1.text(time[self.ddQ.data[i]], 0.02 + second[self.ddQ.data[i]], "ddQ", fontsize=9)

            # # P Peaks
            ax1.scatter(time[self.P.data[i]], signal[self.P.data[i]], c='blue', marker = "D")
            ax1.text(time[self.P.data[i]], 0.02 + signal[self.P.data[i]], "P", fontsize=9)

            # # S Peaks
            ax1.scatter(time[self.S.data[i]], signal[self.S.data[i]], c='yellow', marker = "D")
            ax1.text(time[self.S.data[i]],  0.02 + signal[self.S.data[i]], "S", fontsize=9)

            # T Peaks
            ax1.scatter(time[self.T.data[i]], signal[self.T.data[i]], c='magenta', marker = "D")
            ax1.text(time[self.T.data[i]], 0.02 + signal[self.T.data[i]], "T", fontsize=9)

            # # ST Start Peaks
            # ax1.scatter(time[self.ST_start.data[i]], signal[self.ST_start.data[i]], c='k', marker = "D")
            # ax1.text(time[self.ST_start.data[i]], 0.02 + signal[self.ST_start.data[i]], "S-T Start", fontsize=9)

            # # T'max Peaks
            # ax1.scatter(time[self.dT.data[i]], signal[self.dT.data[i]], c='cyan', marker = "D")
            # ax1.text(time[self.dT.data[i]], 0.02 + signal[self.dT.data[i]], "T'max", fontsize=9)

            # # ddT Peaks
            # ax1.scatter(time[self.ddT.data[i]], second[self.ddT.data[i]], c='cyan', marker = "D")
            # ax1.text(time[self.ddT.data[i]], 0.02 + second[self.ddT.data[i]], "T'max", fontsize=9)

            # # Plot Seismocardiogram
            if seis is not None:                
                # Q-M
                ax1.scatter(alt_time[self.QM.data[i]], seis[self.QM.data[i]], c='y', marker = "D")
                ax1.text(alt_time[self.QM.data[i]], 0.02 + seis[self.QM.data[i]], "Q-M-Seis I", fontsize=9)

                # T-M
                ax1.scatter(alt_time[self.TM.data[i]], seis[self.TM.data[i]], c='y', marker = "D")
                ax1.text(alt_time[self.TM.data[i]], 0.02 + seis[self.TM.data[i]], "T-M-Seis I", fontsize=9)

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

        self.ddT = peaks.dT
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


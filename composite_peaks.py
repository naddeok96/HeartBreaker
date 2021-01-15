# Imports
from variable_statistics import Variable
from scipy.signal import find_peaks
import heartbreaker as hb
import matplotlib.pyplot as plt
import numpy as np
import pickle

class CompositePeaks:

    def __init__(self, peaks = None):

        super(CompositePeaks, self).__init__()
        
        self.peaks = peaks
        self.composites = None

        self.P = Variable([])
        self.Q = Variable([])
        self.R = Variable([])
        self.S = Variable([])
        self.T = Variable([])

        self.ST_start = Variable([])
        self.ST_end   = Variable([])

        self.ddT_max = Variable([])

        self.QM  = Variable([])
        self.QM_seis  = Variable([])
        self.QM_phono = Variable([])
        
        self.TM  = Variable([])
        self.TM_seis  = Variable([])
        self.TM_phono = Variable([])

    def get_N_composite_endpoints(self, N, slide_step_size):
        # Get all heartbeats in composite
        composite_endpoints = []
        i = np.floor(N/2)
        while (i + np.floor(N/2)) <= (self.peaks.T.sample_size - 2):
            
            start = int(i - np.floor(N/2))
            end   = int(start + (N - 1))

            composite_endpoints.append((start, end))

            i = i + slide_step_size
        
        return composite_endpoints

    def get_composite_bounds(self, start, end):
        # Find bounds for composite endpoints
        composite_start = 1e10
        composite_end   = 1e10
        for i in range(end - start + 1):
            # Define current peak interval
            interval = range(self.peaks.T.data[start + i], self.peaks.P.data[start + i + 2])

            # Find endpoints of current peak interval and compare with current composite endpoints
            heartbeat_time   = self.peaks.time[interval] - self.peaks.time[self.peaks.R.data[start + 1 + i]]
            composite_start  = min(composite_start, np.where(heartbeat_time == 0)[0])
            composite_end    = min(composite_end, len(heartbeat_time) - np.where(heartbeat_time == 0)[0])

        return composite_start, composite_end

    def get_clipped_heartbeat_signals(self, i, start, interval, composite_start, composite_end):
        # Center signals in current interval at the R peak
        heartbeat_time   = self.peaks.time[interval]   - self.peaks.time[self.peaks.R.data[start + 1 + i]]
        heartbeat_signal = self.peaks.signal[interval] - self.peaks.signal[self.peaks.R.data[start + 1 + i]]
        heartbeat_seis   = self.peaks.seis[interval]   - self.peaks.signal[self.peaks.R.data[start + 1 + i]]
        heartbeat_phono  = self.peaks.phono[interval]  - self.peaks.signal[self.peaks.R.data[start + 1 + i]]

        # Clip Front
        remove_from_start = int(np.where(heartbeat_time == 0)[0] - composite_start)
        if remove_from_start != 0:
            heartbeat_time   = np.delete(heartbeat_time  , range(remove_from_start))
            heartbeat_signal = np.delete(heartbeat_signal, range(remove_from_start))
            heartbeat_seis   = np.delete(heartbeat_seis  , range(remove_from_start))
            heartbeat_phono  = np.delete(heartbeat_phono , range(remove_from_start))

        # Clip Back
        remove_from_end  = int((len(heartbeat_time) - np.where(heartbeat_time == 0)[0]) - composite_end)
        if remove_from_end != 0:
            heartbeat_time   = np.delete(heartbeat_time  , range(len(heartbeat_time) - remove_from_end , len(heartbeat_time)))
            heartbeat_signal = np.delete(heartbeat_signal, range(len(heartbeat_time) - remove_from_end , len(heartbeat_time)))
            heartbeat_seis   = np.delete(heartbeat_seis  , range(len(heartbeat_time) - remove_from_end , len(heartbeat_time)))
            heartbeat_phono  = np.delete(heartbeat_phono , range(len(heartbeat_time) - remove_from_end , len(heartbeat_time)))

        # Normalize
        heartbeat_signal = hb.normalize(heartbeat_signal)
        heartbeat_seis   = hb.normalize(heartbeat_seis)
        heartbeat_phono  = hb.normalize(heartbeat_phono)

        return heartbeat_time, heartbeat_signal, heartbeat_seis, heartbeat_phono

    def get_N_composite_signal_dataset(self, N, slide_step_size, display = False, dosage = None):
        # Get all heartbeats in composite
        composite_endpoints = self.get_N_composite_endpoints(N, slide_step_size)

        composites = []
        count = 0
        for start, end in composite_endpoints:
            count += 1
             # Find bounds for composite endpoints
            composite_start, composite_end   = self.get_composite_bounds(start, end)

            # Clip all heartbeats to same length
            for i  in range(N):
                # Define current interval
                interval = range(self.peaks.T.data[start + i], self.peaks.P.data[start + i + 2])

                # Clip signals and center them based off R peak
                heartbeat_time, heartbeat_signal, heartbeat_seis, heartbeat_phono = self.get_clipped_heartbeat_signals(i, start, interval, composite_start, composite_end)            

                # Cumulative sum
                composite_time = heartbeat_time if i == 0 else composite_time + heartbeat_time
                composite_signal = heartbeat_signal if i == 0 else composite_signal + heartbeat_signal
                composite_seis = heartbeat_seis if i == 0 else composite_seis + heartbeat_seis
                composite_phono = heartbeat_phono if i == 0 else composite_phono + heartbeat_phono
            
            # Divide by sample size
            composite_time   /= N
            composite_signal /= N
            composite_seis   /= N
            composite_phono  /= N

            composites.append([composite_time, composite_signal, composite_seis, composite_phono])

            if display:
                # Display
                fig, axes2d = plt.subplots(nrows=N, ncols=3)
                signal_lay_over_cell = fig.add_subplot(3,3,2)
                seis_lay_over_cell = fig.add_subplot(3,3,5)
                phono_lay_over_cell = fig.add_subplot(3,3,8)
                composite_cell = fig.add_subplot(1,3,3)

                if dosage is None:
                    plt.suptitle(str(count) + " of " + str(len(composite_endpoints)) + " Composites given " + str(N) + " Heartbeats w/ Step Size of " + str(slide_step_size) , fontsize=20)
                else:
                    plt.suptitle(str(count) + " of " + str(len(composite_endpoints)) + " Composites for Dosage " + str(dosage) + " given " + str(N) + " Heartbeats w/ Step Size of " + str(slide_step_size) , fontsize=20)
                
                for i, row in enumerate(axes2d):
                    for j, cell in enumerate(row):
                        interval = range(self.peaks.T.data[start + i], self.peaks.P.data[start + i + 2])
                        cell.set_xticks([])
                        cell.set_yticks([])
                        
                        if j == 0:
                            if i == 0:
                                cell.set_title("Individual Signals")
                            cell.plot(self.peaks.time[interval], hb.normalize(self.peaks.signal[interval]))
                            cell.plot(self.peaks.time[interval], hb.normalize(self.peaks.seis[interval]))
                            cell.plot(self.peaks.time[interval], hb.normalize(self.peaks.phono[interval]))
                            cell.set_ylabel(start + i)
                            
                        elif j == 1:
                            signal_lay_over_cell.plot(self.peaks.time[interval] - self.peaks.time[self.peaks.R.data[start + 1 + i]], hb.normalize(self.peaks.signal[interval] - self.peaks.signal[self.peaks.R.data[start + 1 + i]]), "--", linewidth= 0.5)
                            seis_lay_over_cell.plot(self.peaks.time[interval] - self.peaks.time[self.peaks.R.data[start + 1 + i]],   hb.normalize(self.peaks.seis[interval] - self.peaks.signal[self.peaks.R.data[start + 1 + i]]), "--", linewidth= 0.5)
                            phono_lay_over_cell.plot(self.peaks.time[interval] - self.peaks.time[self.peaks.R.data[start + 1 + i]],  hb.normalize(self.peaks.phono[interval] - self.peaks.signal[self.peaks.R.data[start + 1 + i]]), "--", linewidth= 0.5)

                            signal_lay_over_cell.set_xticks([])
                            signal_lay_over_cell.set_yticks([])

                            seis_lay_over_cell.set_xticks([])
                            seis_lay_over_cell.set_yticks([])

                            phono_lay_over_cell.set_xticks([])
                            phono_lay_over_cell.set_yticks([])

                        else:
                            if i == 0:
                                signal_lay_over_cell.plot(composite_time, hb.normalize(composite_signal), 'r', linewidth= 2, label = "Composite Signal")
                                signal_lay_over_cell.legend(loc = 'lower left')

                                seis_lay_over_cell.plot(composite_time, hb.normalize(composite_seis), 'r', linewidth= 2, label = "Composite Seis")
                                seis_lay_over_cell.legend(loc = 'lower left')

                                phono_lay_over_cell.plot(composite_time, hb.normalize(composite_phono), 'r', linewidth= 2, label = "Composite Phono")
                                phono_lay_over_cell.legend(loc = 'lower left')


                                composite_cell.plot(composite_signal, 'r', label = "EKG")
                                composite_cell.plot(composite_seis,  'b', label = "Seis",linewidth= 0.5)
                                composite_cell.plot(composite_phono,  'g', label = "Phono",linewidth= 0.5)
                                composite_cell.legend(loc = 'lower left')
                                composite_cell.set_xticks([])
                                composite_cell.set_yticks([])
                                composite_cell.set_title("Composite")
                                signal_lay_over_cell.set_title("Superimposed") 
                
                # Maximize Frame
                mng = plt.get_current_fig_manager()
                mng.full_screen_toggle()

                plt.show()

        self.composites = composites
        return composites

    def get_composite_R_peak(self, signal):
        self.R.data.append(np.argmax(signal))

    def add_composite_Q_peak(self, index, signal, q_window_ratio):
        # Find lower bound of Q windows and define windows
        q_lower_bound = self.R.data[index] - int(q_window_ratio * len(signal))
        q_window = list(range(q_lower_bound, self.R.data[index]))

        # Add Q peak
        self.Q.data.append(self.R.data[index] - len(q_window) + np.argmin(signal[q_window]))

    def add_composite_P_peak(self, index, signal, p_window_ratio):            
            # Find lower bound of P windows and define windows
            p_lower_bound = self.R.data[index] - int(p_window_ratio * len(signal))
            p_window  = list(range(p_lower_bound, self.Q.data[index])) 

            # Add P peak
            self.P.data.append(int(self.Q.data[index] - len(p_window) + np.argmax(signal[p_window])))

    def add_composite_S_peak(self, index, signal, s_window_ratio):
        # Find upper bound of S peak and define windows
        s_upper_bound = self.R.data[index] + int(s_window_ratio * len(signal))
        if s_upper_bound > len(signal):
            s_upper_bound = len(signal)

        s_window = range(self.R.data[index], s_upper_bound)

        # Look for defined peaks
        if len(s_window)/2 > 1:
            possible_defined_peaks = find_peaks(-signal[s_window], distance = len(s_window)/2)[0]
        else:
            possible_defined_peaks = find_peaks(-signal[s_window])[0]

        # If there are no defined peaks just use the max otherwise use the first defined peak
        if len(possible_defined_peaks) == 0:
            self.S.data.append(int(np.argmax(-signal[s_window]) + self.R.data[index]))
        else:
            self.S.data.append(int(possible_defined_peaks[0] + self.R.data[index]))

    def add_composite_T_peak(self, index, signal, t_window_ratio):
        # Find upper bound of T peak and define windows
        t_upper_bound = self.R.data[index] + int(t_window_ratio * len(signal))
        if t_upper_bound > len(signal):
            t_upper_bound = len(signal)

        t_window = list(range(self.S.data[index], t_upper_bound))
        
        # Add T peak
        self.T.data.append(int(np.argmax(signal[t_window]) + self.S.data[index]))

    def add_composite_ST_segment(self, index, signal):

        # Calculate second derivative
        smoothed_first, second = hb.get_derivatives(signal)

        # Look at interval between s and t peak
        s_t_interval = range(self.S.data[index], self.T.data[index])

        # Find start of S-T segment
        if len(s_t_interval) > 1:
            self.ST_start.data.append(self.S.data[index] + np.argmin(second[s_t_interval[ :int(0.15 * len(s_t_interval))]]))
        else:
            self.ST_start.data.append(self.S.data[index])

        # Look at interval between st_start and t peak
        st_start_t_interval = range(self.ST_start.data[index], self.T.data[index])

        # Find end of S-T segment
        if len(st_start_t_interval)/6 > 1:
            smoothed_first_peaks = find_peaks(smoothed_first[st_start_t_interval], distance = len(st_start_t_interval)/6)[0]
        else:
            smoothed_first_peaks = find_peaks(smoothed_first[st_start_t_interval])[0]
        
        if len(smoothed_first_peaks) > 1:
            st_end = int(smoothed_first_peaks[-2])
        elif len(smoothed_first_peaks) == 1:
            st_end = smoothed_first_peaks[0]
        else:
            st_end = ((self.T.data[index] - self.ST_start.data[index])/2) + self.ST_start.data[index]
    
        self.ST_end.data.append(self.ST_start.data[index] + st_end)

    def add_composite_ddT_max(self, index, signal):

        # Calculate second derivative
        _, second = hb.get_derivatives(signal)

        # Look at interval between st_start and t peak
        st_start_t_interval = range(self.ST_start.data[index], self.T.data[index])

        # Locate T''max
        if len(st_start_t_interval)/6 > 1:
            second_peaks = find_peaks(5 * second[st_start_t_interval], distance = len(st_start_t_interval)/6)[0]
        else:
            second_peaks = find_peaks(5 * second[st_start_t_interval])[0]

        if len(second_peaks) == 0:
            second_peaks = [self.T.data[index]] if np.isnan(np.median(second[st_start_t_interval])) else [np.median(second[st_start_t_interval])]
        

        # Look at interval between big hump and t peak
        low_mag_interval = range(self.ST_start.data[index] + int(second_peaks[0]), self.T.data[index])

        # Locate T''max
        if len(low_mag_interval)/2 > 1:
            second_low_mag_peaks = find_peaks(5 * second[low_mag_interval], distance = len(low_mag_interval)/2)[0]
        else:
            second_low_mag_peaks = find_peaks(5 * second[low_mag_interval])[0]

        if len(second_low_mag_peaks) == 0:
            # Add data
            if len(st_start_t_interval) == 0:
                st_start_t_interval = [self.ST_start.data[index]]
            self.ddT_max.data.append(st_start_t_interval[0] + int(second_peaks[-1]))

        else:
            # Add data
            self.ddT_max.data.append(low_mag_interval[0] + int(second_low_mag_peaks[-1]))

    def add_composite_QM_seis_peak(self, index, seis, qm_max_to_mean_ratio, qm_bound_indices):
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
        self.QM_seis.data.append(self.Q.data[index] + qm_seis_peaks_temp[np.argmin(distances)])

    def add_composite_QM_phono_peak(self, index, phono):
        qm_window = range(self.Q.data[index], self.T.data[index])

        # Find all defined peaks in the window with a certain height and width
        qm_phono_peaks_temp = find_peaks(phono[qm_window], 
                                        height   = np.mean(phono[qm_window]),
                                        distance = len(qm_window)/8)[0]

        if len(qm_phono_peaks_temp) == 0:
            self.QM_phono.data.append(self.Q.data[index] + (self.T.data[index] - self.Q.data[index])/2)
        else:
            # Pick the first such peak
             self.QM_phono.data.append(self.Q.data[index] + qm_phono_peaks_temp[0])

    def add_composite_TM_seis_peak(self, index, seis, tm_max_to_mean_ratio):   
        # Look between current T peak and next R peak
        tm_window = range(self.T.data[index], len(seis))
        
        # Find all defined peaks in the window with a certain height and width
        if len(tm_window)/8 >= 1:
            tm_seis_peaks_temp = find_peaks(seis[tm_window], 
                                            height   = (tm_max_to_mean_ratio * (np.max(seis[tm_window]) - np.mean(seis[tm_window]))) + np.mean(seis[tm_window]),
                                            distance = len(tm_window)/8)[0]

        else:
            tm_seis_peaks_temp = find_peaks(seis[tm_window], 
                                            height   = (tm_max_to_mean_ratio * (np.max(seis[tm_window]) - np.mean(seis[tm_window]))) + np.mean(seis[tm_window]))[0]
        
        if len(tm_seis_peaks_temp) == 0:
            self.TM_seis.data.append(self.T.data[index] + (( len(seis) - self.T.data[index])/2))
        else:
            # Pick the first such peak
            self.TM_seis.data.append(self.T.data[index] + tm_seis_peaks_temp[0])
        
    def add_composite_TM_phono_peak(self, index, phono):   
        # Look between current T peak and next R peak
        tm_window = range(self.T.data[index], len(phono))

        # Find all defined peaks in the window with a certain height and width
        if len(tm_window)/8 >= 1:
            tm_peaks_temp = find_peaks(phono[tm_window], 
                                        height   = np.mean(phono[tm_window]),
                                        distance = len(tm_window)/8)[0]
        else:
            tm_peaks_temp = find_peaks(phono[tm_window], 
                                    height   = np.mean(phono[tm_window]))[0]

        if len(tm_peaks_temp) == 0:
            self.TM_seis.data.append(self.T.data[index] + ((len(phono) - self.T.data[index])/2))
        else:
            # Pick the first such peak
             self.TM_phono.data.append(self.T.data[index] + tm_peaks_temp[0])
       
    def update_composite_peaks(self, dosage):
        r_window_ratio = 0.3       ## r_window_ratio == 150 beats per min
        q_window_ratio = 0.08      ## 
        s_window_ratio = 0.07      ## 
        p_window_ratio = 0.30      ## 
        t_window_ratio = 0.45
        qm_max_to_mean_ratio = 0.4
        tm_max_to_mean_ratio = 0.4 

        # Convert to time steps
        qm_bounds    = { 0: [0.2500, 0.3333],
                        10: [0.1000, 0.1111],
                        20: [0.0714, 0.0769],
                        30: [0.0625, 0.0667],
                        40: [0.0488, 0.0513],
                        42: [0.0488, 0.0513]}
        qm_bound_indices = [x * 4000 for x in qm_bounds[dosage]]

        for i, (time, signal, seis, phono) in enumerate(self.composites):

            # Find R Peaks in Interval
            self.get_composite_R_peak(signal) 

            # Use R peak to find other peaks
            self.add_composite_Q_peak(i, signal, q_window_ratio)
            self.add_composite_P_peak(i, signal, p_window_ratio)
            self.add_composite_S_peak(i, signal, s_window_ratio)
            self.add_composite_T_peak(i, signal, t_window_ratio)

            self.add_composite_ST_segment(i, signal)
            self.add_composite_ddT_max(i, signal)

            self.add_composite_QM_seis_peak(i, seis, qm_max_to_mean_ratio, qm_bound_indices) 
            self.add_composite_QM_phono_peak(i, phono)

            self.add_composite_TM_seis_peak(i, seis, tm_max_to_mean_ratio)
            self.add_composite_TM_phono_peak(i, phono)

    def save(self, filename):
        with open(filename + '.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename + '.pkl', 'rb') as input:
            peaks = pickle.load(input)

        # Load Data
        self.peaks = peaks.peaks
        self.composites = peaks.composites
        
        self.P = peaks.P
        self.Q = peaks.Q
        self.R = peaks.R
        self.S = peaks.S
        self.T = peaks.T

        self.ST_start = peaks.ST_start
        self.ST_end   = peaks.ST_end

        self.ddT_max = peaks.ddT_max

        self.QM  = peaks.QM
        self.QM_seis  = peaks.QM_seis
        self.QM_phono = peaks.QM_phono
        
        self.TM  = peaks.TM
        self.TM_seis  = peaks.TM_seis
        self.TM_phono = peaks.TM_phono

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

        self.ddT_max._get_inital_statistics()

        self.QM._get_inital_statistics()
        self.QM_seis._get_inital_statistics()
        self.QM_phono._get_inital_statistics()
        
        self.TM._get_inital_statistics()
        self.TM_seis._get_inital_statistics()
        self.TM_phono._get_inital_statistics()




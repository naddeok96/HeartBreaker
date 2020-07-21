# Imports
import matplotlib.pyplot as plt
from patient import Patient 
from heartbreaker import HeartBreak
# Initalize heartbreaker
hb = HeartBreak()
from files_w_ints import files
import numpy as np
import math
import os
import csv


# Hyperparameters
#---------------------#
# Signal Settings
preloaded_signal = False
save_signal      = True
peaks_to_excel   = True
use_intervals    = True

# Display Settings
show_signal      = False
show_freq_domain = False
show_bandpass    = False
show_derivatives = False
show_peaks       = True

folder_name = "1 9 2020 AH TDMS ESSENTIAL"

# Implicit
dosage = 0
interval_number = 2 # "None"

# Iterate
# for dosage in files[folder_name].keys():
#      print(dosage)
#      for interval_number in files[folder_name][dosage]["intervals"]:
#           print(interval_number)

# Pick file
file_name = files[folder_name][dosage]["file_name"]
save_file_name = folder_name + "_" + file_name + "_d" + str(dosage) if use_intervals == False else folder_name + "_" + file_name + "_d" + str(dosage) + "_i" + str(interval_number)

# Pick Interval
if use_intervals == True and files[folder_name][dosage]["intervals"] != ["None"]:
     start_time = files[folder_name][dosage]["intervals"][interval_number][0]
     end_time   = files[folder_name][dosage]["intervals"][interval_number][1]
#---------------------------------------------------------------------------------#
# End of Hyperparameters

# Initalize patient and interval
if preloaded_signal == False:
     # Change directory
     wd = 'data/' + folder_name + '/files_of_interest'

     # Load TDMS file into Patient object
     patient = Patient(wd, file_name)

     # Declare time and signal
     if use_intervals == True and files[folder_name][dosage]["intervals"] != ["None"]:
          start = (start_time - np.min(patient.times))*patient.frequency
          end   = patient.total_time*patient.frequency if end_time == "end" else (end_time - np.min(patient.times))*patient.frequency
          interval = range(int(start), int(end))
          time, signal = patient.get_interval(interval)
     else:
          time = patient.times
          signal = patient.ecg
     
     # Save signal
     if save_signal == True:
          np.savetxt('time_'  + save_file_name + '.csv', time, delimiter=',')
          np.savetxt('signal_'+ save_file_name + '.csv', signal, delimiter=',')

else:
     os.chdir("data/Derived")
     time   = np.loadtxt('time_' + save_file_name + '.csv', delimiter=',')
     signal = np.loadtxt('signal_' + save_file_name + '.csv', delimiter=',')

# View Signal
if show_signal == True:
     hb.plot_signal(time, signal)

# View Frequency Domain
if show_freq_domain == True:
     hb.get_fft(time = time, 
                    signal = signal,
                    plot = True)

     # hb.get_spectrum(time = time, 
     #                     signal = signal)

# Bandblock
if show_bandpass == True:
     bandpass_signal = hb.lowpass_filter(time = time, 
                                   signal = signal,
                                   cutoff_freq = 10)

     hb.get_fft(time = time, 
                    signal = bandpass_signal,
                    plot = True)

# View Derivatives
if show_derivatives == True:
     bandpass_signal = hb.bandpass_filter(time = time, 
                                        signal = signal,
                                        freqmin = 59, 
                                        freqmax = 61)
     hb.get_derivatives(signal = bandpass_signal, 
                         plot =True)

# Find Peaks
if show_peaks == True:
     lowpass_signal = hb.bandpass_filter(time    = time, 
                                        signal  = signal,
                                        freqmin = 59, 
                                        freqmax = 61)

     # Low-Pass filter under 10Hz
     lowpass_signal = hb.lowpass_filter(time = time, 
                                        signal = lowpass_signal,
                                        cutoff_freq = 50)

     peaks = hb.get_ecg_peaks(time = time, 
                              signal = lowpass_signal,
                              plot = False,
                              plot_st_segments = False,
                              plot_segmentation_decisons = False)

     

     if peaks_to_excel == True:
          hb.save_peaks_to_excel(save_file_name, time, peaks)
          os.chdir("../..")
     
     


     


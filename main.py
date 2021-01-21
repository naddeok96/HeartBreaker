# Imports
import matplotlib.pyplot as plt
from patient import Patient 
import random
import heartbreaker as hb
from files_w_doseage_and_ints import files
from bobs_intervals import bobs_waves
import numpy as np
import math
import os
import csv

# Hyperparameters
#---------------------#
# Signal Settings
preloaded_signal = False
save_signal      = False
save_peaks       = False
peaks_to_excel   = False
use_intervals    = False

# Display Settings
show_signal      = False
show_freq_domain = False
show_bandpass    = False
show_derivatives = False
show_peaks       = True

folder_name = "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL" #  
print(folder_name)
# Iterate
for dosage in files[folder_name].keys():
     if dosage != 0:
          continue

     # sample_interval = random.choice(list(files[folder_name][dosage]["intervals"]))

     print("D: ", dosage)
     for interval_number in files[folder_name][dosage][1]["intervals"]:
          # if interval_number != sample_interval:
          #      continue
          # if interval_number != 1:
          #      continue

          if use_intervals == True:
               print("I: ", interval_number)

          # Pick file
          file_name = files[folder_name][dosage][1]["file_name"]
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
                    time, signal, seis1, seis2, phono1, phono2 = patient.get_interval(interval)
               else:
                    time = patient.times
                    signal = patient.ecg
                    seis1 = patient.seis1
                    seis2 = patient.seis2
                    phono1 = patient.phono1
                    phono2 = patient.phono2
               
               # Save signal
               if save_signal == True:
                    np.savetxt('time_'  + save_file_name + '.csv', time, delimiter=',')
                    np.savetxt('signal_'+ save_file_name + '.csv', signal, delimiter=',')
                    np.savetxt('seis1_'+ save_file_name + '.csv', seis1, delimiter=',')
                    np.savetxt('seis2_'+ save_file_name + '.csv', seis2, delimiter=',')
                    np.savetxt('phono1_'+ save_file_name + '.csv', phono1, delimiter=',')
                    np.savetxt('phono2_'+ save_file_name + '.csv', phono2, delimiter=',')

          else:
               os.chdir("data/Derived")
               time   = np.loadtxt('time_' + save_file_name + '.csv', delimiter=',')
               signal = np.loadtxt('signal_' + save_file_name + '.csv', delimiter=',')
               seis1 = np.loadtxt('seis1_' + save_file_name + '.csv', delimiter=',')
               seis2 = np.loadtxt('seis2_' + save_file_name + '.csv', delimiter=',')
               phono1 = np.loadtxt('phono1_' + save_file_name + '.csv', delimiter=',')
               phono2 = np.loadtxt('phono2_' + save_file_name + '.csv', delimiter=',')

          # View Signal
          if show_signal == True:
               hb.plot_signal(time, signal, intervals = files[folder_name][dosage]["intervals"] if use_intervals else None)

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

               # lowpass_signal = hb.bandpass_filter(time    = time, 
               #                                    signal  = signal,
               #                                    freqmin = 59, 
               #                                    freqmax = 61)
               
               # # Low-Pass filter under 10Hz
               # lowpass_signal = hb.lowpass_filter(time = time, 
               #                                    signal = lowpass_signal,
               #                                    cutoff_freq = 50)
               print(min(time), max(time))
               os.chdir("../..")
               continue

               peaks = hb.temp_get_ecg_peaks(time = time, 
                                             signal = signal, #signal = lowpass_signal,
                                             dosage = dosage,
                                             peaks_dict = None,
                                             plot = True,
                                             plot_st_segments = False,
                                             random_sample_size = 25,
                                             plot_segmentation_decisons = False,
                                             seis1 = seis1,
                                             seis2 = seis2,
                                             phono1 = phono1,
                                             phono2 = phono2)

               if save_peaks:
                    peaks.save(save_file_name)

               if peaks_to_excel:
                    hb.save_peaks_to_excel(save_file_name, time, peaks)
               os.chdir("../..")
               
               





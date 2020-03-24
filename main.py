# Imports
import matplotlib.pyplot as plt
from patient import Patient 
from heartbreaker import HeartBreak
import numpy as np
import os
import csv
import xlwt 
from xlwt import Workbook 

# Hyperparameters
#---------------------#
# Patient Information
folder_name = "1 9 2020 AH TDMS ESSENTIAL"
dosage      = 40 # [mcg/kg/min]

# Settings
preloaded_signal = True
save_signal      = True
peaks_to_excel    = False
use_intervals    = False

# Display Settings
show_signal      = False
show_freq_domain = False
show_bandpass    = False
show_derivatives = False
show_peaks       = True

# Optinal Intervals
if use_intervals == True:
    start_time = 10
    end_time   = 20

# Known Files
files = {"1 9 2020 AH TDMS ESSENTIAL": {10: "DAQData_010920140634",
                                        20: "DAQData_010920141106",
                                        30: "DAQData_010920141627",
                                        40: "DAQData_010920142816"}}

# Pick file
file_name = files[folder_name]["files_of_interest"][dosage]

# Initalize patient and interval
if preloaded_signal == False:
    # Change directory
    wd = 'data/' + folder_name

    # Load TDMS file into Patient object
    patient = Patient(wd, file_name)

    # Declare time and signal
    if use_intervals == True:
        interval = range(start_time*patient.frequency, end_time*patient.frequency) # Clean Data
        time, signal = patient.get_interval(interval)
    else:
        time = patient.times
        signal = patient.ecg
    
    # Save signal
    if save__signal == True:
        np.savetxt('time_'+ file_name + '.csv', time, delimiter=',')
        np.savetxt('signal_'+ file_name + '.csv', signal, delimiter=',')

else:
    os.chdir("data/Derived")
    time   = np.loadtxt('time_' + file_name + '.csv', delimiter=',')
    signal = np.loadtxt('signal_' + file_name + '.csv', delimiter=',')


# Initalize heartbreaker
hb = HeartBreak()

# View Signal
if show_signal == True:
    hb.plot_signal(time, signal)

# View Frequency Domain
if show_freq_domain == True:
    hb.get_fft(time = time, 
               signal = signal,
               plot = True)

    hb.get_spectrum(time = time, 
                    signal = signal)

# Bandblock
if show_bandpass_freq_domain == True:
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
    lowpass_signal = hb.bandpass_filter(time = time, 
                               signal = signal,
                               freqmin = 59, 
                               freqmax = 61)

    # Low-Pass filter under 10Hz
    lowpass_signal = hb.lowpass_filter(time = time, 
                                signal = lowpass_signal,
                                cutoff_freq = 50)

    peaks = hb.get_ecg_peaks(time = time, 
                             signal = lowpass_signal,
                             plot  = True,
                             plot_st_segments = True)
   
    # Save the peaks to excel
    if peaks_to_excel == True:
        # Excel Workbook Object is created 
        wb = Workbook() 
        
        # Create sheet
        sheet = wb.add_sheet('Peaks') 

        # Write out each peak and data
        for i, peak in enumerate(peaks):
            sheet.write(0, i, peak)
            for j, value in enumerate(peaks[peak]):
                sheet.write(j + 1, i, float(value))

        wb.save(file_name + '.xls') 
        


                


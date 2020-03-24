'''
This code will run the classes in this repo
'''
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
preloaded_signal = False
show_signal = False
show_freq_domain = False
show_bandpass_freq_domain = False
show_derivatives = False
show_peaks = True
add_to_excel = True
#file_name = "DAQData_010920140634"
#file_name = "DAQData_010920141106"
#file_name = "DAQData_010920141627"
file_name = "DAQData_010920142816"
excel_filename = file_name

# Initalize patient and interval
if preloaded_signal == False:
    wd = 'data/1 9 2020 AH TDMS ESSENTIAL'
    patient = Patient(wd, file_name)
    interval = range(10*patient.frequency, 20*patient.frequency) # Clean Data
    #interval = range(45*patient.frequency, 55*patient.frequency) # Show possible issues with data
    #time, signal = patient.get_interval(interval)
    time = patient.times
    signal = patient.ecg
    
    # Save signal
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
    signal2 = signal1 = hb.lowpass_filter(time = time, 
                                signal = signal,
                                cutoff_freq = 10)

    hb.get_fft(time = time, 
               signal = signal2,
               plot = True)

# View Derivatives
if show_derivatives == True:
    signal2 = hb.bandpass_filter(time = time, 
                               signal = signal,
                               freqmin = 59, 
                               freqmax = 61)
    hb.get_derivatives(signal = signal2, 
                       plot =True)

# Find Peaks
if show_peaks == True:
    signal3 = hb.bandpass_filter(time = time, 
                               signal = signal,
                               freqmin = 59, 
                               freqmax = 61)

    # Low-Pass filter under 10Hz
    signal3 = hb.lowpass_filter(time = time, 
                                signal = signal3,
                                cutoff_freq = 50)

    # Low-Pass filter under 10Hz
    signal4 = hb.lowpass_filter(time = time, 
                                signal = signal,
                                cutoff_freq = 10)

    peaks = hb.get_ecg_peaks(time = time, 
                             signal = signal3,
                             plot  = True,
                             plot_st_segments = True)
   
    if add_to_excel == True:
        # Excel Workbook Object is created 
        wb = Workbook() 
        
        # Create sheet
        sheet1 = wb.add_sheet('Peaks') 

        # Write out each peak and data
        for i, peak in enumerate(peaks):
            sheet1.write(0, i, peak)
            for j, value in enumerate(peaks[peak]):
                sheet1.write(j + 1, i, float(value))

        wb.save(excel_filename + '.xls') 
        


                


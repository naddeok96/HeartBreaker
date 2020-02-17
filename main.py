'''
This code will run the classes in this repo
'''
# Imports
from heartbreaker import HeartBreak

# Hyperparameters
show_freq_domain = False
show_bandpass_freq_domain = False
show_derivatives = False
show_peaks = True

# Main
wd = 'data/Dobutamine Stress Bob 5-22-2019'
file_name = "DAQData_052219142754" 
patient = HeartBreak(wd, file_name)

# Data Set Up
interval = range(3*patient.frequency, 7*patient.frequency)

# Find Peaks
if show_peaks == True:
    patient.ecg = patient.ecg_bandpass_filter(29, 31)
    peaks = patient.ecg_peaks(interval = interval, 
                            plot = True)

# View Derivatives
if show_derivatives == True:
    patient.ecg_derivatives(interval = interval, plot =True)

# View Frequency Domain
if show_freq_domain == True:
    patient.ecg_fft(interval = interval, 
                    plot = True)

# Bandblock
if show_bandpass_freq_domain == True:
    patient.ecg = patient.ecg_bandpass_filter(29, 31)
    patient.ecg_fft(interval = interval, 
                    plot = True)


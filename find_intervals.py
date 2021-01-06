# Imports
import os
import numpy as np
import heartbreaker as hb
import matplotlib.pyplot as plt
from interval_finder_gui import HeartbeatIntervalFinder
from files_w_doseage_and_ints import files

# Iterate
for folder_name in files.keys():
    if folder_name != "ECG-Phono-Seismo DAQ Data 8 20 2020 2": # "1 9 2020 AH TDMS ESSENTIAL"
        continue

    print("Folder: ", folder_name)

    for dosage in files[folder_name].keys():
        # if dosage != 10:
        #     continue
        # sample_interval = random.choice(list(files[folder_name][dosage]["intervals"]))

        print("Doseage: ", dosage)

        if files[folder_name][dosage] is None:
            continue

        
        for file_number in files[folder_name][dosage]:
            
            finder = HeartbeatIntervalFinder(files = files,
                                                folder_name = folder_name,
                                                dosage = dosage,
                                                file_number = file_number,
                                                interval_number = 1,
                                                interval_size = 240)
            exit()

            # Calculate Composite 
            signal = hb.bandpass_filter(time    = time, 
                                        signal  = signal,
                                        freqmin = 59, 
                                        freqmax = 61)

            seis = hb.bandpass_filter(time    = time, 
                                        signal  = seis,
                                        freqmin = 59, 
                                        freqmax = 61)

            phono = hb.bandpass_filter(time    = time, 
                                        signal  = phono,
                                        freqmin = 59, 
                                        freqmax = 61)
            
            # Low-Pass filter under 10Hz
            signal = hb.lowpass_filter(time = time, 
                                        signal = signal,
                                        cutoff_freq = 50)

            seis = hb.lowpass_filter(time = time, 
                                        signal = seis,
                                        cutoff_freq = 50)

            # phono = hb.lowpass_filter(time = time, 
            #                             signal = phono,
            #                             cutoff_freq = 10)

            
            plt.plot(time, signal, label = "ECG")
            # plt.plot(time, 5*seis, label = "Seis")
            plt.plot(time, phono, label = "Phono")
            plt.vlines(echo_time, min(signal), 2*max(signal), label = "2D Echo Time")
            plt.legend(loc = "upper left")
            plt.show()


            os.chdir("../..")

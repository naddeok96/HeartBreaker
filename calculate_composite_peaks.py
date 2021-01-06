# Imports
import os
import numpy as np
import heartbreaker as hb 
from variable_statistics import Variable
from composite_peaks import CompositePeaks
from files_w_doseage_and_ints import files
from verification_gui import HeartbeatVerifier

# Hyperparameters
#---------------------#
# Signal Settings
preloaded_signal = False
save_signal      = False
save_peaks       = True

# Pick Folder
folder_name = "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL" # "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "Dobutamine Stress Test 62719JS"
#---------------------#

print("Folder: ", folder_name)
# Iterate
for dosage in files[folder_name].keys():
    # if dosage != 40:
    #     continue
    # sample_interval = random.choice(list(files[folder_name][dosage]["intervals"]))

    print("Doseage: ", dosage)

    if files[folder_name][dosage] is None:
        continue

    
    for file_number in files[folder_name][dosage]:
        # if file_number != 1:
        #         continue

        print("File Number: ", file_number)

        for interval_number in files[folder_name][dosage][file_number]["intervals"]:
        #     # if interval_number != sample_interval:
        #     #      continue
            # if interval_number != 1:
            #     continue

            print("Interval: ", interval_number)

            # Load Data
            time, signal, seis1, seis2, phono1, phono2 = hb.load_file_data( files, folder_name, 
                                                                            dosage, file_number,
                                                                            interval_number, preloaded_signal, 
                                                                            save_signal)

            # Calculate Composite 
            lowpass_signal = hb.bandpass_filter(time    = time, 
                                                  signal  = signal,
                                                  freqmin = 59, 
                                                  freqmax = 61)
               
               # Low-Pass filter under 10Hz
            lowpass_signal = hb.lowpass_filter(time = time, 
                                                signal = lowpass_signal,
                                                cutoff_freq = 50)



            peaks = hb.get_peaks_for_composites(time   = time, 
                                                signal = lowpass_signal,
                                                dosage = dosage,
                                                seis1  = seis1,
                                                phono1 = phono1)
            peaks.get_inital_statistics()

            print("Time Length: ", max(time) - min(time))
            print("R Peaks: ", len(peaks.R.data))
            if (len(peaks.R.data) > 10):
                composite_peaks = CompositePeaks(peaks)
                composite_peaks.get_N_composite_signal_dataset(10, 5, display = True)
                composite_peaks.update_composite_peaks(dosage = dosage)

                print("Composites: ", len(composite_peaks.composites))

                if save_peaks:
                    os.chdir(os.getcwd() + "/composites")

                    # Pick file
                    file_name = files[folder_name][dosage][file_number]["file_name"]
                    save_file_name = folder_name + "_d" + str(dosage) + "_" + file_name + "_i" + str(interval_number)
                    composite_peaks.save(save_file_name)
                os.chdir("../../..")

            else:
                exit()
                os.chdir("../..")


    
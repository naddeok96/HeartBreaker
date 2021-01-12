
# BOB ONLY EDIT THE HYPERPARAMETER SECTION
# Hyperparameters
#------------------------------------------------------------------------------------#
folder_name = "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL" 
area_around_echo_size = 240 #  In Seconds
composite_size        = 10
step_size             = 5

use_intervals    = False
preloaded_signal = False
save_signal      = False

display_intervals  = False
display_composites = False
verify_labels      = True
#------------------------------------------------------------------------------------#
# END of Hyperparameters

# Imports
import os
import numpy as np
import heartbreaker as hb
import matplotlib.pyplot as plt
from composite_peaks import CompositePeaks
from files_w_doseage_and_ints import files
from verification_gui import HeartbeatVerifier
from composite_statistics import CompositeStats
from interval_finder_gui import HeartbeatIntervalFinder

# Initalize Composite Stats
composite_statistics = {0  : CompositeStats(), 
                        10 : CompositeStats(),
                        20 : CompositeStats(),
                        30 : CompositeStats(),
                        40 : CompositeStats(), 
                        42 : CompositeStats()}

print("Folder: ", folder_name)
        
# Check Intervals
if use_intervals: 
    os.chdir("../Derived")
    files = hb.load_intervals(folder_name)
    os.chdir("../..")

if display_intervals:
    finder = HeartbeatIntervalFinder(files = files,
                                        folder_name = folder_name,
                                        area_around_echo_size = 240,
                                        use_intervals = use_intervals,
                                        preloaded_signal = preloaded_signal,
                                        save_signal = save_signal)

    # Use updated intervals moving forward
    os.chdir("../Derived")
    files = hb.load_intervals(folder_name)
    os.chdir("../..")

# Check Composites
for dosage in files[folder_name]:
    # Pick file
    file_name = files[folder_name][dosage][1]["file_name"]
    save_file_name = folder_name + "_d" + str(dosage) + "_" + file_name + "_i" + str(1)                                                                                

    # Load Data
    time, signal, seis1, seis2, phono1, phono2 = hb.load_file_data( files = files, 
                                                                    folder_name = folder_name, 
                                                                    dosage = dosage, 
                                                                    file_number = 1,
                                                                    interval_number = 1, 
                                                                    preloaded_signal = preloaded_signal, 
                                                                    save_signal = save_signal)

    # Low Pass Signal 
    signal = hb.bandpass_filter(time    = time, 
                                signal  = signal,
                                freqmin = 59, 
                                freqmax = 61)
    signal = hb.lowpass_filter(time = time, 
                                signal = signal,
                                cutoff_freq = 50)


    # Define T and P peaks to build composites
    peaks = hb.get_peaks_for_composites(time   = time, 
                                        signal = signal,
                                        dosage = dosage,
                                        seis1  = seis1,
                                        phono1 = phono1)
    peaks.get_inital_statistics()

    # Build Composites
    composite_peaks = CompositePeaks(peaks)
    composite_peaks.get_N_composite_signal_dataset(composite_size, step_size, display = display_composites, dosage = dosage)
    composite_peaks.update_composite_peaks(dosage = dosage)

    if verify_labels:
        verifier = HeartbeatVerifier(composite_peaks, 
                                    folder_name = folder_name,
                                    dosage      = dosage,
                                    file_name   = file_name,
                                    interval_number = 1)

        composite_peaks.load(save_file_name)

    composite_statistics[dosage].add_data(composite_peaks)

    os.chdir("../..")


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.axis('off')

ax2 = fig.add_subplot(212)
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.axis('off')

ax1.set_title("1/(E-M)ino")

ax2.set_title("1/(E-M)lusi")


qm_seis, qm_phono, tm_seis, tm_phono, doses = [], [], [], [], []
for dosage in composite_statistics:
    if composite_statistics[dosage].QM_interval_seis is not None:
        doses.append(dosage)

        qm_seis.append(1/composite_statistics[dosage].QM_interval_seis)
        qm_phono.append(1/composite_statistics[dosage].QM_interval_phono)

        tm_seis.append(1/composite_statistics[dosage].TM_interval_seis)
        tm_phono.append(1/composite_statistics[dosage].TM_interval_phono)


ax11 = fig.add_subplot(221)
ax11.scatter(qm_seis, doses, c = 'r', label = "Seis")
qm_seis_m, qm_seis_b = np.polyfit(qm_seis, doses, 1)
ax11.plot(qm_seis, qm_seis_m*np.asarray(qm_seis) + qm_seis_b, c = 'r')
ax11.set_ylim(-10, 50)
ax11.set_yticks(np.arange(0, 50, 10), minor=False)
ax11.set_xlim(min(qm_seis) - 0.1*(max(qm_seis) - min(qm_seis)), max(qm_seis) + 0.1*(max(qm_seis) - min(qm_seis)))
ax11.set_title("Seismo")
ax11.set_ylabel("Dobutamine Infusion (mcg/kg/min)")
ax11.grid()

ax12 = fig.add_subplot(222)
ax12.scatter(qm_phono, doses, c = 'b', label = "Phono")
qm_phono_m, qm_phono_b = np.polyfit(qm_phono, doses, 1)
ax12.plot(qm_phono, qm_phono_m*np.asarray(qm_phono) + qm_phono_b, c = 'b')
ax12.set_ylim(-10, 50)
ax12.set_yticks(np.arange(0, 50, 10), minor=False)
ax12.set_xlim(min(qm_phono) - 0.1*(max(qm_phono) - min(qm_phono)), max(qm_phono) + 0.1*(max(qm_phono) - min(qm_phono)))
ax12.set_title("Phono")
ax12.grid()

ax21 = fig.add_subplot(223)
ax21.scatter(tm_seis, doses, c = 'r',  label = "Seis")
tm_seis_m, tm_seis_b = np.polyfit(tm_seis, doses, 1)
ax21.plot(tm_seis, tm_seis_m*np.asarray(tm_seis) + tm_seis_b, c = 'r')
ax21.set_ylim(-10, 50)
ax21.set_yticks(np.arange(0, 50, 10), minor=False)
ax21.set_xlim(min(tm_seis) - 0.1*(max(tm_seis) - min(tm_seis)), max(tm_seis) + 0.1*(max(tm_seis) - min(tm_seis)))
ax21.set_ylabel("Dobutamine Infusion (mcg/kg/min)")
ax21.grid()


ax22 = fig.add_subplot(224)
ax22.scatter(tm_phono, doses, c = 'b', label = "Phono")
tm_phono_m, tm_phono_b = np.polyfit(tm_phono, doses, 1)
ax22.plot(tm_phono, tm_phono_m*np.asarray(tm_phono) + tm_phono_b, c = 'b')
ax22.set_ylim(-10, 50)
ax22.set_yticks(np.arange(0, 50, 10), minor=False)
ax22.set_xlim(min(tm_phono) - 0.1*(max(tm_phono) - min(tm_phono)), max(tm_phono) + 0.1*(max(tm_phono) - min(tm_phono)))
ax22.grid()


plt.show()              

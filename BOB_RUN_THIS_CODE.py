# BOB ONLY EDIT THE SETTING SECTION
# Settings
#------------------------------------------------------------------------------------#
BLUE_PATH_IN_TERMINAL = '/mnt/c/Python Codes/HeartBreaker'

folder_name = "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL" 
area_around_echo_size = 240 # In Seconds
composite_size        = 10  # In number of heartbeats
step_size             = 5   # In number of heartbeats

save_signal          = False
preloaded_signal     = True

use_intervals        = True

save_composites      = True
preloaded_composites = True

display_intervals       = False
display_INO_composites  = False
display_LUSI_composites = False
verify_INO_labels       = False
verify_LUSI_labels      = True
#------------------------------------------------------------------------------------#
# END of Setting

# Open Virtual Envornment
activate_this = BLUE_PATH_IN_TERMINAL + '/hbenv/bin/activate_this.py'
exec(open(activate_this).read(), {'__file__': activate_this})
import sys
assert sys.prefix != sys.base_prefix, "Virual Environment not working!" 

# Imports
import os
import numpy as np
import heartbreaker as hb
import matplotlib.pyplot as plt
from composite_peaks import CompositePeaks
from ino_composite_peaks import InoCompositePeaks
from lusi_composite_peaks import LusiCompositePeaks
from files_w_doseage_and_ints import files
from verification_gui import HeartbeatVerifier
from ino_verification_gui  import InoHeartbeatVerifier
from lusi_verification_gui import LusiHeartbeatVerifier
from composite_statistics import CompositeStats
from interval_finder_gui import HeartbeatIntervalFinder

print("Folder: ", folder_name)
        
# Load Intervals
if use_intervals and not preloaded_composites: 
    files = hb.load_intervals(folder_name)

# Check Intervals
if display_intervals and not preloaded_composites:
    finder = HeartbeatIntervalFinder(files = files,
                                    folder_name = folder_name,
                                    area_around_echo_size = 240,
                                    use_intervals = use_intervals,
                                    preloaded_signal = preloaded_signal,
                                    save_signal = save_signal)

    # Use updated intervals moving forward
    files = hb.load_intervals(folder_name)

# Initalize Composite Stats
composite_statistics = {0  : CompositeStats(), 
                        10 : CompositeStats(),
                        20 : CompositeStats(),
                        30 : CompositeStats(),
                        40 : CompositeStats(), 
                        42 : CompositeStats()}

# Check Composites
for dosage in files[folder_name]:

    # Pick file
    file_name = files[folder_name][dosage][1]["file_name"]
    ino_composite_save_file_name  = "ino_composites_"  + folder_name + "_" + file_name + "_d" + str(dosage) 
    lusi_composite_save_file_name = "lusi_composites_" + folder_name + "_" + file_name + "_d" + str(dosage) 

    if preloaded_composites:
        ino_composite_peaks = InoCompositePeaks()
        ino_composite_peaks.load("data/Derived/composites/ino/"  + ino_composite_save_file_name)

        lusi_composite_peaks = LusiCompositePeaks()
        lusi_composite_peaks.load("data/Derived/composites/lusi/"  + lusi_composite_save_file_name)                                                                          

    else:
        # Load Data
        if preloaded_signal or save_signal:

            #  Load Signals
            signal_save_file_name = folder_name + "_d" + str(dosage)
            time   = np.loadtxt('data/Derived/signals/time_' + signal_save_file_name + '.csv', delimiter=',')
            signal = np.loadtxt('data/Derived/signals/signal_' + signal_save_file_name + '.csv', delimiter=',')
            seis   = np.loadtxt('data/Derived/signals/seis_' + signal_save_file_name + '.csv', delimiter=',')
            phono  = np.loadtxt('data/Derived/signals/phono_' + signal_save_file_name + '.csv', delimiter=',')

            # Take interval
            interval = [min(np.searchsorted(time, int(x)), len(time) - 1) for x in files[folder_name][dosage][1]["intervals"][1]]
            interval = range(interval[0], interval[1])

            time   = time[interval]
            signal = signal[interval]
            seis   = seis[interval]
            phono  = phono[interval]

        else:
            time, signal, seis, _, phono, _ = hb.load_file_data(files = files, 
                                                                folder_name = folder_name, 
                                                                dosage = dosage, 
                                                                file_number = 1)

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
                                            seis1  = seis,
                                            phono1 = phono)
        peaks.get_inital_statistics()

        # Build Composites
        ino_composite_peaks = InoCompositePeaks(peaks)
        ino_composite_peaks.get_N_composite_signal_dataset(composite_size, step_size, display = display_INO_composites, dosage = dosage)
        ino_composite_peaks.update_composite_peaks(dosage = dosage)
        ino_composite_peaks.get_inital_statistics()

        lusi_composite_peaks = LusiCompositePeaks(peaks)
        lusi_composite_peaks.get_N_composite_signal_dataset(composite_size, step_size, display = display_LUSI_composites, dosage = dosage)
        lusi_composite_peaks.update_composite_peaks(dosage = dosage)
        lusi_composite_peaks.get_inital_statistics()

    if save_composites:
        # Get File Name
        ino_save_file_name = "ino_composites_" + folder_name + "_" + file_name + "_d" + str(dosage) 
        lusi_save_file_name = "lusi_composites_" + folder_name + "_" + file_name + "_d" + str(dosage) 

        # Save
        ino_composite_peaks.save("data/Derived/composites/ino/"   + ino_save_file_name)
        lusi_composite_peaks.save("data/Derived/composites/lusi/" + lusi_save_file_name)

    # Verify Composites
    if verify_INO_labels:
        InoHeartbeatVerifier(ino_composite_peaks, 
                            folder_name = folder_name,
                            dosage      = dosage,
                            file_name   = file_name,
                            interval_number = 1)

        ino_composite_peaks.load("data/Derived/composites/ino/"  + ino_composite_save_file_name)
    
    if verify_LUSI_labels:
        LusiHeartbeatVerifier(lusi_composite_peaks, 
                            folder_name = folder_name,
                            dosage      = dosage,
                            file_name   = file_name,
                            interval_number = 1)

        lusi_composite_peaks.load("data/Derived/composites/lusi/"  + lusi_composite_save_file_name)

    # Add data to stats
    composite_statistics[dosage].add_ino_data(ino_composite_peaks)
    composite_statistics[dosage].add_lusi_data(lusi_composite_peaks)

    


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

        qm_seis.append(4000/composite_statistics[dosage].QM_interval_seis)
        qm_phono.append(4000/composite_statistics[dosage].QM_interval_phono)

        tm_seis.append(4000/composite_statistics[dosage].TM_interval_seis)
        tm_phono.append(4000/composite_statistics[dosage].TM_interval_phono)


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

# Maximize frame
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()

plt.show()              

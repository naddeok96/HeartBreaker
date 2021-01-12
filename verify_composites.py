
# Imports
import os
import numpy as np
import heartbreaker as hb
import matplotlib.pyplot as plt
from files_w_doseage_and_ints import files
from composite_peaks import CompositePeaks
from verification_gui import HeartbeatVerifier
from composite_statistics import CompositeStats


os.chdir(os.getcwd() + "/data/Derived/composites")
composite_statistics = {0  : CompositeStats(), 
                        10 : CompositeStats(),
                        20 : CompositeStats(),
                        30 : CompositeStats(),
                        40 : CompositeStats(), 
                        42 : CompositeStats()}

folder_name = "1 9 2020 AH TDMS ESSENTIAL" # "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL"
print("Folder: ", folder_name)
# Iterate
for dosage in files[folder_name].keys():
    # if dosage != 20 or dosage != 40:
    #     continue

    print("Doseage: ", dosage)

    if files[folder_name][dosage] is None:
        continue

    for file_number in files[folder_name][dosage]:

        print("File: ", files[folder_name][dosage][file_number]["file_name"])

        for interval_number in files[folder_name][dosage][file_number]["intervals"]:

            print("Interval: ", interval_number)

            # Pick file
            file_name = files[folder_name][dosage][file_number]["file_name"]
            save_file_name = folder_name + "_d" + str(dosage) + "_" + file_name + "_i" + str(interval_number)

            # Load Composites
            if os.path.isfile(save_file_name + '.pkl'):

                temp_composite_peaks = CompositePeaks()
                temp_composite_peaks.load(save_file_name)

                # Verify
                # verifier = HeartbeatVerifier(temp_composite_peaks, 
                #                                 folder_name = folder_name,
                #                                 dosage      = dosage,
                #                                 file_name   = file_name,
                #                                 interval_number = interval_number)

                # temp_composite_peaks.load(save_file_name)

                # Accumulate Stats
                composite_statistics[dosage].add_data(temp_composite_peaks)

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



        










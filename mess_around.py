# Imports
import numpy as np
import matplotlib.pyplot as plt  
import heartbreaker as hb
import os
# os.chdir("../wfdb_python/")
# os.chdir("../")
import wfdb
# print(wfdb.__path__.__dict__["_path"][0])
# os.chdir("Heartbreaker")
# os.chdir("../Heartbreaker/")

# Settings
#---------------------------------------------------#
folder_name = "data/MIT-BIH Long-Term ECG Database" # "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL" 
set_name    = '14046'
interval_size = 10 # [s]
#---------------------------------------------------#

# Load data
# data = np.loadtxt( folder_name + "/" + set_name )
# data = np.fromfile( folder_name + "/" + set_name , dtype="byte")
record = wfdb.rdsamp(folder_name + "/" + set_name)
signals, fields = wfdb.rdsamp(folder_name + "/" + set_name)
print(fields.keys())
print("Freq: ", fields["fs"])
print("Sig Len: ", fields["sig_len"])
total_time = fields["sig_len"]/fields["fs"]
num_intervals = int(total_time/interval_size)
for i in range(num_intervals):
    # plt.plot(signals[(0 + i*(interval_size*fields["fs"])):interval_size*fields["fs"] + i*(interval_size*fields["fs"]), 0 ])
    # plt.show()
    signal = signals[0 + i*(interval_size*fields["fs"]):interval_size*fields["fs"] + i*(interval_size*fields["fs"]), 0 ]
    time = np.linspace(0 + i*(interval_size), interval_size + i*(interval_size), num = len(signal))

    hb.get_ecg_peaks_v2(time = time, 
                        signal = signal,
                        dosage = 0,
                        plot  = True,
                        num_pts = 1)

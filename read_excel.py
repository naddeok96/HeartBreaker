# Imports
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from heartbreaker import HeartBreak

# Initalize heartbreaker
hb = HeartBreak()

# Hyperparameters
folder_name = "1 9 2020 AH TDMS ESSENTIAL"
dosages     = ["10", "20", "30", "40"]

# Find all files in folder
os.chdir('data/Derived')
path = os.getcwd()
files = [f for f in os.listdir(path) if (f[-3:] == 'xls' and f[:len(folder_name)] == folder_name)]

# Load peaks
phase_data = []
peaks = {"P": {}, "Q": {}, "R": {}, "S": {}, "T": {}}
# cycle through files
for dose in dosages:
    t_rp_mean, t_rp_std, t_rq_mean, t_rq_std, t_rs_mean, t_rs_std, t_rt_mean, t_rt_std, tn = 0,0,0,0,0,0,0,0,0
    for i, file_name in enumerate(files):
        if "d" + dose in file_name:
            df = pd.ExcelFile(file_name).parse('Peaks')

            # Cycle through peaks
            for peak_name in peaks:
                peaks[peak_name][file_name] = df[peak_name].to_numpy()

        # Find Phase Space Data
        for file_name in peaks["P"]:
            # Collect data for file
            p = peaks["P"][file_name]
            q = peaks["Q"][file_name]
            r = peaks["R"][file_name]
            s = peaks["S"][file_name]
            t = peaks["T"][file_name]

            
            # Calculate phase ratios for file
            phase_ratios = hb.get_phase_ratios_to_RR_intervals(p,q,r,s,t)
            if i == 0:
                # Calulate files weight
                tn = len(r) - 1
            else:
                # Calulate files weight
                weight = len(r) - 1
                
                # Combine stats
                t_rp_mean, t_rp_std, _  = hb.add_stats(t_rp_mean, t_rp_std, tn, phase_ratios[0], phase_ratios[1], weight)
                t_rq_mean, t_rq_std, _  = hb.add_stats(t_rq_mean, t_rq_std, tn, phase_ratios[2], phase_ratios[3], weight)
                t_rs_mean, t_rs_std, _  = hb.add_stats(t_rs_mean, t_rs_std, tn, phase_ratios[4], phase_ratios[5], weight)
                t_rt_mean, t_rt_std, tn = hb.add_stats(t_rt_mean, t_rt_std, tn, phase_ratios[6], phase_ratios[7], weight)

    phase_data.append((t_rp_mean, t_rp_std, t_rq_mean, t_rq_std, t_rs_mean, t_rs_std, t_rt_mean, t_rt_std))

# Display
hb.plot_phase_space_for_dosages(folder_name, phase_data)

        
    

    
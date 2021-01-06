# Imports
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import heartbreaker as hb

# Hyperparameters
folder_name =  "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL" # "Dobutamine Stress Test 62719JS" # 
dosages     = ["0", "10", "20", "30", "40"]

# Find all files in folder
os.chdir('data/Derived')
path = os.getcwd()
files = [f for f in os.listdir(path) if (f[-3:] == 'xls' and f[:len(folder_name)] == folder_name)]

# Load peaks
phase_data = []
peaks = {"P": {}, "Q": {}, "R": {}, "S": {}, "T": {}, "S-T Start": {}, "S-T End": {}, "T''max": {}, "Q-M Seis I": {}, "T-M Seis I": {}}
t_rp_mean, t_rp_std, t_rq_mean, t_rq_std, t_rr_mean, t_rr_std, t_rs_mean, t_rs_std, t_rt_mean, t_rt_std, t_st_seg_mean, t_st_seg_std, t_tddot_mean, t_tddot_std, t_qm_seis1_mean, t_qm_seis1_std, t_tm_seis1_mean, t_tm_seis1_std, tn = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
# cycle through files
for dose in dosages:
    print("D: ", dose)
    dosage_files = [file_name for file_name in files if "d" + dose in file_name] 
    
    for i, file_name in enumerate(dosage_files):
        print("I: ", i)
        
        df = pd.ExcelFile(file_name).parse('Peaks')

        # Cycle through peaks
        for peak_name in peaks:
            peaks[peak_name][file_name] = df[peak_name].to_numpy()

        # Find Phase Space Data
        # Collect data for file
        p = peaks["P"][file_name]
        q = peaks["Q"][file_name]
        r = peaks["R"][file_name]
        s = peaks["S"][file_name]
        t = peaks["T"][file_name]
        st_start = peaks["S-T Start"][file_name]
        st_end = peaks["S-T End"][file_name]
        t_ddot = peaks["T''max"][file_name]
        qm_seis1 = peaks["Q-M Seis I"][file_name]
        tm_seis1 = peaks["T-M Seis I"][file_name]

        # Calculate phase ratios for file
        phase_ratios = hb.get_phase_ratios_to_RR_intervals(p,q,r,s,t,st_start,st_end,t_ddot, qm_seis1, tm_seis1)

        if i == 0:
            # Calulate files weight
            tn = len(r) - 1

            # Store
            t_rp_mean, t_rp_std  = phase_ratios[0], phase_ratios[1]
            t_rq_mean, t_rq_std  = phase_ratios[2], phase_ratios[3]
            t_rr_mean, t_rr_std  = phase_ratios[4], phase_ratios[5]
            t_rs_mean, t_rs_std  = phase_ratios[6], phase_ratios[7]
            t_rt_mean, t_rt_std  = phase_ratios[8], phase_ratios[9]
            t_st_seg_mean, t_st_seg_std = phase_ratios[10], phase_ratios[11]
            t_tddot_mean, t_tddot_std = phase_ratios[12], phase_ratios[13]
            t_qm_seis1_mean, t_qm_seis1_std = phase_ratios[14], phase_ratios[15]
            t_tm_seis1_mean, t_tm_seis1_std = phase_ratios[16], phase_ratios[17]

        else:
            # Calulate files weight
            weight = len(r) - 1
            
            # Combine stats
            t_rp_mean, t_rp_std, _  = hb.add_stats(t_rp_mean, t_rp_std, tn, phase_ratios[0], phase_ratios[1], weight)
            t_rq_mean, t_rq_std, _  = hb.add_stats(t_rq_mean, t_rq_std, tn, phase_ratios[2], phase_ratios[3], weight)
            t_rr_mean, t_rr_std, _  = hb.add_stats(t_rr_mean, t_rr_std, tn, phase_ratios[4], phase_ratios[5], weight)
            t_rs_mean, t_rs_std, _  = hb.add_stats(t_rs_mean, t_rs_std, tn, phase_ratios[6], phase_ratios[7], weight)
            t_rt_mean, t_rt_std, _  = hb.add_stats(t_rt_mean, t_rt_std, tn, phase_ratios[8], phase_ratios[9], weight)
            t_st_seg_mean, t_st_seg_std, _  = hb.add_stats(t_st_seg_mean, t_st_seg_std, tn, phase_ratios[10], phase_ratios[11], weight)
            t_tddot_mean, t_tddot_std, _ = hb.add_stats(t_tddot_mean, t_tddot_std, tn, phase_ratios[12], phase_ratios[13], weight)
            t_qm_seis1_mean, t_qm_seis1_std, _ = hb.add_stats(t_qm_seis1_mean, t_qm_seis1_std, tn, phase_ratios[14], phase_ratios[15], weight)
            t_tm_seis1_mean, t_tm_seis1_std, tn = hb.add_stats(t_tm_seis1_mean, t_tm_seis1_std, tn, phase_ratios[16], phase_ratios[17], weight)

    phase_data.append((t_rp_mean, t_rp_std, t_rq_mean, t_rq_std, t_rr_mean, t_rr_std, t_rs_mean, t_rs_std, t_rt_mean, t_rt_std, t_st_seg_mean, t_st_seg_std, t_tddot_mean, t_tddot_std, t_qm_seis1_mean, t_qm_seis1_std, t_tm_seis1_mean, t_tm_seis1_std, tn))

# Display
# hb.plot_phase_space_for_dosages(folder_name, phase_data)
qm_seis1es = [phase_data[0][-5], phase_data[1][-5], phase_data[2][-5], phase_data[3][-5], phase_data[4][-5]]
tm_seis1es = [phase_data[0][-3], phase_data[1][-3], phase_data[2][-3], phase_data[3][-3], phase_data[4][-3]]
ino_strains  = [-1.47, -2.0, -2.5, -2.73, -3.67]
lusi_strains = [ 1.87, 2.30, 2.83,  3.50,  3.67]
hb.plot_em_vs_strain_rate(dosages, qm_seis1es, tm_seis1es, ino_strains, lusi_strains)

        
    

    
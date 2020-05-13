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

# Find all files in folder
os.chdir('data/Derived')
path = os.getcwd()
files = [f for f in os.listdir(path) if f[-3:] == 'xls']

# Load peaks
peaks = {"P": {}, "Q": {}, "R": {}, "S": {}, "T": {}}
# cycle through files
for file_name in files:
    df = pd.ExcelFile(file_name).parse('Peaks')

    # Cycle through peaks
    for peak_name in peaks:
        peaks[peak_name][file_name] = df[peak_name].to_numpy()

# Find Phase Space Data
t_rp_mean, t_rp_std, t_rq_mean, t_rq_std, t_rs_mean, t_rs_std, t_rt_mean, t_rt_std, tn = 0,0,0,0,0,0,0,0,0
for i, file_name in enumerate(peaks["P"]):
    # Collect data for file
    p = peaks["P"][file_name]
    q = peaks["Q"][file_name]
    r = peaks["R"][file_name]
    s = peaks["S"][file_name]
    t = peaks["T"][file_name]

    # Calculate phase data
    if i == 0:
        # Calculate phase ratios for file
        t_rp_mean, t_rp_std, t_rq_mean, t_rq_std, t_rs_mean, t_rs_std, t_rt_mean, t_rt_std = hb.get_ratios_to_RR_intervals(p,q,r,s,t)

        # Calulate files weight
        tn = len(r) - 1
    else:
        # Calculate phase ratios for file
        rp_mean, rp_std, rq_mean, rq_std, rs_mean, rs_std, rt_mean, rt_std = hb.get_ratios_to_RR_intervals(p,q,r,s,t)

        # Calulate files weight
        weight = len(r) - 1

        # Combine stats
        t_rp_mean, t_rp_std, _  = hb.add_stats(t_rp_mean, t_rp_std, tn, rp_mean, rp_std, weight)
        t_rq_mean, t_rq_std, _  = hb.add_stats(t_rq_mean, t_rq_std, tn, rq_mean, rq_std, weight)
        t_rs_mean, t_rs_std, _  = hb.add_stats(t_rs_mean, t_rs_std, tn, rs_mean, rs_std, weight)
        t_rt_mean, t_rt_std, tn = hb.add_stats(t_rt_mean, t_rt_std, tn, rt_mean, rt_std, weight)

# Display
hb.plot_phase_space("Phase Space",
                t_rp_mean, t_rp_std,
                t_rq_mean, t_rq_std,
                t_rs_mean, t_rs_std,
                t_rt_mean, t_rt_std)

        
    

    
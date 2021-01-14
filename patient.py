'''
This class will read a patients tdms file
'''
# Imports
from nptdms import TdmsFile as tdms
import numpy as np
import math
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Class
class Patient:

    def __init__(self, working_directory,
                       file_name):
        
        super(Patient, self).__init__()

        os.chdir(working_directory)
        self.patient_name = file_name
        self.tdms_file = tdms.read(file_name + ".tdms")
        os.chdir("../../Derived")

        self.times  = self.tdms_file['Data']['Time(S)'].data
        self.seis1  = self.tdms_file['Data']['Seismocardiogram I'].data
        self.seis2  = self.tdms_file['Data'][' Seismocardiogram II'].data
        self.phono1 = self.tdms_file['Data'][' Precordial Phonocardiogram-I'].data
        self.phono2 = self.tdms_file['Data'][' Precordial Phonocardiogram-II'].data
        self.ecg    = self.tdms_file['Data'][' Direct ECG'].data

        self.total_time = math.floor(np.max(self.times)) - math.floor(np.min(self.times))
        self.num_data_points = len(self.times)

        self.time_step = self.total_time/self.num_data_points
        self.frequency = 4000

    def get_interval(self, interval):
        '''
        Return an interval of the signal
        '''
        return self.times[interval], self.ecg[interval], self.seis1[interval], self.seis2[interval], self.phono1[interval], self.phono2[interval]

'''
This class will read a patients tdms file
'''
# Imports
from nptdms import TdmsFile as tdms
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Class
class Patient:

    def __init__(self, working_directory,
                       file_name,
                       frequency = 4000):
        
        super(Patient, self).__init__

        os.chdir(working_directory)
        self.patient_name = file_name
        self.tdms_file = tdms(file_name + ".tdms")
        os.chdir("../Derived")

        self.times  = self.tdms_file.object('Data','Time(S)').data
        self.seis1  = self.tdms_file.object('Data','Seismocardiogram I').data
        self.seis2  = self.tdms_file.object('Data',' Seismocardiogram II').data
        self.phono1 = self.tdms_file.object('Data',' Precordial Phonocardiogram-I').data
        self.phono2 = self.tdms_file.object('Data',' Precordial Phonocardiogram-II').data
        self.ecg    = self.tdms_file.object('Data',' Direct ECG').data

        self.num_data_points = len(self.times)
        self.frequency = frequency

    def normalize(self, signal):
        '''
        Normalize a signal 
        '''
        mean = np.mean(signal)
        std = np.std(signal)
        return (signal - mean) / std

    def get_interval(self, interval):
        '''
        Return an interval of the signal
        '''
        return self.times[interval], self.normalize(self.ecg[interval])

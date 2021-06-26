# Imports
import torch
import wfdb
# from ecgdetectors import Detectors
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.data import DataLoader, random_split


# Class
class MITBIHLongTermDataset(torch.utils.data.Dataset):
    """Pytorch Dataloader for MIT-BIH Long-Term ECG Database data."""

    def __init__(self,  folder_name = "data/MIT-BIH Long-Term ECG Database", # "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL" 
                        set_name    = '14046',
                        signal_length = 300,
                        show_annotation_labels = False,
                        show_regular_and_irregular = False, 
                        save_data = False,
                        preloaded_data = True):

        # Give access to methods and properties of a parent or sibling class
        super(MITBIHLongTermDataset,self).__init__()

        # Parameters
        #---------------------------------------------------#
        self.folder_name   = folder_name
        self.set_name      = set_name
        self.signal_length = signal_length
        #---------------------------------------------------#

        # Load data from wfdb
        self.record                   = wfdb.rdsamp(folder_name + "/" + set_name + "/" + set_name)
        self.signal_data, self.fields = wfdb.rdsamp(folder_name + "/" + set_name + "/" + set_name)
        self.annotation               = wfdb.rdann(folder_name + "/" + set_name + "/" + set_name, 'atr', summarize_labels = True)

        # Find RR Intervals
        self.r_peaks = np.unique(self.annotation.sample) 
        rr_intervals = np.zeros(((len(self.r_peaks) - 1), 2))
        for i in range(len(self.r_peaks) - 1):
            rr_intervals[i, :] = self.r_peaks[i],  self.r_peaks[i + 1]
            
        # Get Boolean of Regualar Beats
        regular_beats = np.in1d(self.annotation.symbol, ["N"])[:-1]

        # Remove Regular beats just prior to irregular beats for safety
        regular_beats = np.asarray([i==2 for i in regular_beats + np.append(regular_beats[1:], 1)])

        # Split regular and irregular beats
        self.regular_rr_intervals   = torch.tensor(rr_intervals[regular_beats, :])
        self.irregular_rr_intervals = torch.tensor(rr_intervals[np.invert(regular_beats), :])

        # Upsample the irregular
        upsample = None
        for i in range(math.floor(self.regular_rr_intervals.size(0)/self.irregular_rr_intervals.size(0))):
            if i == 0:
                upsample = self.irregular_rr_intervals
            else:
                upsample = torch.cat((upsample, self.irregular_rr_intervals), dim = 0)

        self.irregular_rr_intervals = upsample

        # Declare Labels
        regular_labels = torch.cat((torch.ones(self.regular_rr_intervals.size(0)).view(-1,1), torch.zeros(self.regular_rr_intervals.size(0)).view(-1, 1)), dim = 1)
        irregular_labels = torch.cat((torch.zeros(self.irregular_rr_intervals.size(0)).view(-1,1), torch.ones(self.irregular_rr_intervals.size(0)).view(-1, 1)), dim = 1)

        self.labels = torch.cat((regular_labels, irregular_labels), dim = 0)

        # Save QRS location of Irregular beats for trouble shooting 
        # irregular_beats_qrs = np.unique(self.annotation.sample[np.invert(np.in1d(self.annotation.symbol, ["N"]))])

        if preloaded_data:
            # Define File Names
            filename  = self.folder_name + "/" + self.set_name  + "/" + self.set_name 
            
            # Save Signals
            self.signals = torch.load(filename + "_signal.pt")

            # Save Labels
            self.labels = torch.load(filename + "_labels.pt")

        else:

            # Load individual signals
            self.signals = torch.empty_like(torch.empty((self.regular_rr_intervals.size(0) + self.irregular_rr_intervals.size(0) , signal_length)))
            
            # Regular signals
            print("Working on Regular Beats")
            for i in range(self.regular_rr_intervals.size(0)):
                signal = self.signal_data[range(int(self.regular_rr_intervals[i, 0]), int(self.regular_rr_intervals[i, 1])), 0]
                signal = (signal - signal.mean())/signal.std()

                signal = np.interp(np.linspace(int(self.regular_rr_intervals[i, 0]), int(self.regular_rr_intervals[i, 1]), num=signal_length), range(int(self.regular_rr_intervals[i, 0]), int(self.regular_rr_intervals[i, 1])), signal)
                
                self.signals[i, :] = torch.tensor(signal)
            
            print("Working on Irregular Beats")
            # Irregular signals
            for i in range(self.irregular_rr_intervals.size(0)):
                signal = self.signal_data[range(int(self.irregular_rr_intervals[i, 0]), int(self.irregular_rr_intervals[i, 1])), 0]
                signal = (signal - signal.mean())/signal.std()

                signal = np.interp(np.linspace(int(self.irregular_rr_intervals[i, 0]), int(self.irregular_rr_intervals[i, 1]), num=signal_length), range(int(self.irregular_rr_intervals[i, 0]), int(self.irregular_rr_intervals[i, 1])), signal)
                
                self.signals[self.regular_rr_intervals.size(0) + i, :] = torch.tensor(signal)

            # Load Signals in standard sizes
            self.signals = self.signals.view(-1, 1, signal_length)

        
            # Save Data
            if save_data:
                # Define File Names
                filename  = self.folder_name + "/" + self.set_name + "/" + self.set_name 
                
                # Save Signals
                torch.save(self.signals, filename + "_signal.pt")

                # Save Labels
                torch.save(self.labels, filename + "_labels.pt")

        # Display Annotation Lables
        if show_annotation_labels:
            self.display_annotaion_labels()

        # Display regular/irregular windowing
        if show_regular_and_irregular:
            self.display_regular_and_irregular()

    # Display data fields
    def display_annotaion_labels(self):
        # Show Field Attributes
        print("-------------------------------------------------------------")
        print("Avaliable dataset info keys under self.fields attrbute: ")
        for key in self.fields.keys():
            print(key, ": ", self.fields[key])
        print("-------------------------------------------------------------")

        # Show Annotation Lables
        wfdb.show_ann_labels()

        # Display Freq Table
        def CountFrequency(my_list):
            # Creating an empty dictionary
            freq = {}
            for items in my_list:
                freq[items] = my_list.count(items)
            
            for key, value in freq.items():
                print(key, value)

        CountFrequency(self.annotation.symbol)
    
    # Plot Regular and Irregular Windows
    def display_regular_and_irregular(self):
        
        signal = self.signal_data[:, 0]
        plt.plot(signal)
        print("Plotted Signal")
        # plt.scatter(r_peaks, signal[r_peaks], marker = "D", color= "red", label="R peaks")
        print("Plotted R peaks")
        
        for j in range(self.regular_rr_intervals.size(0)):
            plt.axvspan(self.regular_rr_intervals[j, 0], self.regular_rr_intervals[j, 1], color = "green", alpha=0.15)  
            if j > 1e3:
                break      
        print("Plotted Regular RR")

        for j in range(self.irregular_rr_intervals.size(0)):    
            plt.axvspan(self.irregular_rr_intervals[j, 0], self.irregular_rr_intervals[j, 1], color = "red", alpha=0.15)
            if j > 1e3:
                break    
        print("Plotted Irregular RR")

        # plt.scatter(irregular_beats_qrs, signal[irregular_beats_qrs] + 1, marker = "o", color= "g", label="Irregular HB")
        # print(self.annotation.sample)
        # plt.scatter(self.annotation.sample, signal[self.annotation.sample])
        # print(type(self.annotation.symbol))
        
        # print("Plotted QRS of Irregular Beats")
        plt.show()

        plt.plot(self.signals[0, 0, :])
        plt.title(self.labels[0])
        plt.show()

    # Return Number of Observations
    def __len__(self):
        return int(self.signals.size(0))

    # Return Observation
    def __getitem__(self, idx):
        return self.labels[idx, :], self.signals[idx, :, :]

# Uncomment to test
# MITBIHLongTermDataset(show_annotation_labels = False,
#                         show_regular_and_irregular = True, 
#                         save_data = False,
#                         preloaded_data = False)
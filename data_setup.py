# Imports
from torch.utils.data import DataLoader, random_split
from data_loader import MITBIHLongTermDataset
import numpy as np
import torch
import math

class Data:
    """Opens, splits and puts data in a dataloader"""
    def __init__(self,  folder_name = "data/MIT-BIH Long-Term ECG Database", # "ECG-Phono-Seismo DAQ Data 8 20 2020 2" # "1 9 2020 AH TDMS ESSENTIAL" 
                        set_name    = '14046',
                        signal_length = 300,
                        gpu = False,
                        test_batch_size  = 100,
                        train_percentage = 0.8):
        """
        Args:
            csv_file (string): Relative path to the csv file
            gpu (bool, optional): If True use GPU's. Defaults to False.
            test_batch_size (int, optional): Size of batches in the test set. Defaults to 100.
            train_percentage (float, optional): Percentage of dataset to be allocated for training. Defaults to 0.8.
            category(int, optional): Column to have an autoencoder dataset. 
                                        # 1 is year   (100 Types)
                                        # 2 is month
                                        # 3 is day
                                        # 4 is geoid
                                        # Defaults to None.
        """
        super(Data, self).__init__()

        # Hyperparameters
        self.folder_name = folder_name
        self.set_name    = set_name
        self.gpu = gpu
        self.test_batch_size = test_batch_size

        # Pull in data
        self.dataset = MITBIHLongTermDataset(folder_name = self.folder_name,
                                            set_name    = self.set_name,
                                            signal_length = signal_length)
        print("Dataset Loaded")

        # Split Data
        train_proportion = math.ceil(train_percentage * len(self.dataset))
        test_proportion  = len(self.dataset) - train_proportion

        # Split data
        self.train_set, self.test_set = random_split(self.dataset, [train_proportion, test_proportion])
        
        #Test and validation loaders have constant batch sizes, so we can define them 
        if self.gpu:
            self.test_loader = DataLoader(self.test_set, 
                                            batch_size = test_batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            pin_memory = True)
        else:
            self.test_loader = DataLoader(self.test_set, 
                                            batch_size = test_batch_size,
                                            shuffle = False)

    # Fucntion to break training set into batches
    def get_train_loader(self, batch_size):
        """Loads training data into a dataloader

        Args:
            batch_size (int): Size of batches in the train set

        Returns:
            Pytorch Dataloader object: Training data in a dataloader
        """
        if self.gpu:
            self.train_loader = DataLoader(self.train_set,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8,
                                            pin_memory=True)
        else:
            self.train_loader = DataLoader(self.train_set, 
                                            batch_size = batch_size,
                                            shuffle = True)
            

        return self.train_loader


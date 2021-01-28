import numpy as np
from variable_statistics import Variable

class CompositeStats():

    def __init__(self):

        super(CompositeStats, self).__init__()

        self.Q = Variable([])
        self.ddT = Variable([])

        self.QM_seis  = Variable([])
        self.QM_phono = Variable([])
        
        self.TM_seis  = Variable([])
        self.TM_phono = Variable([])

        self.QM_interval_seis = None
        self.QM_interval_phono = None

        self.TM_interval_seis = None
        self.TM_interval_phono = None

    def _add_statistics(self, current_variable, new_variable):
        '''
        Takes stats of two sets (assumed to be from the same distribution) and combines them
        Method from https://www.statstodo.com/CombineMeansSDs_Pgm.php
        '''
        if current_variable.sample_size is None:
            return new_variable.mean, new_variable.std, new_variable.sample_size

        elif new_variable.sample_size is None:
            return current_variable.mean, current_variable.std, current_variable.sample_size

        else:
            mean1   = current_variable.mean
            std1    = current_variable.std
            weight1 = current_variable.sample_size
            
            mean2   = new_variable.mean
            std2    = new_variable.std
            weight2 = new_variable.sample_size

            # Calculate E[x] and E[x^2] of each
            sig_x1 = weight1 * mean1
            sig_x2 = weight2 * mean2

            sig_xx1 = ((std1 ** 2) * (weight1 - 1)) + (((sig_x1 ** 2) / weight1))
            sig_xx2 = ((std2 ** 2) * (weight2 - 1)) + (((sig_x2 ** 2) / weight2))

            # Calculate sums
            tn  = weight1 + weight2
            tx  = sig_x1  + sig_x2
            txx = sig_xx1 + sig_xx2

            # Calculate combined stats
            mean = tx / tn
            std = np.sqrt((txx - (tx**2)/tn) / (tn - 1))

            return mean, std, tn

    def add_data(self, composite_peaks):
        # Update data
        self.Q.mean, self.Q.std, self.Q.sample_size = self._add_statistics(self.Q, composite_peaks.Q)
        self.ddT.mean, self.ddT.std, self.ddT.sample_size = self._add_statistics(self.ddT, composite_peaks.ddT)

        self.QM_seis.mean , self.QM_seis.std, self.QM_seis.sample_size = self._add_statistics(self.QM_seis, composite_peaks.QM_seis)
        self.QM_phono.mean, self.QM_phono.std, self.QM_phono.sample_size = self._add_statistics(self.QM_phono, composite_peaks.QM_phono)
        
        self.TM_seis.mean , self.TM_seis.std, self.TM_seis.sample_size = self._add_statistics(self.TM_seis, composite_peaks.TM_seis)
        self.TM_phono.mean, self.TM_phono.std, self.TM_phono.sample_size = self._add_statistics(self.TM_phono, composite_peaks.TM_phono)
        
        # Update intervals
        self.QM_interval_seis  = self.QM_seis.mean - self.Q.mean
        self.QM_interval_phono = self.QM_phono.mean - self.Q.mean

        self.TM_interval_seis  = self.TM_seis.mean - self.ddT.mean
        self.TM_interval_phono = self.TM_phono.mean - self.ddT.mean

    def add_ino_data(self, composite_peaks):
        # Update data
        self.Q.mean, self.Q.std, self.Q.sample_size = self._add_statistics(self.Q, composite_peaks.Q)
        
        self.QM_seis.mean , self.QM_seis.std, self.QM_seis.sample_size = self._add_statistics(self.QM_seis, composite_peaks.QM_seis)
        self.QM_phono.mean, self.QM_phono.std, self.QM_phono.sample_size = self._add_statistics(self.QM_phono, composite_peaks.QM_phono)
        
        # Update intervals
        self.QM_interval_seis  = self.QM_seis.mean - self.Q.mean
        self.QM_interval_phono = self.QM_phono.mean - self.Q.mean
        
    def add_lusi_data(self, composite_peaks):
        # Update data
        self.ddT.mean, self.ddT.std, self.ddT.sample_size = self._add_statistics(self.ddT, composite_peaks.ddT)

        self.TM_seis.mean , self.TM_seis.std, self.TM_seis.sample_size = self._add_statistics(self.TM_seis, composite_peaks.TM_seis)
        self.TM_phono.mean, self.TM_phono.std, self.TM_phono.sample_size = self._add_statistics(self.TM_phono, composite_peaks.TM_phono)
        
        # Update intervals
        self.TM_interval_seis  = self.TM_seis.mean - self.ddT.mean
        self.TM_interval_phono = self.TM_phono.mean - self.ddT.mean
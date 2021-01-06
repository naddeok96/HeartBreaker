import numpy as np

class Variable():

    def __init__(self, data = []):

        super(Variable, self).__init__()

        self.data = data

        self._get_inital_statistics()

    def _get_inital_statistics(self):

        if len(self.data) != 0:
            self.mean = np.mean(self.data)
            self.std  = np.std(self.data)
            self.sample_size = len(self.data)
        else:
            self.mean = None
            self.std  = None
            self.sample_size = None

    def _add_statistics(mean1, std1, weight1, mean2, std2, weight2):
        '''
        Takes stats of two sets (assumed to be from the same distribution) and combines them
        Method from https://www.statstodo.com/CombineMeansSDs_Pgm.php
        '''
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

    def update_statistics(self, new_data):

        # Calculate new datas statistics
        mean        = np.mean(new_data)
        std         = np.std(new_data)
        sample_size = len(new_data)

        # Add old and new statistics
        self.mean, self.std, self.sample_size = self._add_stats(self.mean, self.std, self.sample_size,
                                                                mean, std, sample_size)

        self.data.append(new_data)



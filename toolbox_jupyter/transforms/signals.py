import numpy as np
from pywt import dwt
from scipy.signal import medfilt
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


class FilterTransform(BaseEstimator, TransformerMixin):

    def __init__(self, size, mode='avg'):
        self.size = size
        self.mode = mode

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        if self.mode == 'avg':
            return np.apply_along_axis((lambda x: np.correlate(x, np.ones(self.size) / self.size, mode='same')), 1, x)
        else:
            np.apply_along_axis((lambda x: medfilt(x, self.size)), 1, x)


class ScaleTransform(BaseEstimator, TransformerMixin):

    def __init__(self, method='default'):
        self.method = method

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        if self.method == 'max':
            return np.apply_along_axis((lambda x: preprocessing.maxabs_scale(x)), 1, x)
        elif self.method == 'minmax':
            return np.apply_along_axis((lambda x: preprocessing.minmax_scale(x)), 1, x)
        elif self.method == 'robust':
            return np.apply_along_axis((lambda x: preprocessing.robust_scale(x)), 1, x)
        else:
            return np.apply_along_axis((lambda x: preprocessing.scale(x)), 1, x)


class DWTTransform(BaseEstimator, TransformerMixin):
    """Class that manage a DWT Transform

     This class is made the same way as sklearn transform to be fit in a Pipe

     Attributes:
         mode (:obj:'str'): A mode for DWT extraction.

     """

    def __init__(self, mode='db1', segment=-1):
        """Make an initialisation of DWTTransform object.

        Take a string that represent extraction mode, default use 'db1'

        Args:
             mode (:obj:'str'): The mode as string.
        """
        self.mode = mode
        self.segment = segment

    def fit(self, x, y=None):
        """
        This should fit this transformer, but DWT doesn't need to fit to train data

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
        """
        return self

    def transform(self, x):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
        """
        features = None
        spectrum_length = x.shape[1]
        for start in range(0, spectrum_length, self.segment):
            approx, _ = dwt(x[:, start:start+self.segment], self.mode)
            if features is None:
                features = approx
            else:
                features = np.concatenate((features, approx), axis=1)
        return features

    #
    # @staticmethod
    # def integrate(inputs, tags):
    #     self.data['datum'] = self.data.apply(lambda x: [0, np.trapz(x['datum'], x['wavelength'])], axis=1)
    #
    #
    # @staticmethod
    # def norm_patient_by_healthy(self):
    #     query = self.to_query(self.filters)
    #     if query:
    #         data = self.data.query(query)
    #     else:
    #         data = self.data
    #
    #     for name, group in data.groupby(self.tags['group']):
    #         # Get features by group
    #         row_ref = group[group.label == 'Sain']
    #         if len(row_ref) == 0:
    #             data.drop(group.index)
    #             continue
    #         mean = np.mean(row_ref.iloc[0]['datum'])
    #         std = np.std(row_ref.iloc[0]['datum'])
    #         for index, current in group.iterrows():
    #             data.iat[index, data.columns.get_loc('datum')] = (current['datum'] - mean) / std
    #
    #
    # @staticmethod
    # def norm_patient(self):
    #     query = self.to_query(self.filters)
    #     if query:
    #         data = self.data.query(query)
    #     else:
    #         data = self.data
    #
    #     for name, group in data.groupby(self.tags['group']):
    #         # Get features by group
    #         group_data = np.array([current['datum'] for index, current in group.iterrows()])
    #         mean = np.mean(group_data)
    #         std = np.std(group_data)
    #         for index, current in group.iterrows():
    #             data.iat[index, data.columns.get_loc('datum')] = (current['datum'] - mean) / std
    #
    # @staticmethod
    # def ratios(self):
    #     for name, current in self.data.iterrows():
    #         wavelength = current['wavelength']
    #         data_1 = current['datum'][np.logical_and(540 < wavelength, wavelength < 550)]
    #         data_2 = current['datum'][np.logical_and(570 < wavelength, wavelength < 580)]
    #         data_1 = np.mean(data_1)
    #         data_2 = np.mean(data_2)
    #         self.data.iloc[name, self.data.columns.get_loc('datum')] = data_1 / data_2
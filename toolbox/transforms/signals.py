import warnings

import numpy as np
from pywt import dwt
import scipy.stats as st
from scipy.signal import medfilt
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


class FittingTransform(BaseEstimator, TransformerMixin):

    # Create models from data
    def fit(self, x, y=None):
        x = np.mean(x, axis=0)

        # Distributions to check
        DISTRIBUTIONS = [
            st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
            st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
        ]

        # Best holders
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf

        # Estimate distribution parameters from data
        for distribution in DISTRIBUTIONS:

            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = distribution.fit(x)

                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            except Exception:
                pass

        self.distribution = best_distribution
        return self

    def transform(self, x, y=None, copy=True):
        features = []
        for row in x:
            features.append(list(self.distribution.fit(row)))
        return np.array(features)


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


class RatioTransform(BaseEstimator, TransformerMixin):

    def __init__(self, ratios, wavelength):
        wavelength = list(wavelength)
        index_ratios = []
        for ratio in ratios:
          index_ratios.append((wavelength.index(ratio[0]), wavelength.index(ratio[1])))
        self.index_ratios = index_ratios

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        features = np.zeros((x.shape[0], len(self.index_ratios)))
        for index, index_ratio in enumerate(self.index_ratios):
            features[:, index] = x[:, index_ratio[0]]/x[:, index_ratio[1]]
        features[features == -np.inf] = 0
        features[features == np.inf] = 0
        features[np.isnan(features)] = 0
        return features


class ScaleTransform(BaseEstimator, TransformerMixin):

    def __init__(self, method='default'):
        self.method = method

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        if self.method == 'max':
            return np.apply_along_axis((lambda row: preprocessing.maxabs_scale(row)), 1, x)
        elif self.method == 'minmax':
            return np.apply_along_axis((lambda row: preprocessing.minmax_scale(row)), 1, x)
        elif self.method == 'robust':
            return np.apply_along_axis((lambda row: preprocessing.robust_scale(row)), 1, x)
        else:
            return np.apply_along_axis((lambda row: preprocessing.scale(row)), 1, x)


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
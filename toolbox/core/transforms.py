import pywt
from PIL import Image
import numpy as np
from pywt import dwt
from joblib import Parallel, delayed
from scipy import stats as sstats
from skimage.draw import line_aa
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mahotas
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import Imputer


class SelectAtMostKBest(SelectKBest):

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"


class LDAAtMost(LinearDiscriminantAnalysis):

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        if not (0 <= self.n_components <= min(n_samples, n_features)):
            # set k to "all" (skip feature selection), if less than k features are available
            self.n_components = min(n_samples, n_features)
        return super().fit(X, y)


class PCAAtMost(PCA):

    def fit_transform(self, X, y=None):
        n_samples, n_features = X.shape
        if not (0 <= self.n_components <= min(n_samples, n_features)):
            # set k to "all" (skip feature selection), if less than k features are available
            self.n_components = min(n_samples, n_features)
        return super().fit_transform(X, y)


class OrderedEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, unknown='Unknown'):
        self.unknown = unknown

    def fit(self, y):
        self.map_list = y
        return self

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        elements = y.tolist()
        if not isinstance(elements, list):
            elements = [elements]
        return np.array([self.__inverse_element(element) for element in elements])

    def transform(self, y):
        elements = y.tolist()
        if not isinstance(elements, list):
            elements = [elements]
        return np.array([self.__transform_element(element) for element in elements])

    def __inverse_element(self, element):
        if element == -1:
            return self.unknown
        else:
            return self.map_list[element]

    def __transform_element(self, element):
        try:
            return self.map_list.index(element)
        except:
            return -1


class PredictorTransform(BaseEstimator, TransformerMixin):

    def __init__(self, predictor, probabilities=True):
        self.predictor = predictor
        self.probabilities = probabilities
        if probabilities:
            self.name = 'PredictorTransformProbabilities'
        else:
            self.name = 'PredictorTransform'

    def fit(self, x, y=None):
        """
        This should fit this transformer, but DWT doesn't need to fit to train data

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
        """
        self.predictor.fit(self, x, y=None)
        return self

    def transform(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """
        if self.probabilities:
            return np.array(self.predictor.predict_proba(x))
        else:
            return np.array(self.predictor.predict(x))


class DWTDescriptorTransform(BaseEstimator, TransformerMixin):

    def __init__(self, wavelets='db4', scale=1, mean=False):
        self.mean = mean
        self.scale = scale
        self.wavelets = wavelets

    def fit(self, x, y=None):
        """
        This should fit this transformer, but DWT doesn't need to fit to train data

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
        """
        return self

    def transform(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """
        features = []
        for index, data in enumerate(x):
            image = np.array(Image.open(data).convert('L'))
            coefficients = []
            for scale in range(0, self.scale+1):
                cA, (cH, cV, cD) = pywt.dwt2(image, self.wavelets)
                image = cA
                directions = []
                # Squeeze first scale
                if not scale == 0:
                    directions.extend([cH, cV, cD])
                # Concatenate last image
                if scale == (self.scale-1):
                    directions.append(image)
                # Compute coefficients
                coefficients.extend([DWTDescriptorTransform.get_coefficients(direction) for direction in directions])
            features.append(np.array(coefficients).flatten())
        return np.array(features)

    @staticmethod
    def get_coefficients(x):
        squared = x ** 2
        sum_quared = sum(sum(squared))
        entropy = sstats.entropy(squared.flatten() / sum_quared)
        return [np.sum(squared) / x.size, entropy, np.std(x)]


class FourierDescriptorTransform(BaseEstimator, TransformerMixin):

    def __init__(self, radius_feat=22, directions_feat=16):
        self.radius_feat = radius_feat
        self.directions_feat = directions_feat

    def fit(self, x, y=None):
        """
        This should fit this transformer, but DWT doesn't need to fit to train data

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
        """
        return self

    def transform(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """
        features = []
        for index, data in enumerate(x):
            image = np.array(Image.open(data).convert('F'))/255
            power_spectrum = np.abs(np.fft.rfft(image))
            local_features = []
            for radius in range(0, self.radius_feat):
                mask = FourierDescriptorTransform.circular_mask(power_spectrum.shape,
                                                                center=[0, power_spectrum.shape[1]/2],
                                                                np_split=self.radius_feat,
                                                                index=radius)
                local_features.append(np.sum(power_spectrum[mask])/np.sum(mask))

            for direction in range(0, self.directions_feat):
                mask = FourierDescriptorTransform.direction_mask(power_spectrum.shape,
                                                                 center=[0, power_spectrum.shape[1]/2],
                                                                 np_split=self.directions_feat,
                                                                 index=direction)
                local_features.append(np.sum(power_spectrum[mask])/np.sum(mask))
            features.append(np.array(local_features).flatten())
        return features

    @staticmethod
    def circular_mask(shape, center, np_split, index):
        Y, X = np.ogrid[:shape[0], :shape[1]]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        radius = np.amax(dist_from_center) / np_split
        # Compute mask
        mask_inner = (radius * index) <= dist_from_center
        mask_outer = dist_from_center <= radius * (index + 1)
        return np.logical_and(mask_inner, mask_outer)

    @staticmethod
    def direction_mask(shape, center, np_split, index):
        tolerance = 0.5
        radius = 180. / np_split
        Y, X = np.ogrid[:shape[0], :shape[1]]
        angle_from_center = np.rad2deg(np.arctan2((X - center[0]), (Y - center[1])))
        # Compute mask
        mask_inner = (radius * index) - tolerance<= angle_from_center
        mask_outer = angle_from_center <= (radius * index) + tolerance
        return np.logical_and(mask_inner, mask_outer)


class DWTFitDescriptorTransform(BaseEstimator, TransformerMixin):

    def __init__(self, wavelets='db4', scale=1, mean=False):
        self.mean = mean
        self.scale = scale
        self.wavelets = wavelets

    def fit(self, x, y=None):
        """
        This should fit this transformer, but DWT doesn't need to fit to train data

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
        """
        return self

    def transform(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """
        features = []
        for index, data in enumerate(x):
            image = np.array(Image.open(data).convert('L'))
            coefficients = []
            for scale in range(0, self.scale):
                cA, (cH, cV, cD) = pywt.dwt2(image, self.wavelets)
                image = cA
                directions = [cH, cV, cD]
                coefficients.extend(
                    Parallel(n_jobs=3)(delayed(self.get_coefficients)(direction) for direction in directions))

            features.append(np.array(coefficients).flatten())
        return np.array(features)

    def get_coefficients(self, x):
        params = gennorm.fit(x)
        beta = params[0]  # Shape
        alpha = params[2]  # Scale
        return [alpha, beta]


class ArgMaxTransform(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        return x.argmax(axis=1)


class FlattenTransform(BaseEstimator, TransformerMixin):

    def __init__(self, axis=False):
        self.axis = axis

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        return x.reshape((x.shape[0], -1))


class HaralickTransform(BaseEstimator, TransformerMixin):

    def __init__(self, mean=False):
        self.mean = mean

    def fit(self, x, y=None):
        """
        This should fit this transformer, but DWT doesn't need to fit to train data

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
        """
        return self

    def transform(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """

        haralick = []
        for index, data in enumerate(x):
            image = np.array(Image.open(data).convert('L'))
            features = mahotas.features.haralick(image)
            if self.mean:
                haralick.append(features.mean(axis=0))
            else:
                haralick.append(features.flatten())
        return np.array(haralick)


class DWTTransform(BaseEstimator, TransformerMixin):
    """Class that manage a DWT Transform

     This class is made the same way as sklearn transform to be fit in a Pipe

     Attributes:
         mode (:obj:'str'): A mode for DWT extraction.

     """

    def __init__(self, mode='db1', segment_length=-1):
        """Make an initialisation of DWTTransform object.

        Take a string that represent extraction mode, default use 'db1'

        Args:
             mode (:obj:'str'): The mode as string.
        """
        self.mode = mode
        self.segment_length = segment_length

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
        (cA, _) = dwt(x, self.mode)
        return cA
        # if x.ndim == 2:
        #     data_length = x.shape[1]
        # else:
        #     data_length = len(x)
        #
        # if self.segment_length == -1:
        #     length = data_length
        # else:
        #     length = self.segment_length
        #
        # chunks = [x[:, i:i+length] for i in range(0, data_length, length)]
        #
        # res = None
        # for i in range(len(chunks)):
        #     (cA, _) = wavelet(chunks[i], self.mode)
        #     if res is None:
        #         res = cA
        #     else:
        #         res = concatenate((res, cA), axis=1)
        # return res


class PLSTransform(PLSRegression):

    def transform(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """
        return super(PLSRegression, self).transform(x)


class PNormTransform(BaseEstimator, TransformerMixin):
    """Class that p-norm normalization

     This class is made for sklearn and build upon scipy.

     Attributes:
         p (:obj:'int'): An integer that give the normalization coefficient.

     """

    def __init__(self, p=1, axis=1):
        self.p = p
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X):
        if X.dtype == object:
            normalized = []
            for x in X:
                normalized.append(np.linalg.norm(x, ord=self.p, axis=self.axis - 1))
            return np.array(normalized)
        return np.linalg.norm(X, ord=self.p, axis=self.axis)


class CorrelationArrayTransform(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        labels = np.unique(y)
        self.means = np.zeros((len(labels), X.shape[1]))
        for label in labels:
            self.means[label, :] = np.mean(X[y == label, :], axis=0)
        return self

    def transform(self, X):
        transform = np.zeros((len(X), len(self.means)))
        for index, mean in enumerate(self.means):
            transform[:, index] = ((X - mean) ** 2).mean(axis=1)
        return transform


class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh

        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = Imputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped = True
        return X


class LDATransform(LinearDiscriminantAnalysis):

    def transform(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """
        return super(LinearDiscriminantAnalysis, self).transform(x)

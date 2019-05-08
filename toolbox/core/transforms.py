import pywt
from PIL import Image
import numpy as np
from pywt import dwt
from scipy.stats import gennorm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mahotas
from sklearn.preprocessing import Imputer


class OrderedEncoder(BaseEstimator, TransformerMixin):

    def fit(self, y):
        self.map_list = y
        return self

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def transform(self, y):
        elements = y.tolist()
        if not isinstance(elements, list):
            elements = [elements]
        return np.array([self.map_list.index(element) for element in elements])

    def inverse_transform(self, y):
        elements = y.tolist()
        if not isinstance(elements, list):
            elements = [elements]
        return np.array([self.map_list[element] for element in elements])


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

    def __init__(self, wavelets=None, mode='db', scale=1, mean=False):
        self.mean = mean
        self.scale = scale
        if wavelets is None:
            self.wavelets = pywt.wavelist(mode)[:5]
        else:
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
                for wavelet in self.wavelets:
                    cA, (cH, cV, cD) = pywt.dwt2(image, wavelet)
                    image = cA
                    directions = [cH, cV, cD]
                    for direction in directions:
                        coefficients.append(self.get_coefficients(direction.flatten()))

            features.append(np.array(coefficients).flatten())
        return np.array(features)

    def get_coefficients(self, x):
        params = gennorm.fit(x)
        beta = params[0] # Shape
        alpha = params[2] # Scale
        return [alpha, beta]


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
                normalized.append(np.linalg.norm(x, ord=self.p, axis=self.axis-1))
            return np.array(normalized)
        return np.linalg.norm(X, ord=self.p, axis=self.axis)


class CorrelationArrayTransform(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        labels = np.unique(y)
        self.means = np.zeros((len(labels),X.shape[1]))
        for label in labels:
            self.means[label, :] = np.mean(X[y==label, :], axis=0)
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

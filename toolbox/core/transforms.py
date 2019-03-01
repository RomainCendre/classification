import pywt
from PIL import Image
from numpy import array
from pywt import dwt
from scipy.stats import gennorm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mahotas


class PredictorTransform(BaseEstimator, TransformerMixin):

    def __init__(self, predictor, probabilities=True):
        self.predictor = predictor
        self.probabilities = probabilities

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
            return array(self.predictor.predict_proba(x))
        else:
            return array(self.predictor.predict(x))


class ImageDWTTransform(BaseEstimator, TransformerMixin):

    def __init__(self, mode='db1', mean=False):
        self.mode = mode
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
        features = []
        for index, data in enumerate(x):
            image = array(Image.open(data).convert('L'))
            cA, (cH, cV, cD) = pywt.dwt2(image, self.mode)
            directions = [cH, cV, cD]
            coefficients = []
            for direction in directions:
                coefficients.append(self.get_coefficients(direction.flatten()))

            features.append(array(coefficients).flatten())
        return array(features)

    def get_coefficients(self, x):
        return gennorm.fit(x)


class ImageHaralickTransform(BaseEstimator, TransformerMixin):

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
            image = array(Image.open(data).convert('L'))
            features = mahotas.features.haralick(image)
            if self.mean:
                haralick.append(features.mean(axis=0))
            else:
                haralick.append(features.flatten())
        return array(haralick)


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

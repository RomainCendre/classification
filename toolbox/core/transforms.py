import hashlib
from os.path import normpath, join

from PIL import Image
from numpy import array
from pywt import dwt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import mahotas


class PatchMakerTransform(BaseEstimator, TransformerMixin):

    def __init__(self, folder, patch_size=250):
        self.folder = folder
        self.patch_size = patch_size

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

        patches = []
        for index, data in enumerate(x):
            patches.append(self.__get_patches(data))
        return array(patches)

    def __get_patches(self, filename):
        hash_name = hashlib.md5(filename.encode('utf-8')).hexdigest()
        X = array(Image.open(filename).convert('L'))
        patches = []
        index = 0
        for r in range(0, X.shape[0] - self.patch_size + 1, self.patch_size):
            for c in range(0, X.shape[1] - self.patch_size + 1, self.patch_size):
                patch = X[r:r + self.patch_size, c:c + self.patch_size]
                filename = '{hash_name}_{id}.png'.format(hash_name=join(self.folder, hash_name), id=index)
                filename = normpath(filename)
                Image.fromarray(patch).save(filename)
                patches.append(filename)
                index += 1
        return patches


class HaralickDescriptorTransform(BaseEstimator, TransformerMixin):

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
            if not isinstance(data, str):
                features = []
                for sub_data in data:
                    image = array(Image.open(sub_data).convert('L'))
                    sub_features = mahotas.features.haralick(image)
                    if self.mean:
                        sub_features = sub_features.mean(axis=0)
                    else:
                        sub_features = sub_features.flatten()
                    features.append(sub_features)
                haralick.append(features)
            else:
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
        #     (cA, _) = dwt(chunks[i], self.mode)
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
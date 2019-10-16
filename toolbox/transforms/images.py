import mahotas
import numpy as np
import pywt
from joblib import Parallel, delayed
from PIL import Image
from scipy import stats as sstats
from sklearn.base import BaseEstimator, TransformerMixin


class DistributionImageTransform(BaseEstimator, TransformerMixin):

    def __init__(self, distribution=sstats.gengamma, coefficients=['beta', 'loc', 'scale']):
        self.distribution = distribution
        self.coefficients = coefficients

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
            # Compute coefficients
            hist, bin_edges = np.histogram(image.flatten())
            features.append(np.array(self.__get_coefficients(hist)).flatten())
        return np.array(features)

    def __get_coefficients(self, x):
        beta, loc, scale = self.distribution.fit(x)
        # Concatenate coefficients
        coefficients = []
        if 'beta' in self.coefficients:
            coefficients.append(beta)
        if 'loc' in self.coefficients:
            coefficients.append(loc)
        if 'scale' in self.coefficients:
            coefficients.append(scale)
        return coefficients


class DWTImageTransform(BaseEstimator, TransformerMixin):

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
            for scale in range(0, self.scale + 1):
                cA, (cH, cV, cD) = pywt.dwt2(image, self.wavelets)
                image = cA
                directions = []
                # Squeeze first scale
                if not scale == 0:
                    directions.extend([cH, cV, cD])
                # Concatenate last image
                if scale == (self.scale - 1):
                    directions.append(image)
                # Compute coefficients
                coefficients.extend([DWTImageTransform.get_coefficients(direction) for direction in directions])
            features.append(np.array(coefficients).flatten())
        return np.array(features)

    @staticmethod
    def get_coefficients(x):
        squared = x ** 2
        sum_quared = sum(sum(squared))
        entropy = sstats.entropy(squared.flatten() / sum_quared)
        return [np.sum(squared) / x.size, entropy, np.std(x)]


class DWTGGDImageTransform(BaseEstimator, TransformerMixin):

    def __init__(self, wavelets='db4', scale=1, parameter='both'):
        self.parameter = parameter
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
        hist, bin_edges = np.histogram(x.flatten(), bins=10)
        shape, loc, scale = sstats.gennorm.fit(hist)
        if self.parameter == 'both':
            return [scale, shape]
        elif self.parameter == 'beta' or self.parameter == 'shape':
            return [shape]
        else:
            return [scale]


class HaralickImageTransform(BaseEstimator, TransformerMixin):

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


class FourierImageTransform(BaseEstimator, TransformerMixin):

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
            image = np.array(Image.open(data).convert('F')) / 255
            power_spectrum = np.abs(np.fft.rfft(image))
            local_features = []
            for radius in range(0, self.radius_feat):
                mask = FourierImageTransform.circular_mask(power_spectrum.shape,
                                                           center=[0, power_spectrum.shape[1] / 2],
                                                           np_split=self.radius_feat,
                                                           index=radius)
                local_features.append(np.sum(power_spectrum[mask]) / np.sum(mask))

            for direction in range(0, self.directions_feat):
                mask = FourierImageTransform.direction_mask(power_spectrum.shape,
                                                            center=[0, power_spectrum.shape[1] / 2],
                                                            np_split=self.directions_feat,
                                                            index=direction)
                local_features.append(np.sum(power_spectrum[mask]) / np.sum(mask))
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
        mask_inner = (radius * index) - tolerance <= angle_from_center
        mask_outer = angle_from_center <= (radius * index) + tolerance
        return np.logical_and(mask_inner, mask_outer)


class SpatialImageTransform(BaseEstimator, TransformerMixin):

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
            current_features = []
            current_features.extend(mahotas.features.haralick(image).mean(axis=0))
            current_features.extend(SpatialImageTransform.histogram_features(image))
            features.append(current_features)
        return np.array(features)

    @staticmethod
    def histogram_features(images):
        features = []
        features.append(np.mean(images))
        features.append(np.std(images))
        features.append(sstats.skew(images.flatten()))
        features.append(sstats.kurtosis(images.flatten()))
        features.append(sstats.entropy(images.flatten()))
        return features

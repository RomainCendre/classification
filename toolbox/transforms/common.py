import numpy as np
from pywt import dwt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression


#####################################
# Make transforms
class ArgMaxTransform(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        return x.argmax(axis=1)


class FlattenTransform(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        return x.reshape((x.shape[0], -1))


class ReshapeTrickTransform(BaseEstimator, TransformerMixin):

    def __init__(self, shape):
        self.shape = shape

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None, copy=True):
        return x.reshape(self.shape)


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


# class ReduceVIF(BaseEstimator, TransformerMixin):
#     def __init__(self, thresh=5.0, impute=True, impute_strategy='median'):
#         # From looking at documentation, values between 5 and 10 are "okay".
#         # Above 10 is too high and so should be removed.
#         self.thresh = thresh
#
#         # The statsmodel function will fail with NaN values, as such we have to impute them.
#         # By default we impute using the median value.
#         # This imputation could be taken out and added as part of an sklearn Pipeline.
#         if impute:
#             self.imputer = Imputer(strategy=impute_strategy)
#
#     def fit(self, X, y=None):
#         print('ReduceVIF fit')
#         if hasattr(self, 'imputer'):
#             self.imputer.fit(X)
#         return self
#
#     def transform(self, X, y=None):
#         print('ReduceVIF transform')
#         columns = X.columns.tolist()
#         if hasattr(self, 'imputer'):
#             X = pd.DataFrame(self.imputer.transform(X), columns=columns)
#         return ReduceVIF.calculate_vif(X, self.thresh)
#
#     @staticmethod
#     def calculate_vif(X, thresh=5.0):
#         # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
#         dropped = True
#         while dropped:
#             variables = X.columns
#             dropped = False
#             vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
#
#             max_vif = max(vif)
#             if max_vif > thresh:
#                 maxloc = vif.index(max_vif)
#                 print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
#                 X = X.drop([X.columns.tolist()[maxloc]], axis=1)
#                 dropped = True
#         return X

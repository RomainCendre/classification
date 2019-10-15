from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest


#####################################
# Redefinition in at most way
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


class SelectAtMostKBest(SelectKBest):

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"

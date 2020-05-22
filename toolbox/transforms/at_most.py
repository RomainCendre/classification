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


# Redefinition in at most way
class LDAAtRatio(LinearDiscriminantAnalysis):

    def fit(self, X, y=None):
        ratio = self.n_components
        self.n_components = None
        super().fit(X, y)
        self._max_components = self.__select_n_components(ratio)
        return self
        # lda_fitted = super().fit(X, y)
        # test = lda_fitted.__select_n_components(ratio)

    def __select_n_components(self, goal_var: float) -> int:
        var_ratio = self.explained_variance_ratio_

        # Set initial variance explained so far
        total_variance = 0.0

        # Set initial number of features
        n_components = 0

        # For the explained variance of each feature:
        for explained_variance in var_ratio:

            # Add the explained variance to the total
            total_variance += explained_variance

            # Add one to the number of components
            n_components += 1

            # If we reach our goal level of explained variance
            if total_variance >= goal_var:
                # End the loop
                break

        # Return the number of components
        return n_components


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

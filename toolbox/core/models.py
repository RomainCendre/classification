import types
import numpy as np
from copy import deepcopy
from keras import Sequential
from keras.utils.generic_utils import has_arg
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels

from toolbox.core.generators import ResourcesGenerator


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


class KerasBatchClassifier(KerasClassifier):

    def check_params(self, params):
        """Checks for user typos in `params`.

        # Arguments
            params: dictionary; the parameters to be checked

        # Raises
            ValueError: if any member of `params` is not a valid argument.
        """
        local_param = deepcopy(params)
        legal_params_fns = [ResourcesGenerator.__init__, ResourcesGenerator.flow_from_paths,
                            Sequential.fit_generator, Sequential.predict_generator,
                            Sequential.evaluate_generator]
        found_params = set()
        for param_name in params:
            for fn in legal_params_fns:
                if has_arg(fn, param_name):
                    found_params.add(param_name)
        if not len(found_params) == 0:
            [local_param.pop(key) for key in list(found_params)]
        super().check_params(local_param)

    def init_model(self, y=[]):
        # If already init
        if hasattr(self, 'model'):
            return

        self.sk_params.update({'output_classes': len(unique_labels(y))})
        # Get the deep model
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        # Store labels
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))

    def fit(self, X, y, callbacks=[], X_validation=None, y_validation=None, **kwargs):
        self.init_model(y)

        # Get arguments for predict
        params_fit = deepcopy(self.sk_params)
        params_fit.update(kwargs)
        params_fit = self.__filter_params(params_fit, Sequential.fit_generator)

        # Get generator
        train = self.__create_generator(X=X, y=y, params=kwargs)
        validation = None
        if X_validation is not None:
            validation = self.__create_generator(X=X_validation, y=y_validation, params=kwargs)

        if not self.model._is_compiled:
            tr_x, tr_y = train[0]
            self.model.fit(tr_x, tr_y)

        self.history = self.model.fit_generator(generator=train, validation_data=validation, callbacks=callbacks,
                                                **params_fit)

        return self.history

    def predict(self, X, **kwargs):
        probs = self.predict_proba(X, **kwargs)
        if probs.shape[-1] > 1:
            classes = probs.argmax(axis=-1)
        else:
            classes = (probs > 0.5).astype('int32')
        return self.classes_[classes]

    def predict_proba(self, X, **kwargs):
        self.init_model()

        # Define some local arguments
        copy_kwargs = deepcopy(kwargs)
        copy_kwargs.update({'shuffle': False})
        copy_kwargs.update({'batch_size': 1})

        # Create generator
        valid = self.__create_generator(X=X, params=copy_kwargs)

        # Get arguments for predict
        params_pred = deepcopy(self.sk_params)
        params_pred.update(copy_kwargs)
        params_pred = self.__filter_params(params_pred, Sequential.predict_generator)

        # Predict!
        probs = self.model.predict_generator(generator=valid, **params_pred)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = hstack([1 - probs, probs])
        return probs

    def score(self, X, y, **kwargs):
        kwargs = self.filter_sk_params(Sequential.evaluate_generator, kwargs)

        # Create generator
        generator = ResourcesGenerator(preprocessing_function=kwargs.get('Preprocess', None))
        valid = generator.flow_from_paths(X, y, batch_size=1, shuffle=False)

        # Get arguments for fit
        fit_args = deepcopy(self.filter_sk_params(Sequential.evaluate_generator))
        fit_args.update(kwargs)

        # sparse to numpy array
        outputs = self.model.evaluate_generator(generator=valid, **fit_args)
        outputs = np.to_list(outputs)
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')

    def __create_generator(self, X, y=None, params={}):
        # Init generator
        params_init = deepcopy(self.sk_params)
        params_init.update(params)
        params_init = self.__filter_params(params_init, ResourcesGenerator.__init__)
        generator = ResourcesGenerator(**params_init)

        # Create iterator
        params_flow = deepcopy(self.sk_params)
        params_flow.update(params)
        params_flow = self.__filter_params(params_flow, ResourcesGenerator.flow_from_paths)
        if y is not None:
            return generator.flow_from_paths(X, y, **params_flow)
        else:
            return generator.flow_from_paths(X, **params_flow)

    def __filter_params(self, params, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `sk_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res


class ClassifierPatch(BaseEstimator, ClassifierMixin):

    def __init__(self, extractor, model, patch_size):
        self.extractor = extractor
        self.model = model
        self.patch_size = patch_size

    def fit(self, X, y, **kwargs):
        features = self.__extract_features_patch(X, **kwargs)
        self.model.fit(features, y)
        return self

    def predict(self, X, **kwargs):
        features = self.__extract_features_patch(X, **kwargs)
        return self.model.predict(features)

    def predict_proba(self, X, **kwargs):
        features = self.__extract_features_patch(X, **kwargs)
        return self.model.predict_proba(features)

    def __extract_features_patch(self, X, **kwargs):
        # Browse images
        predictions = []
        probabilities = []
        for patches in X:
            sub_predictions = None
            sub_probabilities = []
            for patch in patches:
                current = self.extractor.predict_proba(patch.reshape(1, -1))
                if sub_predictions is None:
                    sub_predictions = np.zeros(current.shape)
                sub_predictions[:, current.argmax(axis=1)] += 1
                # sub_probabilities.append(current)
            predictions.append(sub_predictions/len(patches))

        return np.concatenate(np.array(predictions), axis=0)

    @staticmethod
    def __filter_params(params, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `sk_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res


class PatchClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, hierarchies, metric=None):
        if metric:
            self.metric = metric
        else:
            self.metric = accuracy_score
        self.hierarchies = hierarchies
        self.thresholds = None

    def fit(self, x, y=None):
        """
        This should fit this transformer, but DWT doesn't need to fit to train data

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
        """
        x_probas = np.zeros((len(x), len(self.hierarchies)))
        patches_number = x.shape[1]
        for hierarchy in self.hierarchies:
            x_probas[:, hierarchy] = np.sum(x == hierarchy, axis=1) / patches_number

        global_score = 0
        self.thresholds = np.zeros(len(self.hierarchies))
        for hierarchy in self.hierarchies:
            potential_thresholds = np.sort(np.unique(x_probas[:, hierarchy]))
            for thresh in potential_thresholds:
                thresholds = np.copy(self.thresholds)
                thresholds[hierarchy] = thresh
                score = self.metric(self.get_predictions(x_probas, thresholds), y)
                if global_score < score:
                    global_score = score
                    self.thresholds[hierarchy] = thresh
        print(self.thresholds)
        return self

    def predict(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """
        return self.get_predictions(x, self.thresholds)

    def get_predictions(self, x, thresholds):
        return np.argmax((x>thresholds)*self.hierarchies, axis=1)

    def predict_proba(self, x, y=None, copy=True):
        return x

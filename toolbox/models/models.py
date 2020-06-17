import copy
import inspect
import io
import sys
import types

import h5py
import numpy as np
from copy import deepcopy
from joblib import delayed, Parallel
from numpy import hstack
from tensorflow.keras import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.metrics import accuracy_score
from sklearn.multiclass import _fit_binary
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import unique_labels, _ovr_decision_function
import tensorflow as tf
from toolbox.models.generators import ResourcesGenerator


# class CustomMIL(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
#
#     def __init__(self, estimator, instance_prediction=False):
#         self.is_inconsistent = True
#         self.estimator = estimator
#         self.instance_prediction = instance_prediction
#
#
# class OvOMIL(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):  # Based on OneVsOne
#
#     def __init__(self, estimator, instance_prediction=False, n_jobs=None):
#         self.is_inconsistent = True
#         self.estimator = estimator
#         self.n_jobs = n_jobs
#         self.instance_prediction = instance_prediction
#
#     def fit(self, bags, y):
#         self.classes_ = np.unique(y)
#         if len(self.classes_) == 1:
#             raise ValueError("OneVsOneClassifier can not be fit when only one"
#                              " class is present.")
#         n_classes = self.classes_.shape[0]
#         estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
#             delayed(OvOMIL.__est_fit_ovo_binary)
#             (self.estimator, bags, y, self.classes_[i], self.classes_[j])
#             for i in range(n_classes) for j in range(i + 1, n_classes)))))
#
#         self.estimators_ = estimators_indices[0]
#         try:
#             self.pairwise_indices_ = (
#                 estimators_indices[1] if self._pairwise else None)
#         except AttributeError:
#             self.pairwise_indices_ = None
#
#         return self
#
#     def predict(self, X):
#         Y = self.decision_function(X)
#         # if self.n_classes_ == 2:
#         #     return self.classes_[(Y > 0).astype(np.int)]
#         return self.classes_[Y.argmax(axis=1)]
#
#     def predict_proba(self, X):
#         Y = self.decision_function(X)
#         Y = (Y - np.min(Y))
#         Y = Y / np.max(Y)
#         return Y
#
#     def decision_function(self, X):
#         indices = self.pairwise_indices_
#         if indices is None:
#             Xs = [X] * len(self.estimators_)
#         else:
#             Xs = [X[:, idx] for idx in indices]
#
#         predictions = np.vstack([self.__est_predict(est, Xi) for est, Xi in zip(self.estimators_, Xs)]).T
#         confidences = np.vstack([self.__est_predict_binary(est, Xi) for est, Xi in zip(self.estimators_, Xs)]).T
#         Y = OvOMIL._ovr_decision_function(predictions, confidences, len(self.classes_))
#         return Y
#
#     @staticmethod
#     def _ovr_decision_function(predictions, confidences, n_classes):
#         n_samples = predictions.shape[0]
#         votes = np.zeros((n_samples, n_classes))
#         sum_of_confidences = np.zeros((n_samples, n_classes))
#
#         k = 0
#         for i in range(n_classes):
#             for j in range(i + 1, n_classes):
#                 sum_of_confidences[:, i] -= confidences[:, k]
#                 sum_of_confidences[:, j] += confidences[:, k]
#                 votes[predictions[:, k] == 0, i] += 1
#                 votes[predictions[:, k] == 1, j] += 1
#                 k += 1
#
#         max_confidences = sum_of_confidences.max()
#         min_confidences = sum_of_confidences.min()
#
#         if max_confidences == min_confidences:
#             return votes
#
#         # Scale the sum_of_confidences to (-0.5, 0.5) and add it with votes.
#         # The motivation is to use confidence levels as a way to break ties in
#         # the votes without switching any decision made based on a difference
#         # of 1 vote.
#         eps = np.finfo(sum_of_confidences.dtype).eps
#         max_abs_confidence = max(abs(max_confidences), abs(min_confidences))
#         scale = (0.5 - eps) / max_abs_confidence
#         return votes + sum_of_confidences * scale
#
#     @property
#     def n_classes_(self):
#         return len(self.classes_)
#
#     @property
#     def _pairwise(self):
#         """Indicate if wrapped estimator is using a precomputed Gram matrix"""
#         return getattr(self.estimator, "_pairwise", False)
#
#     def __est_predict(self, estimator, bags):
#         max_prediction = []
#         predictions_list = self.__est_predict_proba(estimator, bags)
#         for predictions in predictions_list:
#             if len(predictions.shape) == 1:
#                 max_prediction.append(np.argmax(predictions))
#             else:
#                 max_prediction.append(np.argmax(predictions, axis=1))
#         return max_prediction
#
#     def __est_predict_binary(self, estimator, bags):
#         bin_prediction = []
#         predictions_list = self.__est_predict_proba(estimator, bags)
#         for predictions in predictions_list:
#             if len(predictions.shape) == 1:
#                 bin_prediction.append(predictions[1])
#             else:
#                 bin_prediction.append(predictions[:, 1])
#         return bin_prediction
#
#     def __est_predict_proba(self, estimator, bags):
#         if self.instance_prediction and 'instancePrediction' in list(inspect.signature(estimator.predict).parameters):
#             _, predictions = estimator.predict(bags, self.instance_prediction)
#             max_value = np.max(np.abs(predictions))
#             predictions = np.nan_to_num(predictions/max_value)*0.5+0.5
#             predictions = np.array([1 - predictions, predictions]).T
#             predictions_list = []
#             count = 0
#             for bag in bags:
#                 predictions_list.append(predictions[count:count + len(bag)])
#                 count += len(bag)
#         else:
#             predictions = estimator.predict(bags)
#             max_value = np.max(np.abs(predictions))
#             predictions = np.nan_to_num(predictions/max_value)*0.5+0.5
#             predictions = np.array([1 - predictions, predictions]).T
#             predictions_list = list(predictions)
#         return predictions_list
#
#     @staticmethod
#     def __est_fit_ovo_binary(estimator, bags, y, i, j):
#         #
#         # return _fit_ovo_binary(estimator, bags, y, classes_1, classes_2)
#
#         cond = np.logical_or(y == i, y == j)
#         y = y[cond]
#         y_binary = np.empty(y.shape, np.int)
#         y_binary[y == i] = 0
#         y_binary[y == j] = 1
#         indcond = np.arange(len(bags))[cond]
#
#         # bags, y = CustomMIL.__prepare_fit(bags, y_binary)
#         return _fit_binary(estimator,
#                            *OvOMIL.__prepare_fit(_safe_split(estimator, bags, None, indices=indcond)[0], y_binary),
#                            classes=[i, j]), indcond
#
#     @staticmethod
#     def __prepare_fit(bags, y):
#         y = 2 * y - 1
#         # unique_labels = np.unique(y)
#         # LIMIT = 20000
#         # BAG_LIMIT = int(LIMIT/bags.shape[1])
#         # LAB_BAG_LIMIT = int(BAG_LIMIT/len(unique_labels))
#         # new_bags = []
#         # new_y = []
#         # for label in unique_labels:
#         #     current_bags = bags[label == y, :, :]
#         #     current_y = y[label == y]
#         #     new_bags.append(current_bags[:LAB_BAG_LIMIT, :, :])
#         #     new_y.append(current_y[:LAB_BAG_LIMIT])
#         # return np.concatenate(new_bags), np.concatenate(new_y)
#         return bags, y
class CustomMIL(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):  # Based on OneVsOne

    def __init__(self, estimator, n_jobs=None):
        self.is_inconsistent = True
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, bags, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 1:
            raise ValueError("OneVsOneClassifier can not be fit when only one"
                             " class is present.")
        n_classes = self.classes_.shape[0]
        estimators_indices = list(zip(*(Parallel(n_jobs=self.n_jobs)(
            delayed(CustomMIL.__est_fit_ovo_binary)
            (self.estimator, bags, y, self.classes_[i], self.classes_[j])
            for i in range(n_classes) for j in range(i + 1, n_classes)))))

        self.estimators_ = estimators_indices[0]
        try:
            self.pairwise_indices_ = (
                estimators_indices[1] if self._pairwise else None)
        except AttributeError:
            self.pairwise_indices_ = None

        return self

    def predict(self, X):
        Y = self.decision_function(X)
        # if self.n_classes_ == 2:
        #     return self.classes_[(Y > 0).astype(np.int)]
        return self.classes_[Y.argmax(axis=1)]

    def predict_proba(self, X):
        Y = self.decision_function(X)
        Y = (Y - np.min(Y))
        Y = Y / np.max(Y)
        return Y

    def decision_function(self, X):
        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            Xs = [X[:, idx] for idx in indices]

        predictions = np.vstack([self.__est_predict(est, Xi) for est, Xi in zip(self.estimators_, Xs)]).T
        confidences = np.vstack([self.__est_predict_binary(est, Xi) for est, Xi in zip(self.estimators_, Xs)]).T
        Y = _ovr_decision_function(predictions, confidences, len(self.classes_))
        # if self.n_classes_ == 2:
        #     return Y[:, 1]
        return Y

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def _pairwise(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return getattr(self.estimator, "_pairwise", False)

    def __est_predict(self, estimator, bags, instancePrediction=None):
        return np.argmax(self.__est_predict_proba(estimator, bags, instancePrediction), axis=1)

    def __est_predict_binary(self, estimator, bags, instancePrediction=None):
        return self.__est_predict_proba(estimator, bags, instancePrediction)[:, 1]

    @staticmethod
    def __est_predict_proba(estimator, bags, instancePrediction=None):
        predictions = estimator.predict(bags)
        max_value = np.max(np.abs(predictions))
        predictions = np.nan_to_num(predictions/max_value)*0.5+0.5
        return np.array([1-predictions, predictions]).T

    @staticmethod
    def __est_fit_ovo_binary(estimator, bags, y, i, j):
        #
        # return _fit_ovo_binary(estimator, bags, y, classes_1, classes_2)

        cond = np.logical_or(y == i, y == j)
        y = y[cond]
        y_binary = np.empty(y.shape, np.int)
        y_binary[y == i] = 0
        y_binary[y == j] = 1
        indcond = np.arange(len(bags))[cond]

        # bags, y = CustomMIL.__prepare_fit(bags, y_binary)
        return _fit_binary(estimator,
                           *CustomMIL.__prepare_fit(_safe_split(estimator, bags, None, indices=indcond)[0], y_binary),
                           classes=[i, j]), indcond

    @staticmethod
    def __prepare_fit(bags, y):
        y = 2 * y - 1
        # unique_labels = np.unique(y)
        # LIMIT = 20000
        # BAG_LIMIT = int(LIMIT/bags.shape[1])
        # LAB_BAG_LIMIT = int(BAG_LIMIT/len(unique_labels))
        # new_bags = []
        # new_y = []
        # for label in unique_labels:
        #     current_bags = bags[label == y, :, :]
        #     current_y = y[label == y]
        #     new_bags.append(current_bags[:LAB_BAG_LIMIT, :, :])
        #     new_y.append(current_y[:LAB_BAG_LIMIT])
        # return np.concatenate(new_bags), np.concatenate(new_y)
        return bags, y


class DecisionVotingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, mode='max', metric=None, is_prior_max=True):
        # Check mandatory mode
        mandatory = ['at_least_one', 'dynamic_thresh', 'max']
        if mode not in mandatory:
            raise Exception(f'Expected modes: {mandatory}, but found: {mode}.')

        self.is_inconsistent = True
        self.mode = mode
        # Set metric mode used for evaluation
        if metric:
            self.metric = metric
        else:
            self.metric = accuracy_score

        self.is_prior_max = is_prior_max

        # Init to default values other properties
        self.number_labels = 0
        self.thresholds = None

    def fit(self, x, y=None):
        self.number_labels = max(y) + 1
        if self.mode == 'dynamic_thresh':
            self.__fit_dynamic_thresh(x, y)
            print(self.thresholds)
        return self

    def predict(self, x, y=None, copy=True):
        probas = self.__get_probas(x)  # Go through probas of elements as it's a contant array
        if self.mode == 'max':
            return self.__get_predictions_max(probas)
        elif self.mode == 'at_least_one':
            return self.__get_predictions_at_least_one(probas)
        elif self.mode == 'dynamic_thresh':
            return self.__get_predictions_dynamic(probas, self.thresholds)

    def predict_proba(self, x, y=None, copy=True):
        return self.__get_probas(x)

    def __fit_dynamic_thresh(self, x, y):
        probabilities = self.__get_probas(x)
        self.thresholds = np.zeros(self.number_labels)

        if self.is_prior_max:
            iterator = reversed(range(self.number_labels))
        else:
            iterator = range(self.number_labels)

        for hierarchy in iterator:
            score = 0
            potential_thresholds = np.sort(np.unique(np.concatenate(([1, 0], probabilities[:, hierarchy]))))
            label = y == hierarchy
            probability = probabilities[:, hierarchy]
            for thresh in potential_thresholds:
                threshed = probability >= thresh
                tmp_score = self.metric(label, threshed)
                if tmp_score > score:
                    score = tmp_score
                    self.thresholds[hierarchy] = thresh

    def __get_probas(self, x):
        x_probas = np.zeros((len(x), self.number_labels))
        # patches_number = x.shape[1]
        for index, group in enumerate(x):
            nb_elements = group.size
            for label in range(self.number_labels):
                x_probas[index, label] = np.sum(group == label, axis=0) / nb_elements
        return x_probas

    def __get_predictions_at_least_one(self, x):
        return np.argmax((x > 0) * self.__get_prior_coefficients(), axis=1)

    def __get_predictions_max(self, x):
        maximum = x.max(axis=1, keepdims=1) == x
        return (maximum * self.__get_prior_coefficients()).argmax(axis=1)

    def __get_prior_coefficients(self):
        coefficients = range(self.number_labels)
        if not self.is_prior_max:
            coefficients = reversed(coefficients)

        return np.array(list(coefficients))

    def __get_predictions_dynamic(self, x, thresholds):
        return np.argmax((x >= thresholds) * self.__get_prior_coefficients(), axis=1)


class ScoreVotingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, low='max', high='max', metric=None, is_prior_max=True, p_norm=None):
        # Check mandatory mode
        mandatory = ['dynamic', 'max']
        if high not in mandatory:
            raise Exception(f'Expected modes: {mandatory}, but found: {high}.')
        mandatory = ['mean', 'max', 'p-norm']
        if low not in mandatory:
            raise Exception(f'Expected modes: {mandatory}, but found: {low}.')

        self.low = low
        self.high = high
        # Set metric mode used for evaluation
        if metric:
            self.metric = metric
        else:
            self.metric = accuracy_score

        self.is_prior_max = is_prior_max

        # Init to default values other properties
        self.number_labels = 0
        self.thresholds = None
        self.p_norm = p_norm

    def fit(self, x, y=None):
        self.number_labels = max(y) + 1
        if self.high == 'dynamic':
            self.__fit_dynamic_thresh(x, y)
        return self

    def predict(self, x, y=None, copy=True):
        probas = self.__get_probas(x)
        if self.high == 'max':
            return np.argmax(probas, axis=1)
        else:
            return self.__get_predictions_dynamic(probas, self.thresholds)

    def predict_proba(self, x, y=None, copy=True):
        return self.__get_probas(x)

    def __fit_dynamic_thresh(self, x, y):
        probabilities = self.__get_probas(x)
        self.thresholds = np.zeros(self.number_labels)

        if self.is_prior_max:
            iterator = reversed(range(self.number_labels))
        else:
            iterator = range(self.number_labels)

        for hierarchy in iterator:
            score = 0
            potential_thresholds = np.sort(np.unique(np.concatenate(([1, 0], probabilities[:, hierarchy]))))
            label = y == hierarchy
            probability = probabilities[:, hierarchy]
            for thresh in potential_thresholds:
                threshed = probability >= thresh
                tmp_score = self.metric(label, threshed)
                if tmp_score > score:
                    score = tmp_score
                    self.thresholds[hierarchy] = thresh

    def __get_predictions_at_least_one(self, x):
        if self.is_prior_max:
            return np.amax(x, axis=1)
        else:
            return np.amin(x, axis=1)

    def __get_predictions_dynamic(self, x, thresholds):
        return np.argmax((x >= thresholds) * self.__get_prior_coefficients(), axis=1)

    def __get_prior_coefficients(self):
        coefficients = range(self.number_labels)
        if not self.is_prior_max:
            coefficients = reversed(coefficients)

        return np.array(list(coefficients))

    def __get_probas(self, x):
        x_probas = np.zeros((len(x), self.number_labels))
        # patches_number = x.shape[1]
        for index, group in enumerate(x):
            if self.low == 'mean':
                x_probas[index, :] = np.mean(group, axis=0)
            elif self.low == 'max':
                x_probas[index, :] = np.max(group, axis=0)
            else:
                x_probas[index, :] = np.linalg.norm(group, ord=self.p_norm, axis=0)
        return x_probas


class MajorityVotingClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, voting='hard', weights=None, flatten_transform=True, verbose=False):
        self.voting = voting
        self.weights = weights
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        if self.voting == 'soft':
            maj = np.argmax(X, axis=1)
        else:
            X = X.astype(int)
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=X)

        return maj


# class MultimodalClassifier(BaseEstimator, ClassifierMixin):
#
#     def __init__(self, threshold='max'):
#         self.threshold = threshold
#
#     def fit(self, x, y=None):
#
#         return self
#
#     def predict(self, x, y=None, copy=True):
        


class KerasClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    """

    def __getstate__(self):
        state = self.__dict__
        if "model" in state:
            model = state["model"]
            model_hdf5_bio = io.BytesIO()
            with h5py.File(model_hdf5_bio, mode="w") as file:
                model.save(file)
            state["model"] = model_hdf5_bio
            state_copy = copy.deepcopy(state)
            state["model"] = model
            return state_copy
        else:
            return state

    def __setstate__(self, state):
        if "model" in state:
            model_hdf5_bio = state["model"]
            with h5py.File(model_hdf5_bio, mode="r") as file:
                state["model"] = tf.keras.models.load_model(file)
        self.__dict__ = state


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
                if Utils.has_arg(fn, param_name):
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

    def fit(self, X, y, callbacks=[], X_validation=None, y_validation=None, **params):
        self.init_model(y)

        # Get arguments for predict
        all_params = deepcopy(self.sk_params)
        all_params.update(params)
        params_fit = self.filter_params(all_params, Sequential.fit_generator)

        # Get generator
        train = self.create_generator(X=X, y=y, params=params)
        validation = None
        if X_validation is not None:
            validation = self.create_generator(X=X_validation, y=y_validation, params=params, prediction_mode=True)

        if not self.model._is_compiled:
            tr_x, tr_y = train[0]
            self.model.fit(tr_x, tr_y)

        return self.model.fit_generator(generator=train, validation_data=validation, callbacks=callbacks, **params_fit)

    def predict(self, X, **params):
        probs = self.predict_proba(X, **params)
        if probs.shape[-1] > 1:
            classes = probs.argmax(axis=-1)
        else:
            classes = (probs > 0.5).astype('int32')
        return self.classes_[classes]

    def predict_proba(self, X, **params):
        self.init_model()

        # Define some local arguments
        all_params = deepcopy(self.sk_params)
        all_params.update(params)

        # Create generator for validation
        valid = self.create_generator(X=X, params=all_params, prediction_mode=True)

        # Get arguments for predict
        params_pred = self.filter_params(all_params, Sequential.predict_generator)
        probs = self.model.predict_generator(generator=valid, **params_pred)

        # check if binary classification
        if len(probs) > 0 and probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = hstack([1 - probs, probs])
        return probs

    def score(self, X, y, **params):
        self.init_model()

        # Define some local arguments
        all_params = deepcopy(self.sk_params)
        all_params.update(params)

        # Create generator for validation
        valid = self.create_generator(X=X, params=all_params, prediction_mode=True)

        # Get arguments for fit
        params_eval = self.filter_params(all_params, Sequential.evaluate_generator)
        outputs = self.model.evaluate_generator(generator=valid, **params_eval)
        outputs = Utils.to_list(outputs)
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')

    def create_generator(self, X, y=None, params={}, prediction_mode=False):
        # Init generator
        params_init = deepcopy(self.sk_params)
        params_init.update(params)
        if prediction_mode:
            # Data need to be preprocessed
            generator = ResourcesGenerator(preprocessing_function=params.get('preprocessing_function', None))
        else:
            params_init = self.filter_params(params_init, ResourcesGenerator.__init__)
            generator = ResourcesGenerator(**params_init)

        # Create iterator
        params_flow = deepcopy(self.sk_params)
        params_flow.update(params)
        params_flow = self.filter_params(params_flow, ResourcesGenerator.flow_from_paths)

        if prediction_mode:
            params_flow.update({'shuffle': False})
            # params_flow.update({'batch_size': 1})

        if y is not None:
            return generator.flow_from_paths(X, y, **params_flow)
        else:
            return generator.flow_from_paths(X, **params_flow)

    def filter_params(self, params, fn, override=None):
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
            if Utils.has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res

    def summary(self):
        self.init_model()
        self.model.summary()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)


class KerasFineClassifier(KerasBatchClassifier):

    def check_params(self, params):
        """Checks for user typos in `params`.

        # Arguments
            params: dictionary; the parameters to be checked

        # Raises
            ValueError: if any member of `params` is not a valid argument.
        """
        local_param = deepcopy(params)
        if 'trainable_layer' in local_param:
            local_param.pop('trainable_layer')
        if 'extractor_layer' in local_param:
            local_param.pop('extractor_layer')
        super().check_params(local_param)

    def fit(self, X, y, callbacks=[], X_validation=None, y_validation=None, **params):
        self.init_model(y)

        # Get arguments for predict
        all_params = deepcopy(self.sk_params)
        all_params.update(params)
        params_fit = self.filter_params(all_params, Sequential.fit_generator)

        # Get generator
        train = self.create_generator(X=X, y=y, params=params)
        validation = None
        if X_validation is not None:
            validation = self.create_generator(X=X_validation, y=y_validation, params=params, prediction_mode=True)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.model.layers:
            if 'prediction' not in layer.name:
                layer.trainable = False

        if hasattr(self, 'two_step_training'):
            # compile the model (should be done *after* setting layers to non-trainable)
            self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            print('Pre-training...')
            self.history = self.model.fit_generator(generator=train, validation_data=validation,
                                                    callbacks=[EarlyStopping(monitor='loss', patience=5)],
                                                    class_weight=train.get_weights(), **params_fit)

            trainable_layer = params.get('trainable_layer', 0)
            for layer in self.model.layers[trainable_layer:]:
                layer.trainable = True

            # we need to recompile the model for these modifications to take effect
            # we use SGD with a low learning rate
            self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

            print('Final-training...')
            # we train our model again (this time fine-tuning the top 2 inception blocks
            # alongside the top Dense layers
            self.history = self.model.fit_generator(generator=train, validation_data=validation, callbacks=callbacks,
                                                    class_weight=train.get_weights(), **params_fit)
        else:
            trainable_layer = params.get('trainable_layer', 0)
            for layer in self.model.layers[trainable_layer:]:
                layer.trainable = True

            # we need to recompile the model for these modifications to take effect
            # we use SGD with a low learning rate
            self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
                               metrics=['accuracy'])

            # we train our model again (this time fine-tuning the top 2 inception blocks
            # alongside the top Dense layers
            self.history = self.model.fit_generator(generator=train, validation_data=validation, callbacks=callbacks,
                                                    class_weight=train.get_weights(), **params_fit)
        return self.history

    def transform(self, X, **params):
        self.init_model()

        # Define some local arguments
        all_params = deepcopy(self.sk_params)
        all_params.update(params)

        # Create generator for validation
        valid = self.create_generator(X=X, params=all_params, prediction_mode=True)

        # Predict!
        extractor_layers = all_params.get('extractor_layers', None)
        if extractor_layers is None:
            model = self.model
        else:
            model = Model(self.model.inputs, self.model.layers[extractor_layers].output)

        # Get arguments for predict
        params_pred = self.filter_params(all_params, Sequential.predict_generator)
        probs = model.predict_generator(generator=valid, **params_pred)

        # check if binary classification
        if len(probs) > 0 and probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = hstack([1 - probs, probs])
        return probs


class Utils:
    def has_arg(fn, name, accept_all=False):
        """Checks if a callable accepts a given keyword argument.
        For Python 2, checks if there is an argument with the given name.
        For Python 3, checks if there is an argument with the given name, and
        also whether this argument can be called with a keyword (i.e. if it is
        not a positional-only argument).
        # Arguments
            fn: Callable to inspect.
            name: Check if `fn` can be called with `name` as a keyword argument.
            accept_all: What to return if there is no parameter called `name`
                        but the function accepts a `**kwargs` argument.
        # Returns
            bool, whether `fn` accepts a `name` keyword argument.
        """
        if sys.version_info < (3,):
            arg_spec = inspect.getargspec(fn)
            if accept_all and arg_spec.keywords is not None:
                return True
            return (name in arg_spec.args)
        elif sys.version_info < (3, 3):
            arg_spec = inspect.getfullargspec(fn)
            if accept_all and arg_spec.varkw is not None:
                return True
            return (name in arg_spec.args or
                    name in arg_spec.kwonlyargs)
        else:
            signature = inspect.signature(fn)
            parameter = signature.parameters.get(name)
            if parameter is None:
                if accept_all:
                    for param in signature.parameters.values():
                        if param.kind == inspect.Parameter.VAR_KEYWORD:
                            return True
                return False
            return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                       inspect.Parameter.KEYWORD_ONLY))

    def to_list(x, allow_tuple=False):
            """Normalizes a list/tensor into a list.
            If a tensor is passed, we return
            a list of size 1 containing the tensor.
            # Arguments
                x: target object to be normalized.
                allow_tuple: If False and x is a tuple,
                    it will be converted into a list
                    with a single element (the tuple).
                    Else converts the tuple to a list.
            # Returns
                A list.
            """
            if isinstance(x, list):
                return x
            if allow_tuple and isinstance(x, tuple):
                return list(x)
            return [x]
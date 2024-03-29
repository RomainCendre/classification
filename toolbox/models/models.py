import copy
import inspect
import io
import sys
import types
import warnings

import h5py
import numpy as np
from copy import deepcopy
from joblib import delayed, Parallel
from numpy import hstack, arange
from sklearn import calibration as calib_file
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from tensorflow.keras import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.multiclass import _fit_binary
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.multiclass import unique_labels, _ovr_decision_function
import tensorflow as tf
from toolbox.models.generators import ResourcesGenerator


class CustomCalibrationCV(CalibratedClassifierCV):
    def fit(self, X, y, sample_weight=None):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # X, y = self._validate_data(X, y, accept_sparse=['csc', 'csr', 'coo'],
        #                            force_all_finite=False, allow_nd=True)
        X, y = calib_file.indexable(X, y)
        le = LabelBinarizer().fit(y)
        self.classes_ = le.classes_

        # Check that each cross-validation fold can have at least one
        # example per class
        n_folds = self.cv if isinstance(self.cv, int) \
            else self.cv.n_folds if hasattr(self.cv, "n_folds") else None
        if n_folds and \
                np.any([np.sum(y == class_) < n_folds for class_ in
                        self.classes_]):
            raise ValueError("Requesting %d-fold cross-validation but provided"
                             " less than %d examples for at least one class."
                             % (n_folds, n_folds))

        self.calibrated_classifiers_ = []
        if self.base_estimator is None:
            # we want all classifiers that don't expose a random_state
            # to be deterministic (and we don't want to expose this one).
            base_estimator = LinearSVC(random_state=0)
        else:
            base_estimator = self.base_estimator

        if self.cv == "prefit":
            calibrated_classifier = calib_file._CalibratedClassifier(
                base_estimator, method=self.method)
            calibrated_classifier.fit(X, y, sample_weight)
            self.calibrated_classifiers_.append(calibrated_classifier)
        else:
            cv = calib_file.check_cv(self.cv, y, classifier=True)
            fit_parameters = calib_file.signature(base_estimator.fit).parameters
            base_estimator_supports_sw = "sample_weight" in fit_parameters

            if sample_weight is not None:
                sample_weight = calib_file._check_sample_weight(sample_weight, X)

                if not base_estimator_supports_sw:
                    estimator_name = type(base_estimator).__name__
                    warnings.warn("Since %s does not support sample_weights, "
                                  "sample weights will only be used for the "
                                  "calibration itself." % estimator_name)

            for train, test in cv.split(X, y):
                this_estimator = clone(base_estimator)

                if sample_weight is not None and base_estimator_supports_sw:
                    this_estimator.fit(X[train], y[train],
                                       sample_weight=sample_weight[train])
                else:
                    this_estimator.fit(X[train], y[train])

                calibrated_classifier = calib_file._CalibratedClassifier(
                    this_estimator, method=self.method, classes=self.classes_)
                sw = None if sample_weight is None else sample_weight[test]
                calibrated_classifier.fit(X[test], y[test], sample_weight=sw)
                self.calibrated_classifiers_.append(calibrated_classifier)

        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        calib_file.check_is_fitted(self)
        # X = check_array(X, accept_sparse=['csc', 'csr', 'coo'],
        #                 force_all_finite=False)
        # Compute the arithmetic mean of the predictions of the calibrated
        # classifiers
        mean_proba = np.zeros((X.shape[0], len(self.classes_)))
        for calibrated_classifier in self.calibrated_classifiers_:
            proba = calibrated_classifier.predict_proba(X)
            mean_proba += proba

        mean_proba /= len(self.calibrated_classifiers_)

        return mean_proba


class CustomMIL(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):  # Based on OneVsOne

    def __init__(self, estimator, data_preparation=None, n_jobs=None):
        self.is_inconsistent = True
        self.data_preparation = data_preparation
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, bags, y):
        if self.data_preparation is not None:
            self.data_preparation.fit(np.concatenate(bags))
            for index, bag in enumerate(bags):
                bags[index] = self.data_preparation.transform(bag)

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

    def predict(self, bags):
        if self.data_preparation is not None:
            for index, bag in enumerate(bags):
                bags[index] = self.data_preparation.transform(bag)

        Y = self.__decision_function(bags)
        Y = self.classes_[Y.argmax(axis=1)]
        return Y

    def predict_instance(self, bags):
        if self.data_preparation is not None:
            for index, bag in enumerate(bags):
                bags[index] = self.data_preparation.transform(bag)

        Y = self.__decision_function(bags, instancePrediction=True)
        Y = self.classes_[Y.argmax(axis=1)]
        Y = Y.tolist()
        # Y = self.estimators_[0].predict(X, instancePrediction=True)[1].tolist()
        predictions = []
        for x in bags:
            predictions.append(Y[:len(x)])
            del Y[:len(x)]
        return predictions

    # def predict_proba(self, bags):
    #     if self.data_preparation is not None:
    #         for index, bag in enumerate(bags):
    #             bags[index] = self.data_preparation.transform(bag)
    #
    #     Y = self.__decision_function(bags)
    #     Y = (Y - np.min(Y))
    #     Y = Y / np.max(Y)
    #     return Y
    #
    # def predict_proba_instance(self, bags):
    #     if self.data_preparation is not None:
    #         for index, bag in enumerate(bags):
    #             bags[index] = self.data_preparation.transform(bag)
    #
    #     Y = self.__decision_function(bags, instancePrediction=True)
    #     Y = (Y - np.min(Y))
    #     Y = Y / np.max(Y)
    #     Y = Y.tolist()
    #     # Y = self.estimators_[0].predict(X, instancePrediction=True)[1].tolist()
    #     predictions = []
    #     for x in bags:
    #         predictions.append(Y[:len(x)])
    #         del Y[:len(x)]
    #     return predictions

    def __decision_function(self, X, instancePrediction=None):
        indices = self.pairwise_indices_
        if indices is None:
            Xs = [X] * len(self.estimators_)
        else:
            Xs = [X[:, idx] for idx in indices]

        predictions = np.vstack(
            [self.__est_predict(est, Xi, instancePrediction=instancePrediction) for est, Xi in
             zip(self.estimators_, Xs)]).T
        confidences = np.vstack(
            [self.__est_predict_binary(est, Xi, instancePrediction=instancePrediction) for est, Xi in
             zip(self.estimators_, Xs)]).T
        Y = _ovr_decision_function(predictions, confidences, len(self.classes_))
        return Y

    def decision_function(self, X, instancePrediction=None):
        Y = self.__decision_function(X, instancePrediction)
        if self.n_classes_ == 2:
            return Y[:, 1]
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
        if instancePrediction is None:
            predictions = estimator.predict(bags)
        else:
            predictions = estimator.predict(bags, instancePrediction)[1]
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
        # Return max, in cas of equality, we return highest priority class
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
        # Return max, in cas of equality, we return highest priority class
        if self.high == 'max':
            maximum = probas.max(axis=1, keepdims=1) == probas
            return (maximum * self.__get_prior_coefficients()).argmax(axis=1)
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
        # Why iterating and not a direct compute on matrices?
        # Because row size can be different on X ( cf patient decision for microscopy, with varying number of images )
        for index, group in enumerate(x):
            if self.low == 'mean':
                x_probas[index, :] = np.mean(group, axis=0)
            elif self.low == 'max':
                x_probas[index, :] = np.max(group, axis=0)
            else:
                x_probas[index, :] = np.linalg.norm(group, ord=self.p_norm, axis=0)
        return x_probas


class MultimodalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, method='modality', ordered=True, metric=None, from_zero=True):
        mandatory = ['modality', 'modality_class']
        if method not in mandatory:
            raise Exception(f'Invalid method.')

        self.method = method
        self.ordered = ordered
        self.from_zero = from_zero

        if metric:
            self.metric = metric
        else:
            self.metric = accuracy_score
        self.thresholds = None

    def fit(self, x, y=None):
        num_modal = x.shape[1]
        if self.from_zero:
            if self.method == 'modality':
                self.thresholds = np.zeros(num_modal)
                for modality in arange(num_modal):
                    x_mod = x[:, modality, :]
                    highest = 0
                    for thresh in sorted(list(x_mod.flatten()), reverse=self.ordered):
                        thresholds = self.thresholds.copy()
                        thresholds[modality] = thresh
                        predictions = MultimodalClassifier.__get_predictions(x, thresholds)
                        score = self.metric(y, predictions)
                        if score > highest:
                            highest = score
                            self.thresholds[modality] = thresh
            else:
                self.thresholds = np.zeros(x.shape[1:])
                for modality in arange(num_modal):
                    for classe in arange(x.shape[2]):
                        x_mod = x[:, modality, classe]
                        highest = 0
                        for thresh in sorted(list(x_mod.flatten()), reverse=self.ordered):
                            thresholds = self.thresholds.copy()
                            thresholds[modality, classe] = thresh
                            predictions = MultimodalClassifier.__get_predictions(x, thresholds)
                            score = self.metric(y, predictions)
                            if score > highest:
                                highest = score
                                self.thresholds[modality, classe] = thresh
        else:
            if self.method == 'modality':
                self.thresholds = np.ones(num_modal)
                for modality in reversed(arange(num_modal)):
                    x_mod = x[:, modality, :]
                    highest = 0
                    for thresh in sorted(list(x_mod.flatten()), reverse=self.ordered):
                        thresholds = self.thresholds.copy()
                        thresholds[modality] = thresh
                        predictions = MultimodalClassifier.__get_predictions(x, thresholds)
                        score = self.metric(y, predictions)
                        if score > highest:
                            highest = score
                            self.thresholds[modality] = thresh
            else:
                self.thresholds = np.ones(x.shape[1:])
                for modality in reversed(arange(num_modal)):
                    for classe in arange(x.shape[2]):
                        x_mod = x[:, modality, classe]
                        highest = 0
                        for thresh in sorted(list(x_mod.flatten()), reverse=self.ordered):
                            thresholds = self.thresholds.copy()
                            thresholds[modality, classe] = thresh
                            predictions = MultimodalClassifier.__get_predictions(x, thresholds)
                            score = self.metric(y, predictions)
                            if score > highest:
                                highest = score
                                self.thresholds[modality, classe] = thresh

        return self

    def predict(self, x, y=None, copy=True):
        return self.__get_predictions(x, self.thresholds)

    def predict_proba(self, x, y=None, copy=True):
        return self.__get_probabilities(x, self.thresholds)

    def predict_steps(self, x, y=None):
        return MultimodalClassifier.__get_steps(x, self.thresholds)

    @staticmethod
    def __get_predictions(x, thresholds):
        probabilities = MultimodalClassifier.__get_probabilities(x, thresholds)
        return np.argmax(probabilities, axis=1)

    @staticmethod
    def __get_probabilities(x, thresholds):
        steps = MultimodalClassifier.__get_steps(x, thresholds)
        probabilities = np.zeros([x.shape[0], x.shape[2]])
        for sample, modality in enumerate(steps):
            mask = x[sample, modality, :] > thresholds[modality]
            probabilities[sample, mask] = x[sample, modality, mask]
        return probabilities

    @staticmethod
    def __get_steps(x, thresholds):
        # Number of classes
        num_classes = x.shape[1]
        # Make priority matrix on first modalities
        modalities = np.tile(np.expand_dims(np.array(list(reversed(range(num_classes)))), axis=1), (1, x.shape[2]))
        indices = np.tile(modalities, (x.shape[0], 1, 1))
        # Get the mask, and put missing values to lower priority
        mask = MultimodalClassifier.__get_mask(x, thresholds)
        indices[~mask] = 0
        # Return modalities
        return (num_classes - 1) - np.amax(np.amax(indices, axis=2), axis=1)

    @staticmethod
    def __get_mask(x, thresholds):
        mask = np.full(x.shape, False)
        for jndex, treshold in enumerate(thresholds):
            cmask = x[:, jndex, :] > treshold
            mask[:, jndex, :] = cmask
        return mask


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
            # state_copy = copy.deepcopy(state)
            # state["model"] = model
            return state
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
        train = self.create_generator(X=X, y=y, params=all_params)
        validation = None
        if X_validation is not None:
            validation = self.create_generator(X=X_validation, y=y_validation, params=all_params, prediction_mode=True)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.model.layers:
            if 'prediction' not in layer.name:
                layer.trainable = False

        # if hasattr(self, 'two_step_training'):
        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Pre-training...')
        self.history = self.model.fit_generator(generator=train, validation_data=validation,
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

    @tf.function
    def macro_soft_f1(y, y_hat):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = tf.reduce_mean(cost)  # average on all labels
        return macro_cost

    @tf.function
    def macro_f1(y, y_hat, thresh=0.5):
        """Compute the macro F1-score on a batch of observations (average F1 across labels)

        Args:
            y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
            thresh: probability value above which we predict positive

        Returns:
            macro_f1 (scalar Tensor): value of macro F1 for the batch
        """
        y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
        tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
        fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
        fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
        f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        macro_f1 = tf.reduce_mean(f1)
        return macro_f1
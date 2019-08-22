import types
import numpy as np
from copy import deepcopy
from keras import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.utils.generic_utils import has_arg, to_list
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import hstack
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from toolbox.core.generators import ResourcesGenerator


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
            validation = self.create_generator(X=X_validation, y=y_validation, params=params)

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
        all_params.update({'shuffle': False})
        all_params.update({'batch_size': 1})

        # Create generator for validation
        params_valid = {'preprocessing_function': all_params.get('preprocessing_function', None)}
        valid = self.create_generator(X=X, params=params_valid)

        # Get arguments for predict
        params_pred = self.filter_params(all_params, Sequential.predict_generator)
        probs = self.model.predict_generator(generator=valid, **params_pred)

        # check if binary classification
        if len(probs) > 0 and probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = hstack([1 - probs, probs])
        return probs

    def score(self, X, y, **params):

        # Define some local arguments
        all_params = deepcopy(self.sk_params)
        all_params.update(params)
        all_params.update({'shuffle': False})
        all_params.update({'batch_size': 1})

        # Create generator for validation
        params_valid = {'preprocessing_function': all_params.get('preprocessing_function', None)}
        valid = self.create_generator(X=X, params=params_valid)

        # Get arguments for fit
        params_eval = self.filter_params(all_params, Sequential.evaluate_generator)
        outputs = self.model.evaluate_generator(generator=valid, **params_eval)
        outputs = to_list(outputs)
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')

    def create_generator(self, X, y=None, params={}):
        # Init generator
        params_init = deepcopy(self.sk_params)
        params_init.update(params)
        params_init = self.filter_params(params_init, ResourcesGenerator.__init__)
        generator = ResourcesGenerator(**params_init)

        # Create iterator
        params_flow = deepcopy(self.sk_params)
        params_flow.update(params)
        params_flow = self.filter_params(params_flow, ResourcesGenerator.flow_from_paths)
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
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res


class KerasFineClassifier(KerasBatchClassifier):

    def check_params(self, params):
        """Checks for user typos in `params`.

        # Arguments
            params: dictionary; the parameters to be checked

        # Raises
            ValueError: if any member of `params` is not a valid argument.
        """
        local_param = deepcopy(params)
        if 'trainable_layers' in local_param:
            local_param.pop('trainable_layers')
        if 'extractor_layers' in local_param:
            local_param.pop('extractor_layers')
        super().check_params(local_param)

    def fit(self, X, y, callbacks=[], X_validation=None, y_validation=None, **kwargs):
        self.init_model(y)

        # Get arguments for predict
        params_fit = deepcopy(self.sk_params)
        params_fit.update(kwargs)
        params_fit = self.filter_params(params_fit, Sequential.fit_generator)

        # Get generator
        train = self.create_generator(X=X, y=y, params=kwargs)
        validation = None
        # No transformation allowed for prediction
        params = {'preprocessing_function': kwargs.get('preprocessing_function', None)}
        if X_validation is not None:
            validation = self.create_generator(X=X_validation, y=y_validation, params=params)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in self.model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print('Pre-training...')
        self.history = self.model.fit_generator(generator=train, validation_data=validation,
                                                callbacks=[EarlyStopping(monitor='loss', patience=5)],
                                                class_weight=train.get_weights(), **params_fit)

        trainable_layer = kwargs.get('trainable_layers', 0)
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

    def predict_proba(self, X, **kwargs):
        self.init_model()

        # Define some local arguments
        copy_kwargs = deepcopy(kwargs)
        copy_kwargs.update({'shuffle': False})
        copy_kwargs.update({'batch_size': 1})

        # No transformation allowed for prediction
        params = {'preprocessing_function': kwargs.get('preprocessing_function', None)}

        # Create generator
        valid = self.create_generator(X=X, params=params)

        # Get arguments for predict
        params_pred = deepcopy(self.sk_params)
        params_pred.update(copy_kwargs)
        params_pred = self.filter_params(params_pred, Sequential.predict_generator)

        # Predict!
        extractor_layers = kwargs.get('extractor_layers', None)
        if extractor_layers is None:
            model = self.model
        else:
            model = Model(self.model.inputs, self.model.layers[extractor_layers])

        probs = model.predict_generator(generator=valid, **params_pred)

        # check if binary classification
        if len(probs) > 0 and probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = hstack([1 - probs, probs])
        return probs


class DecisionVotingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, mode='max', metric=None, prior_class_max=True):
        self.mode = mode
        # Set metric mode used for evaluation
        if metric:
            self.metric = metric
        else:
            self.metric = accuracy_score

        self.is_prior_class_max = prior_class_max

        # Init to default values other properties
        self.number_labels = 0
        self.thresholds = None

    def fit(self, x, y=None):
        self.number_labels = max(y) + 1
        if self.mode == 'dynamic_thresh':
            self._fit_dynamic_thresh(x, y)
            print(self.thresholds)
        return self

    def predict(self, x, y=None, copy=True):
        if self.mode == 'max':
            x = self._get_decisions_probas(x)
            return self._get_predictions_max(x)
        elif self.mode == 'at_least_one':
            return self._get_predictions_at_least_one(x)
        elif self.mode == 'dynamic_thresh':
            x = self._get_decisions_probas(x)
            return self._get_predictions(x, self.thresholds)

    def _fit_dynamic_thresh(self, x, y):
        x_probas = self._get_decisions_probas(x)

        global_score = 0
        self.thresholds = np.zeros(self.number_labels)
        for hierarchy in range(self.number_labels):
            potential_thresholds = np.sort(np.unique(x_probas[:, hierarchy]))
            for thresh in potential_thresholds:
                thresholds = np.copy(self.thresholds)
                thresholds[hierarchy] = thresh
                score = self.metric(self._get_predictions(x_probas, thresholds), y)
                if global_score < score:
                    global_score = score
                    self.thresholds[hierarchy] = thresh

    def _get_decisions_probas(self, x):
        x_probas = np.zeros((len(x), self.number_labels))
        patches_number = x.shape[1]
        for label in range(self.number_labels):
            x_probas[:, label] = np.sum(x == label, axis=1) / patches_number
        return x_probas

    def _get_predictions_at_least_one(self, x):
        if self.is_prior_class_max:
            return np.amax(x, axis=1)
        else:
            return np.amin(x, axis=1)

    def _get_predictions_max(self, x):
        maximum = x.max(axis=1, keepdims=1) == x
        return (maximum * self._get_prior_coefficients()).argmax(axis=1)

    def _get_prior_coefficients(self):
        coefficients = range(self.number_labels)
        if self.is_prior_class_max:
            coefficients = reversed(coefficients)

        return np.array(list(coefficients))

    def _get_predictions(self, x, thresholds):
        return np.argmax((x > thresholds) * self._get_prior_coefficients(), axis=1)

    def predict_proba(self, x, y=None, copy=True):
        return x


class ScoreVotingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, mode='max', hierarchies=None, metric=None):
        self.mode = mode
        # Set metric mode used for evaluation
        if metric:
            self.metric = metric
        else:
            self.metric = accuracy_score

        # Check if we are in dynamic mode, if yes check for args
        if self.mode in ['dynamic', 'min']:
            self.hierarchies = hierarchies
            self.thresholds = None

    def fit(self, x, y=None):
        print('None')

    def predict(self, x, y=None, copy=True):
        print('None')

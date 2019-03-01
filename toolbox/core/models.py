from copy import deepcopy, copy
from sklearn.metrics import accuracy_score
from keras.engine import Layer
from keras.layers import K
from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.generic_utils import has_arg, to_list
from numpy import arange, array, searchsorted, unique, hstack, zeros, concatenate, ones, argmax, sort
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
import types
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
        y = array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = unique(y)
            y = searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))

    def fit(self, X, y, callbacks=[], X_validation=None, y_validation=None, **kwargs):
        self.init_model(y)

        # Get arguments for fit
        fit_args = deepcopy(self.filter_sk_params(Sequential.fit_generator))
        fit_args.update(kwargs)

        # Get generator
        train = self.__create_generator(X=X, y=y)
        validation = None
        if X_validation is not None:
            validation = self.__create_generator(X=X_validation, y=y_validation)

        if not self.model._is_compiled:
            tr_x, tr_y = train[0]
            self.model.fit(tr_x, tr_y)

        self.history = self.model.fit_generator(generator=train, validation_data=validation, callbacks=callbacks,
                                                **fit_args)

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
        # Get arguments for generator
        fit_args = deepcopy(self.filter_sk_params(ResourcesGenerator.__init__))
        fit_args.update(self.__filter_params(kwargs, ResourcesGenerator.flow_from_paths))
        fit_args.update({'shuffle': False})
        fit_args.update({'batch_size': 1})

        # Create generator
        valid = self.__create_generator(X=X, params=fit_args)

        # Get arguments for predict
        fit_args = deepcopy(self.filter_sk_params(Sequential.predict_generator))
        fit_args.update(self.__filter_params(kwargs, Sequential.predict_generator))

        probs = self.model.predict_generator(generator=valid, **fit_args)

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
        outputs = to_list(outputs)
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')

    def __create_generator(self, X, y=None, params={}):
        generator = ResourcesGenerator(**self.__filter_params(params, ResourcesGenerator.__init__))
        if y is not None:
            return generator.flow_from_paths(X, y, **self.__filter_params(params, ResourcesGenerator.flow_from_paths))
        else:
            return generator.flow_from_paths(X, **self.__filter_params(params, ResourcesGenerator.flow_from_paths))

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
                    sub_predictions = zeros(current.shape)
                sub_predictions[:, current.argmax(axis=1)] += 1
                # sub_probabilities.append(current)
            predictions.append(sub_predictions/len(patches))

        return concatenate(array(predictions), axis=0)

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


class RandomLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.random_uniform_variable(shape=(1, self.output_dim), low=0, high=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base_config = super().get_config()
        base_config['output_dim'] = self.output_dim
        return base_config


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
        # if not x:
        #     raise Exception('At least one X has to be found.')
        global_score = 0
        self.thresholds = zeros(len(self.hierarchies))
        for index, hierarchy in enumerate(self.hierarchies):
            potential_thresholds = sort(unique(x[:, hierarchy]))
            for thresh in potential_thresholds:
                thresholds = copy(self.thresholds)
                thresholds[index] = thresh
                score = self.metric(self.get_predictions(x, thresholds), y)
                if global_score < score:
                    global_score = score
                    self.thresholds[index] = thresh
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
        thresh_values = ((ones(x.shape) * thresholds) < x).astype(int)
        priorities = arange(len(thresholds), 0, -1)[self.hierarchies]
        return argmax(thresh_values*priorities, axis=1)

    def predict_proba(self, x, y=None, copy=True):
        return x

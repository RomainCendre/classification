import types
import copy
from keras import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.generic_utils import has_arg, to_list
from numpy import arange, array, unique, searchsorted, hstack
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from toolbox.core.generators import ResourcesGenerator
from toolbox.core.structures import Results, Result


class Classifier:
    """Class that manage a spectrum representation.

     In this class we afford an object that represent a spectrum and all
     needed method to manage it.

     Attributes:
         __pipeline (:obj:):
         __params (:obj:):
         __inner_cv (:obj:):
         __outer_cv (:obj:):

     """

    def __init__(self, model, params, inner_cv, outer_cv, scoring=None):
        """Make an initialisation of SpectraClassifier object.

        Take a pipeline object from scikit learn to experiences data and params for parameters
        to cross validate.

        Args:
             pipeline (:obj:):
             params (:obj:):
             inner_cv (:obj:):
             outer_cv (:obj:):
        """
        self.__model = model
        self.__params = params
        self.__inner_cv = inner_cv
        self.__outer_cv = outer_cv
        self.scoring = scoring

    def change_model(self, model, params):
        self.__model = model
        self.__params = params

    def evaluate(self, inputs, name='Default'):
        """

        Args:
            inputs (:obj:):
            name :
         Returns:

        """
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()
        reference = inputs.get_reference()

        # Encode labels to go from string to int
        results = []
        for fold, (train, test) in enumerate(self.__outer_cv.split(X=datas, y=labels, groups=groups)):
            print('Fold : {fold}'.format(fold=fold+1))

            grid_search = GridSearchCV(estimator=self.__model, param_grid=self.__params, cv=self.__inner_cv,
                                       scoring=self.scoring, verbose=1, iid=False)

            if groups is not None:
                grid_search.fit(X=datas[train], y=labels[train], groups=groups[train])
            else:
                grid_search.fit(X=datas[train], y=labels[train])

            # Try to predict test data
            probabilities = grid_search.predict_proba(datas[test])
            predictions = grid_search.predict(datas[test])

            # Now store all computed data
            for index, test_index in enumerate(test):
                result = Result()
                result.update({"Fold": fold})
                result.update({"Label": labels[test_index]})
                if reference is not None:
                    result.update({"Reference": reference[test_index]})

                # Get probabilities and predictions
                result.update({"Probability": probabilities[index]})
                result.update({"Prediction": predictions[index]})

                # Append element and go on next one
                results.append(result)

        return Results(results, name)

    def fit(self, inputs):
        # Extract needed data
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()

        # Fit to data
        grid_search = GridSearchCV(estimator=self.__model, param_grid=self.__params, cv=self.__inner_cv,
                                   scoring=self.scoring, verbose=1, iid=False)

        if groups is not None:
            grid_search.fit(X=datas, y=labels, groups=groups)
        else:
            grid_search.fit(X=datas, y=labels)

        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate_patch(self, inputs, benign, malignant, name='Default', patch_size=250):
        # Extract needed data
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()
        reference = inputs.get_reference()

        # Prepare data
        generator = ResourcesGenerator(preprocessing_function=self.__params.get('preprocessing_function', None))
        test = generator.flow_from_paths(datas, batch_size=1, shuffle=False)

        # Encode labels to go from string to int
        results = []
        for index in arange(len(test)):
            x, y = test[index]

            result = Result()
            result.update({"Label": labels[index]})
            if reference is not None:
                result.update({"Reference": reference[index]})

            prediction = []
            for i in range(0, 4):
                for j in range(0, 4):
                    x_patch = x[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
                    prediction.append(Classifier.__predict_classes(probabilities=self.__model.predict(x_patch)))

            # Kept predictions
            result.update({"Prediction": malignant if prediction.count(malignant) > 0 else benign})
            results.append(result)

        return Results(results, name)

    @staticmethod
    def __predict_classes(probabilities):
        if probabilities.shape[-1] > 1:
            return probabilities.argmax(axis=-1)
        else:
            return (probabilities > 0.5).astype('int32')


class KerasBatchClassifier(KerasClassifier):

    def check_params(self, params):
        """Checks for user typos in `params`.

        # Arguments
            params: dictionary; the parameters to be checked

        # Raises
            ValueError: if any member of `params` is not a valid argument.
        """
        local_param = copy.deepcopy(params)
        legal_params_fns = [ResourcesGenerator.__init__, ResourcesGenerator.flow_from_paths]
        for params_name in params:
            for fn in legal_params_fns:
                if has_arg(fn, params_name):
                    local_param.pop(params_name)
        super().check_params(local_param)

    def fit(self, X, y, **kwargs):
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

        # Get arguments for fit
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit_generator))
        fit_args.update(kwargs)

        # Create generator
        generator = ResourcesGenerator(**self.filter_sk_params(ResourcesGenerator.__init__))
        train = generator.flow_from_paths(X, y, **self.filter_sk_params(ResourcesGenerator.flow_from_paths))

        self.history = self.model.fit_generator(generator=train, **fit_args)

        return self.history

    def predict(self, X, **kwargs):
        probs = self.predict_proba(X, **kwargs)
        if probs.shape[-1] > 1:
            classes = probs.argmax(axis=-1)
        else:
            classes = (probs > 0.5).astype('int32')
        return self.classes_[classes]

    def predict_proba(self, X, **kwargs):
        kwargs = self.filter_sk_params(Sequential.predict_generator, kwargs)

        # Create generator
        generator = ResourcesGenerator(preprocessing_function=kwargs.get('Preprocess', None))
        valid = generator.flow_from_paths(X, batch_size=1, shuffle=False)

        # Get arguments for fit
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.predict_generator))
        fit_args.update(kwargs)

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
        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.evaluate_generator))
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

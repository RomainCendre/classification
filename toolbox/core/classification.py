import types
import copy
from keras import Model, Sequential
from keras.models import clone_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.generic_utils import has_arg, to_list
from numpy import concatenate, arange, newaxis, array, unique, searchsorted, hstack
from os.path import join
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels

from toolbox.core.generators import ResourcesGenerator
from toolbox.core.structures import Results, Result
from toolbox.tools.tensorboard import TensorBoardWriter


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

        return grid_search.best_estimator_
#
#
# class ClassifierDeep:
#
#     def __init__(self, model, outer_cv, work_dir, preprocess, scoring=None):
#         self.model = model
#         self.outer_cv = outer_cv
#         self.work_dir = work_dir
#         self.scoring = scoring
#         self.generator = ResourcesGenerator(rescale=1. / 255)
#         self.preprocess = preprocess
#
#     def extract_features(self, paths, labels):
#         # Generate an iterator
#         iterator = self.generator.flow_from_paths(paths, labels, batch_size=1, class_mode='sparse')
#
#         ft = self.__get_features_extractor()
#
#         # Now try to get all bottleneck (features extracted by network)
#         bottleneck = []
#         labels = []
#         for i in range(0, 2):
#             x, y = next(iterator)
#             bottleneck.append(self.model.predict(x))
#             labels.append(y)
#
#         bottleneck = concatenate(bottleneck)
#         labels = concatenate(labels)
#         return bottleneck, labels
#
#     @staticmethod
#     def __get_callbacks(directory):
#         callbacks = []
#
#         callbacks.append(TensorBoardWriter(log_dir=directory))
#         # callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min'))
#         # callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto'))
#         return callbacks
#
#     def __get_classifier(self):
#         model = Model(outputs=self.model.output)
#         model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         return model
#
#     @staticmethod
#     def __get_class_number(labels):
#         return len(set(labels))
#
#     def __get_features_extractor(self):
#         model = Model(inputs=self.model.input)
#         model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         return model
#
#     def evaluate(self, inputs, batch_size=32, epochs=100):
#         datas = inputs.get_datas()
#         labels = inputs.get_labels()
#         groups = inputs.get_groups()
#
#         results = []
#         work_dir = self.work_dir
#         for fold, (train, valid) in enumerate(self.outer_cv.split(X=datas, y=labels, groups=groups)):
#             # Announce fold and work dir
#             print('Fold number {}'.format(fold+1))
#             self.work_dir = join(self.work_dir, 'Fold {fold}'.format(fold=(fold + 1)))
#             # Fit model
#             model = self.__fit(datas[train], labels[train], batch_size=batch_size, epochs=epochs)
#
#             # Prepare data
#             generator = ResourcesGenerator(preprocessing_function=self.preprocess)
#             valid_generator = generator.flow_from_paths(datas[valid], labels[valid],
#                                                         batch_size=1, shuffle=False)
#
#             # Folds storage
#             for index in arange(len(valid_generator)):
#                 x, y = valid_generator[index]
#                 result = Result()
#                 result.update({"Fold": fold})
#                 result.update({"Label": labels[index]})
#                 result.update({"Reference": valid_generator.filenames[index]})
#                 probability = model.predict(x)
#                 result.update({"Probability": probability[0]})
#                 # Kept predictions
#                 result.update({"Prediction": ClassifierDeep.__predict_classes(probabilities=probability)})
#                 results.append(result)
#
#             # Restore path
#             self.work_dir = work_dir
#
#         return Results(results, "Deep")
#
#     def fit(self, inputs, batch_size=32, epochs=100):
#         # Extract data for fit
#         datas = inputs.get_datas()
#         labels = inputs.get_labels()
#         return self.__fit(datas, labels, batch_size=batch_size, epochs=epochs)
#
#     def __fit(self, datas, labels, batch_size=32, epochs=100):
#
#         # Clone locally model
#         model = clone_model(self.model)
#         model.set_weights(self.model.get_weights())
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#         # Prepare data
#         generator = ResourcesGenerator(preprocessing_function=self.preprocess)
#         train_generator = generator.flow_from_paths(datas, labels, batch_size=batch_size)
#
#         # Create model and fit
#         model.fit_generator(generator=train_generator, epochs=epochs,
#                             callbacks=ClassifierDeep.__get_callbacks(self.work_dir))
#
#         return model
#
#     def evaluate_patch(self, inputs):
#         # Extract data for fit
#         paths = inputs.get_datas()
#         labels = inputs.get_labels()
#
#         # Prepare data
#         patch_size = 250
#         generator = ResourcesGenerator(preprocessing_function=self.preprocess)
#         test_generator = generator.flow_from_paths(paths, labels, batch_size=1, shuffle=False)
#
#         # Encode labels to go from string to int
#         results = []
#         for index in arange(len(test_generator)):
#             x, y = test_generator[index]
#
#             result = Result()
#             result.update({"Reference": test_generator.filenames[index]})
#             result.update({"Label": labels[index]})
#             result.update({"LabelIndex": labels[index]})
#
#             prediction = []
#             for i in range(0, 4):
#                 for j in range(0, 4):
#                     x_patch = x[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
#                     prediction.append(ClassifierDeep.__predict_classes(probabilities=self.model.predict(x_patch)))
#
#             # Kept predictions
#             result.update({"Prediction": malignant if prediction.count(malignant) > 0 else normal})
#             results.append(result)
#
#         return Results(results, "DeepPatch")
#
#     @staticmethod
#     def __predict_classes(probabilities):
#         if probabilities.shape[-1] > 1:
#             return probabilities.argmax(axis=-1)
#         else:
#             return (probabilities > 0.5).astype('int32')
#

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

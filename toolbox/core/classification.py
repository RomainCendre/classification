import glob
import warnings
from os.path import join, basename, splitext
from copy import deepcopy
from numpy import arange,  unique, asarray, mean, array_equal, save, load
from sklearn.model_selection import GridSearchCV
from toolbox.core.generators import ResourcesGenerator
from toolbox.core.models import KerasBatchClassifier
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

    def __init__(self, inner_cv, outer_cv, n_jobs=-1, callbacks=[], scoring=None):
        """Make an initialisation of SpectraClassifier object.

        Take a pipeline object from scikit learn to experiments data and params for parameters
        to cross validate.

        Args:
             params (:obj:):
             inner_cv (:obj:):
             outer_cv (:obj:):
        """
        self.__callbacks = callbacks
        self.__inner_cv = inner_cv
        self.__outer_cv = outer_cv
        self.__scoring = scoring
        self.n_jobs = n_jobs

    @staticmethod
    def sub(np_array, indices):
        if np_array is None:
            return None
        return np_array[indices]

    def set_model(self, model):

        if isinstance(model, tuple):
            self.__model = model[0]
            self.__params = model[1]
            self.__fit_params = {}
            if len(model) == 3:
                self.__fit_params = model[2]
        else:
            self.__model = model
            self.__params = {}
            self.__fit_params = {}
        self.__format_params()

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

        # Check valid labels, at least several classes
        if not self.__check_labels(labels):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Encode labels to go from string to int
        results = []
        for fold, (train, test) in enumerate(self.__outer_cv.split(X=datas, y=labels, groups=groups)):

            # Check that current fold respect labels
            if not self.__check_labels(labels, self.sub(labels, train)):
                warnings.warn('Invalid fold, missing labels for fold {fold}'.format(fold=fold + 1))
                continue

            print('Fold : {fold}'.format(fold=fold + 1))

            # Clone model
            model = deepcopy(self.__model)

            # Estimate best combination
            if self.__check_params_multiple():
                grid_search = GridSearchCV(estimator=model, param_grid=self.__params, cv=self.__inner_cv,
                                           n_jobs=self.n_jobs, scoring=self.__scoring, verbose=1, iid=False)
                grid_search.fit(self.sub(datas, train), y=self.sub(labels, train), groups=self.sub(groups, train),
                                fit_params=self.__fit_params)
                best_params = grid_search.best_params_
            else:
                best_params = self.__params

            # Fit the model, with the bests parameters
            model.set_params(**best_params)
            if isinstance(model, KerasBatchClassifier):
                model.fit(self.sub(datas, train), y=self.sub(labels, train), callbacks=self.__callbacks,
                          X_validation=self.sub(datas, test), y_validation=self.sub(labels, test), kwargs=self.__fit_params)
            else:
                model.fit(self.sub(datas, train), y=self.sub(labels, train))

            # Try to predict test data
            predictions = model.predict(datas[test])
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(datas[test])

            # Now store all computed data
            for index, test_index in enumerate(test):
                result = Result()
                result.update({"Fold": fold})
                result.update({"Label": labels[test_index]})
                if reference is not None:
                    result.update({"Reference": reference[test_index]})

                # Get probabilities and predictions
                result.update({"Prediction": predictions[index]})
                if probabilities is not None:
                    result.update({"Probability": probabilities[index]})

                # Append element and go on next one
                results.append(result)

        return Results(results, name)

    def fit(self, inputs):
        # Extract needed data
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()

        # Check valid labels, at least several classes
        if not self.__check_labels(labels):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Clone model
        model = deepcopy(self.__model)

        # Estimate best combination
        if self.__check_params_multiple():
            grid_search = GridSearchCV(estimator=self.__model, param_grid=self.__params, cv=self.__inner_cv,
                                       n_jobs=self.n_jobs, refit=False, scoring=self.__scoring, verbose=1, iid=False)
            grid_search.fit(datas, y=labels, groups=groups, **self.__fit_params)
            best_params = grid_search.best_params_
        else:
            best_params = self.__params

        # Fit the model, with the bests parameters
        model.set_params(**best_params)
        if isinstance(model, KerasBatchClassifier):
            model.fit(datas, y=labels, callbacks=self.__callbacks,
                      X_validation=datas, y_validation=labels, kwargs=self.__fit_params)
        else:
            model.fit(datas, y=labels)

        return model, best_params

    def evaluate_patch(self, inputs, benign, malignant, name='Default', patch_size=250):
        # Extract needed data
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()
        reference = inputs.get_reference()

        # Check valid labels, at least several classes
        if not self.__check_labels(labels):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Prepare data
        generator = ResourcesGenerator(preprocessing_function=self.__params.get('preprocessing_function', None))
        test = generator.flow_from_paths(datas, batch_size=1, shuffle=False)

        # Encode labels to go from string to int
        results = []
        for index in arange(len(test)):
            x, y = test[index]

            result = Result()
            result.update({"Fold": 1})
            result.update({"Label": labels[index]})
            if reference is not None:
                result.update({"Reference": reference[index]})

            prediction = []
            probability = []
            for i in range(0, 4):
                for j in range(0, 4):
                    x_patch = x[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
                    probability.append(self.__model.model.predict_proba(x_patch)[0])
                    prediction.append(Classifier.__predict_classes(probabilities=self.__model.model.predict(x_patch)))

            # Kept predictions
            probability = asarray(probability)
            result.update({"Probability": mean(probability, axis=0)})
            result.update({"Prediction": malignant if prediction.count(malignant) > 0 else benign})
            results.append(result)

        return Results(results, name)

    def features_checkpoint(self, inputs, folder):

        if self.__model is None:
            return

        # Extract needed data
        datas = inputs.get_datas()
        references = inputs.get_reference()

        # Location of folder followed by prefix
        if hasattr(self.__model, 'name'):
            prefix = self.__model.name
        else:
            prefix = type(self.__model).__name__

        folder_prefix = '{prefix}_'.format(prefix=join(folder, prefix))
        expected_files = ['{folder_prefix}{reference}.npy'.format(folder_prefix=folder_prefix, reference=reference) for reference in references]

        # Extract files from folder
        files = glob.glob('{folder_prefix}*.npy'.format(folder_prefix=folder_prefix))

        features = []
        # Check if already extracted
        if not set(expected_files).issubset(files):
            # Now browse data
            print('Extraction features with {prefix}'.format(prefix=prefix))
            if hasattr(self.__model, 'transform'):
                features = self.__model.transform(datas, **self.__params)
            else:
                features = self.__model.predict_proba(datas, **self.__params)

            # Now save features as files
            print('Writting data at {folder}'.format(folder=folder))
            for feature, reference in zip(features, references):
                save('{folder_prefix}{reference}.npy'.format(folder_prefix=folder_prefix, reference=reference), feature)
        else:
            print('Loading data at {folder}'.format(folder=folder))
            for expected_file in expected_files:
                features.append(load(expected_file))

        inputs.update_data(prefix, features, references)

    @staticmethod
    def __check_labels(labels, labels_fold=None):
        if labels_fold is None:
            return len(unique(labels)) > 1
        return len(unique(labels)) > 1 and array_equal(unique(labels), unique(labels_fold))

    def __check_params_multiple(self):
        for key, value in self.__params.items():
            if isinstance(value, list) and len(value) > 1:
                return True
        return False

    def __format_params(self):
        if self.__check_params_multiple():  # Here we proceed as multiple combination
            for key, value in self.__params.items():
                if not isinstance(value, list):
                    self.__params[key] = [value]

        else:  # Here we proceed as single combination
            for key, value in self.__params.items():
                if isinstance(value, list):
                    self.__params[key] = value[0]

    @staticmethod
    def __predict_classes(probabilities):
        if probabilities.shape[-1] > 1:
            return probabilities.argmax(axis=-1)
        else:
            return (probabilities > 0.5).astype('int32')

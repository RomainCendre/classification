import warnings
from copy import deepcopy

import h5py
import numpy as np
from numpy import unique, array_equal, array
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from toolbox.core.models import KerasBatchClassifier
from toolbox.core.structures import Outputs


class Classifier:
    """Class that manage a spectrum representation.

     In this class we afford an object that represent a spectrum and all
     needed method to manage it.

     Attributes:
         __pipeline (:obj:):
         __params (:obj:):
         __inner_cv (:obj:):

     """

    def __init__(self, inner_cv, n_jobs=-1, callbacks=[], scoring=None):
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
        self.__scoring = scoring
        self.n_jobs = n_jobs
        self.patients_folds = None

    def set_model(self, model):
        if isinstance(model, tuple):
            self.__model = model[0]
            self.__params = model[1]
            if not isinstance(self.__params, list):
                self.__params = [self.__params]
            self.__fit_params = {}
            if len(model) == 3:
                self.__fit_params = model[2]
        else:
            self.__model = model
            self.__params = []
            self.__fit_params = {}
        self.__format_params()

    def evaluate(self, inputs):
        """

        Args:
            inputs (:obj:):
            name :
         Returns:

        """
        datas = inputs.get('data')
        labels = inputs.get('label')
        groups = inputs.get('group')
        folds = inputs.get('fold')
        reference = inputs.get('reference')

        # Check valid labels, at least several classes
        if not self.__check_labels(labels):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Encode labels to go from string to int
        results = []
        for fold, test in enumerate(unique(folds)):

            train_indices = np.where(np.isin(folds, test, invert=True))[0]
            test_indices = np.where(np.isin(folds, test))[0]

            # Check that current fold respect labels
            if not self.__check_labels(labels, labels[train_indices]):
                warnings.warn('Invalid fold, missing labels for fold {fold}'.format(fold=fold + 1))
                continue

            # Remove unwanted labels that contains -1 for semisupervised
            if not hasattr(self.__model, 'is_semi_supervised') or not self.__model.is_semi_supervised:
                train_indices = train_indices[labels[train_indices] != -1]

            print('Fold : {fold}'.format(fold=fold + 1))

            # Clone model
            model = deepcopy(self.__model)

            # Estimate best combination, if single parameter combination detected,
            # GridSearch is not performed, else we launch it
            params_grid = ParameterGrid(self.__params)
            if len(params_grid) == 1:
                best_params = list(params_grid)[0]
            else:
                grid_search = GridSearchCV(estimator=model, param_grid=self.__params, cv=self.__inner_cv,
                                           n_jobs=self.n_jobs, scoring=self.__scoring, verbose=1, iid=False)
                grid_search.fit(datas[train_indices], y=labels[train_indices], groups=groups[train_indices], **self.__fit_params)
                best_params = grid_search.best_params_

            # Fit the model, with the bests parameters
            model.set_params(**best_params)
            if isinstance(model, KerasBatchClassifier):
                model.fit(datas[train_indices], y=labels[train_indices], callbacks=self.__callbacks,
                          X_validation=datas[test_indices], y_validation=labels[test_indices], **self.__fit_params)
            else:
                model.fit(datas[train_indices], y=labels[train_indices])

            # Try to predict test data
            predictions = model.predict(datas[test_indices])
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(datas[test_indices])

            # Now store all computed data
            result = dict()
            result.update({"Fold": fold})
            result.update({"Label": labels[test_indices]})
            if reference is not None:
                result.update({"Reference": reference[test_indices]})

            # Get probabilities and predictions
            result.update({"Prediction": predictions})
            if probabilities is not None:
                result.update({"Probability": probabilities})

            # Save params
            result.update({"BestParams": best_params})

            # Number of features
            result.update({"FeaturesNumber": Classifier.__number_of_features(model)})

            # Append element and go on next one
            results.append(result)

        return Outputs(results, inputs.encoders, inputs.name)

    def fit(self, inputs):
        # Extract needed data
        datas = inputs.get('data')
        labels = inputs.get('label')
        groups = inputs.get('group')

        # Check valid labels, at least several classes
        if not self.__check_labels(labels):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Clone model
        model = deepcopy(self.__model)

        # Estimate best combination
        grid_search = GridSearchCV(estimator=self.__model, param_grid=self.__params, cv=self.__inner_cv,
                                   n_jobs=self.n_jobs, refit=False, scoring=self.__scoring, verbose=1, iid=False)
        grid_search.fit(datas, y=labels, groups=groups, **self.__fit_params)
        best_params = grid_search.best_params_

        # Fit the model, with the bests parameters
        model.set_params(**best_params)
        if isinstance(model, KerasBatchClassifier):
            model.fit(datas, y=labels, callbacks=self.__callbacks,
                      X_validation=datas, y_validation=labels, **self.__fit_params)
        else:
            model.fit(datas, y=labels)

        return model, best_params

    def features_checkpoint(self, inputs):

        if self.__model is None:
            return

        # Extract needed data
        references = inputs.get('reference')

        # Location of folder followed by prefix
        if hasattr(self.__model, 'name'):
            prefix = self.__model.name
        else:
            prefix = type(self.__model).__name__

        features = None
        if inputs.get_working_folder() is not None:
            # Construct hdf5 file
            file_path = inputs.get_working_folder()/'{prefix}.hdf5'.format(prefix=prefix)
            # Try reading features if exists
            if file_path.is_file():
                try:
                    with h5py.File(file_path, 'r') as features_file:
                        if set(references).issubset(features_file.keys()):
                            features = []
                            print('Loading data at {file}'.format(file=file_path))
                            for reference in references:
                                features.append(features_file[reference][()])
                            features = array(features)
                except:
                    file_path.unlink()

            # If reading fails, so compute and write it
            if features is None:
                with h5py.File(file_path, 'a') as features_file:
                    features = self.__feature_extraction(prefix, inputs)
                    # Now save features as files
                    print('Writing data at {file}'.format(file=file_path))
                    for feature, reference in zip(features, references):
                        if reference not in features_file.keys():
                            print(reference)
                            features_file.create_dataset(reference, data=feature)
        else:
            features = self.__feature_extraction(prefix, inputs)

        # Update input
        inputs.update(prefix, features, references, 'data')

    def __feature_extraction(self, prefix, inputs):
        # Now browse data
        print('Extraction features with {prefix}'.format(prefix=prefix))

        # If needed to fit, so fit model
        if hasattr(self.__model, 'need_fit') and self.__model.need_fit:
            features = self.__feature_extraction_fit(inputs)
        else:
            datas = inputs.get('data')
            labels = inputs.get('label')
            unique_labels = unique(labels)
            features = Classifier.__feature_extraction_simple(self.__model, datas, unique_labels)

        return array(features)

    def __feature_extraction_fit(self, inputs):
        datas = inputs.get('data')
        labels = inputs.get('label')
        unique_labels = unique(labels)
        groups = inputs.get('group')
        folds = inputs.get('fold')
        samples = len(datas)
        features = None

        for fold, test in enumerate(unique(folds)):
            # Clone model
            model = deepcopy(self.__model)
            train_indices = np.where(np.isin(folds, test, invert=True))[0]
            test_indices = np.where(np.isin(folds, test))[0]

            # Check that current fold respect labels
            if not self.__check_labels(labels, labels[train_indices]):
                warnings.warn('Invalid fold, missing labels for fold {fold}'.format(fold=fold + 1))
                continue

            # Remove unwanted labels that contains -1 for semisupervised
            if not hasattr(self.__model, 'is_semi_supervised') or not self.__model.is_semi_supervised:
                train_indices = train_indices[labels[train_indices] != -1]

            # Now fit, but find first hyper parameters
            grid_search = GridSearchCV(estimator=self.__model, param_grid=self.__params, cv=self.__inner_cv,
                                       n_jobs=self.n_jobs, refit=False, scoring=self.__scoring, verbose=1, iid=False)
            grid_search.fit(datas[train_indices], y=labels[train_indices], groups=groups[train_indices], **self.__fit_params)
            best_params = grid_search.best_params_

            # Fit the model, with the bests parameters
            model.set_params(**best_params)
            if isinstance(model, KerasBatchClassifier):
                model.fit(datas[train_indices], y=labels[train_indices], callbacks=self.__callbacks,
                          X_validation=datas[test_indices], y_validation=labels[test_indices], **self.__fit_params)
            else:
                model.fit(datas[train_indices], y=labels[train_indices])

            test_features = Classifier.__feature_extraction_simple(model, datas[test_indices], unique_labels)

            if features is None:
                features = np.zeros(shape=(samples,) + test_features.shape[1:])

            features[test_indices] = test_features

        return features

    def __format_params(self):  # Here we proceed as multiple combination
        for param in self.__params:
            for key, value in param.items():
                if not isinstance(value, list):
                    param[key] = [value]

    @staticmethod
    def __check_labels(labels, labels_fold=None):
        if labels_fold is None:
            return len(unique(labels)) > 1
        return len(unique(labels)) > 1 and array_equal(unique(labels), unique(labels_fold))

    @staticmethod
    def __feature_extraction_simple(model, datas, unique_labels):
        # Now transform data
        if hasattr(model, 'transform'):
            features = model.transform(datas)
        else:
            features = model.predict_proba(datas)
            if len(model.classes_) != 0:
                features = Classifier.predict_proba_ordered(features, model.classes_, unique_labels)
        return features

    @staticmethod
    def __number_of_features(model):
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]

        if isinstance(model, SVC):
            if model.kernel == 'rbf':
                return model.support_vectors_.shape[1]
            else:
                return model.coef_.shape[1]

        return 0

    @staticmethod
    def __predict_classes(probabilities):
        if probabilities.shape[-1] > 1:
            return probabilities.argmax(axis=-1)
        else:
            return (probabilities > 0.5).astype('int32')

    @staticmethod
    def predict_proba_ordered(probabilities, classes_, all_classes):
        """
        probs: list of probabilities, output of predict_proba
        classes_: clf.classes_
        all_classes: all possible classes (superset of classes_)
        """
        all_classes = all_classes[all_classes >= 0]
        proba_ordered = np.zeros((probabilities.shape[0], all_classes.size), dtype=np.float)
        sorter = np.argsort(all_classes)  # http://stackoverflow.com/a/32191125/395857
        idx = sorter[np.searchsorted(all_classes, classes_, sorter=sorter)]
        proba_ordered[:, idx] = probabilities
        return proba_ordered

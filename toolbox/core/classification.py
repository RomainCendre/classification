import glob
import warnings
from os.path import join
from copy import deepcopy

import h5py
import numpy as np
from numpy import unique, array_equal, save, load, array
from sklearn.model_selection import GridSearchCV
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

    def __init__(self, inner_cv, n_jobs=-1, callbacks=[], scoring=None, is_semi=False):
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
        self.is_semi_supervised = is_semi
        self.n_jobs = n_jobs
        self.patients_folds = None

    def split_patients(self, inputs, split_rule):
        # Groups
        groups = list(inputs.get_groups())
        unique_groups = list(unique(groups))
        indices = [groups.index(element) for element in unique_groups]
        # Labels
        groups_labels = list(inputs.get_groups_labels())
        unique_groups_labels = [groups_labels[element] for element in indices]
        # Make folds
        self.patients_folds = list(enumerate(split_rule.split(X=unique_groups, y=unique_groups_labels)))

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
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()
        reference = inputs.get_reference()

        # Check valid labels, at least several classes
        if not self.__check_labels(labels):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Encode labels to go from string to int
        results = []
        for fold, (train, test) in self.patients_folds:

            test_indices = np.where(np.isin(groups, test))[0]
            train_indices = np.where(np.isin(groups, train))[0]
            if not self.is_semi_supervised:
                train_indices = train_indices[labels[train_indices] != -1]

            # Check that current fold respect labels
            if not self.__check_labels(labels, labels[train_indices]):
                warnings.warn('Invalid fold, missing labels for fold {fold}'.format(fold=fold + 1))
                continue

            print('Fold : {fold}'.format(fold=fold + 1))

            # Clone model
            model = deepcopy(self.__model)

            # Estimate best combination
            grid_search = GridSearchCV(estimator=model, param_grid=self.__params, cv=self.__inner_cv,
                                       n_jobs=self.n_jobs, scoring=self.__scoring, verbose=1, iid=False)
            grid_search.fit(datas[train_indices], y=labels[train_indices], **self.__fit_params)
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
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()

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
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        unique_labels = inputs.get_unique_labels()
        groups = inputs.get_groups()
        references = inputs.get_reference()

        # Location of folder followed by prefix
        if hasattr(self.__model, 'name'):
            prefix = self.__model.name
        else:
            prefix = type(self.__model).__name__

        if inputs.temporary is not None:
            # Create HDF5 file
            file_path = '{folder_prefix}.hdf5'.format(folder_prefix=join(inputs.temporary, prefix))
            features_file = h5py.File(file_path, 'a')
            # Now process HDF5 file
            try:
                features = []
                # Check if already extracted
                if not set(references).issubset(features_file.keys()):
                    features = self.__feature_extraction(prefix, datas, labels, unique_labels, groups)

                    # Now save features as files
                    print('Writing data at {file}'.format(file=file_path))
                    for feature, reference in zip(features, references):
                        if reference not in features_file.keys():
                            features_file.create_dataset(reference, data=feature)
                else:
                    print('Loading data at {file}'.format(file=file_path))
                    for reference in references:
                        features.append(features_file[reference][()])
                    features = array(features)
            finally:
                features_file.close()
        else:
            features = self.__feature_extraction(prefix, datas, labels, unique_labels, groups)

        # Update input
        inputs.update(prefix, features, references)

    def __feature_extraction(self, prefix, datas, labels, nb_labels, groups):
        # Now browse data
        print('Extraction features with {prefix}'.format(prefix=prefix))

        # If needed to fit, so fit model
        if hasattr(self.__model, 'need_fit') and self.__model.need_fit:
            features = self.__feature_extraction_fit(datas, labels, nb_labels, groups)
        else:
            features = Classifier.__feature_extraction_simple(self.__model, datas, nb_labels)

        return array(features)

    def __feature_extraction_fit(self, datas, labels, nb_labels, groups):
        features = len(labels) * [None]
        for fold, (train, test) in self.patients_folds:
            # Clone model
            model = deepcopy(self.__model)
            test_indices = np.where(np.isin(groups, test))[0]
            train_indices = np.where(np.isin(groups, train))[0]

            if not self.is_semi_supervised:
                train_indices = train_indices[labels[train_indices] != -1]

            # Now fit, but find first hyper parameters
            grid_search = GridSearchCV(estimator=self.__model, param_grid=self.__params, cv=self.__inner_cv,
                                       n_jobs=self.n_jobs, refit=False, scoring=self.__scoring, verbose=1, iid=False)
            grid_search.fit(datas[train_indices], y=labels[train_indices], **self.__fit_params)
            best_params = grid_search.best_params_

            # Fit the model, with the bests parameters
            model.set_params(**best_params)
            if isinstance(model, KerasBatchClassifier):
                model.fit(datas[train_indices], y=labels[train_indices], callbacks=self.__callbacks,
                          X_validation=datas[test_indices], y_validation=labels[test_indices], **self.__fit_params)
            else:
                model.fit(datas[train_indices], y=labels[train_indices])

            test_features = Classifier.__feature_extraction_simple(model, datas[test_indices], nb_labels)

            for index, feature in enumerate(test_features):
                features[test_indices[index]] = feature

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
    def __feature_extraction_simple(model, datas, nb_labels):
        # Now transform data
        if hasattr(model, 'transform'):
            features = model.transform(datas)
        else:
            features = model.predict_proba(datas)
            if len(model.classes_) != 0:
                features = Classifier.predict_proba_ordered(features, model.classes_, nb_labels)
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

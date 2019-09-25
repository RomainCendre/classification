import warnings

import h5py
from copy import deepcopy
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from toolbox.core.models import KerasBatchClassifier


class Folds:

    @staticmethod
    def build_folds(dataframe, tags, split=5):
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        # Inputs
        data = dataframe[tags['datum']]
        labels = dataframe[tags['label']]

        # Rule to create folds
        split_rule = KFold(n_splits=split)

        # Make folds
        folds = np.zeros(len(labels), dtype=int)
        current_folds = list(split_rule.split(X=data, y=labels))
        for index, fold in enumerate(current_folds):
            folds[fold[1]] = index
        dataframe['Fold'] = folds.tolist()  # Add tests to folds
        return dataframe

    @staticmethod
    def build_group_folds(dataframe, tags, split=5):
        mandatory = ['datum', 'label', 'group']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')
        # Inputs
        data = dataframe[tags['datum']]
        labels = dataframe[tags['label']]
        groups = dataframe[tags['group']]

        # Rule to create folds
        split_rule = GroupKFold(n_splits=split)

        # Make folds
        folds = np.zeros(len(labels), dtype=int)
        current_folds = list(split_rule.split(X=data, y=labels, groups=groups))
        for index, fold in enumerate(current_folds):
            folds[fold[1]] = index
        dataframe['Fold'] = folds.tolist()  # Add
        return dataframe


class Classification:

    @staticmethod
    def evaluate(dataframe, tags, out, model, mask=None):
        # Fold needed for evaluation
        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)

        # Check valid labels, at least several classes
        if not Classification.__check_labels(dataframe[mask], {'label': tags['label']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Encode labels to go from string to int
        folds = dataframe['Fold']

        for fold, test in enumerate(np.unique(folds)):

            test_mask = folds == fold
            print('Fold : {fold}'.format(fold=fold + 1))

            # Check that current fold respect labels
            if not Classification.__check_labels(dataframe[mask], {'label': tags['label']}, ~test_mask):
                warnings.warn(f'Invalid fold, missing labels for fold {fold+1}')
                continue

            # Clone model
            fitted_model = deepcopy(model)
            Classification.fit(dataframe, tags, fitted_model, ~test_mask)
            # Predict
            Classification.predict(dataframe[test_mask], {'datum': tags['datum']}, out, fitted_model)
            Classification.predict_proba(dataframe[test_mask], {'datum': tags['datum']}, out, fitted_model)

        return dataframe

    @staticmethod
    def fit(dataframe, tags, model, mask=None):
        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)

        # Check valid labels, at least several classes
        if not Classification.__check_labels(dataframe[mask], {'label': tags['label']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        labels = np.array(dataframe.loc[mask, tags['label']].to_list())
        model.fit(data, y=labels)
        return model

    @staticmethod
    def fit_and_transform(dataframe, tags, out, model, mask=None):
        if mask is None:
            mask = [True] * len(dataframe.index)
        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)

        folds = dataframe['Fold'].unique()
        for fold in folds:
            mask = dataframe['Fold'] == fold
            fitted_model = Classification.fit(dataframe[mask], tags, deepcopy(model))
            dataframe.loc[mask, out] = Classification.transform(dataframe[mask], tags, out, fitted_model)[out]
        return dataframe

    @staticmethod
    def predict(dataframe, tags, out, model, mask=None):
        # Check predict_proba field
        if not hasattr(model, 'predict'):
            warnings.warn('No method predict found.')
            return

        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)

        # Set de predict values
        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        predictions = model.predict(data)
        dataframe.loc[mask, f'{out}_Predictions'] = pd.Series([f for f in predictions])
        return dataframe

    @staticmethod
    def predict_proba(dataframe, tags, out, model, mask=None):
        # Check predict_proba field
        if not hasattr(model, 'predict_proba'):
            warnings.warn('No method predict_proba found.')
            return

        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)

        # Set de predict probas values
        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        probabilities = model.predict_proba(data)
        dataframe.loc[mask, f'{out}_Probabilities'] = pd.Series([f for f in probabilities])
        return dataframe

    @staticmethod
    def transform(dataframe, tags, model, out, mask=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)

        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        features = model.transform(data)
        dataframe.loc[mask, out] = pd.Series([f for f in features])
        return dataframe

    @staticmethod
    def __check_labels(dataframe, tags, mask_sub=None):
        mandatory = ['label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        labels = dataframe[tags['label']]
        if mask_sub is None:
            return len(np.unique(labels)) > 1
        return len(np.unique(labels)) > 1 and np.array_equal(np.unique(labels),
                                                             np.unique(dataframe.loc[mask_sub, tags['label']]))

    @staticmethod
    def __number_of_features(model):
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]

        if isinstance(model, SVC):
            if model.kernel == 'rbf':
                return model.support_vectors_.shape[1]
            else:
                return model.coef_.shape[1]
        elif isinstance(model, DecisionTreeClassifier):
            return model.n_features_
        return 0

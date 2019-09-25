import warnings

import h5py
from copy import deepcopy
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


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


class IO:

    @staticmethod
    def load(input_file, key):
        return pd.read_hdf(input_file, key)

    @staticmethod
    def save(dataframe, save, key):
        dataframe.to_hdf(save, key)


class Tools:

    @staticmethod
    def evaluate(dataframe, tags, out, model, mask=None):
        # Fold needed for evaluation
        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)
        mask = pd.Series(mask)

        # Check valid labels, at least several classes
        if not Tools.__check_labels(dataframe[mask], {'label': tags['label']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Out fields
        out_preds = f'{out}_Predictions'
        out_probas = f'{out}_Probabilities'
        out_params = f'{out}_Parameters'

        # Browse folds
        folds = dataframe.loc[mask, 'Fold']
        for fold, test in enumerate(np.unique(folds)):
            # Create mask
            test_mask = folds == fold
            print('Fold : {fold}'.format(fold=fold + 1))

            # Check that current fold respect labels
            if not Tools.__check_labels(dataframe[mask], {'label': tags['label']}, ~test_mask):
                warnings.warn(f'Invalid fold, missing labels for fold {fold+1}')
                continue

            # Clone model
            fitted_model = Tools.fit(dataframe[mask], tags, deepcopy(model), ~test_mask)

            # Predict
            dataframe.loc[mask, out_preds] = \
                Tools.predict(dataframe[mask], {'datum': tags['datum']}, out_preds, fitted_model, test_mask)[out_preds]
            dataframe.loc[mask, out_probas] = \
                Tools.predict_proba(dataframe[mask], {'datum': tags['datum']}, out_probas, fitted_model, test_mask)[out_probas]
            dataframe.loc[mask, out_params] = \
                Tools.number_of_features(dataframe[mask], fitted_model, out_params, test_mask)[out_params]

        return dataframe

    @staticmethod
    def fit(dataframe, tags, model, mask=None):
        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)

        # Check valid labels, at least several classes
        if not Tools.__check_labels(dataframe[mask], {'label': tags['label']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        labels = np.array(dataframe.loc[mask, tags['label']].to_list())
        model.fit(data, y=labels)
        return model

    @staticmethod
    def fit_transform(dataframe, tags, model, out, mask=None):
        # Fold needed for evaluation
        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)
        mask = pd.Series(mask)

        # Clone model
        fitted_model = Tools.fit(dataframe[mask], tags, deepcopy(model), mask)

        # Transform
        dataframe.loc[mask, out] = Tools.transform(dataframe[mask], {'datum': tags['datum']}, fitted_model, out)[out]

        return dataframe

    @staticmethod
    def number_of_features(dataframe, model, out, mask=None):
        dataframe.loc[mask, out] = Tools.__number_of_features(model)
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
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)
        mask = pd.Series(mask)

        # Set de predict values
        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        predictions = model.predict(data)
        dataframe.loc[mask, out] = pd.Series([p for p in predictions], index=mask[mask==True].index)
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
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)
        mask = pd.Series(mask)

        # Set de predict probas values
        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        probabilities = model.predict_proba(data)
        dataframe.loc[mask, out] = pd.Series([p for p in probabilities], index=mask[mask==True].index)
        return dataframe

    @staticmethod
    def transform(dataframe, tags, model, out, mask=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)
        mask = pd.Series(mask)

        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        features = model.transform(data)
        dataframe.loc[mask, out] = pd.Series([f for f in features], index=mask[mask==True].index)
        return dataframe

    @staticmethod
    def __check_labels(dataframe, tags, mask_sub=None):
        mandatory = ['label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

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

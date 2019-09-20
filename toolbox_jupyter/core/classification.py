import h5py
from copy import deepcopy
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from toolbox.core.models import KerasBatchClassifier
from toolbox.core.structures import Outputs


class Tools:

    @staticmethod
    def build_folds(dataframe, tags, split=5):
        mandatory = ['data', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception('Not a dict or missing tag: data, label.')
        # Inputs
        datas = dataframe[tags['data']]
        labels = dataframe[tags['label']]

        # Rule to create folds
        split_rule = KFold(n_splits=split)

        # Make folds
        folds = np.zeros(len(labels), dtype=int)
        current_folds = list(split_rule.split(X=datas, y=labels))
        for index, fold in enumerate(current_folds):
            folds[fold[1]] = index
        dataframe['Fold'] = folds.tolist()  # Add tests to folds
        return dataframe

    @staticmethod
    def build_patients_folds(dataframe, tags, split=5):
        mandatory = ['data', 'label', 'group']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception('Not a dict or missing tag: data, label, group.')
        # Inputs
        datas = dataframe[tags['data']]
        labels = dataframe[tags['label']]
        groups = dataframe[tags['group']]

        # Rule to create folds
        split_rule = GroupKFold(n_splits=split)

        # Make folds
        folds = np.zeros(len(labels), dtype=int)
        current_folds = list(split_rule.split(X=datas, y=labels, groups=groups))
        for index, fold in enumerate(current_folds):
            folds[fold[1]] = index
        dataframe['Fold'] = folds.tolist()  # Add
        return dataframe

    @staticmethod
    def fit(dataframe, tags, model):
        mandatory = ['data', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception('Not a dict or missing tag: data, label.')
        model.fit(dataframe[tags['data']], y=dataframe[tags['label']])
        return model

    @staticmethod
    def fit_and_transform(dataframe, tags, out, model):
        mandatory = ['data']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception('Not a dict or missing tag: data.')

        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        folds = dataframe['Fold'].unique()
        for fold in folds:
            mask = dataframe['Fold'] == fold
            fit_model = Tools.fit(dataframe[mask], tags, deepcopy(model))
            dataframe.loc[mask, out] = Tools.transform(dataframe[mask], tags, out, fit_model)[out]
        return dataframe

    @staticmethod
    def evaluate(dataframe, tags, model):
        mandatory = ['data', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception('Not a dict or missing tag: data, label.')


    @staticmethod
    def transform(dataframe, tags, out, model):
        mandatory = ['data']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception('Not a dict or missing tag: data.')
        features = model.transform(dataframe[tags['data']].to_numpy())
        dataframe[out] = [f for f in features]
        return dataframe

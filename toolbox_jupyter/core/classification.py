import h5py
from numpy import array
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
        current_folds = list(split_rule.split(X=datas, y=labels))
        for index, fold in enumerate(current_folds):
            dataframe['Fold'] = index  # Add tests to folds
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
        current_folds = list(split_rule.split(X=datas, y=labels, groups=groups))
        for index, fold in enumerate(current_folds):
            dataframe['Fold'] = index  # Add tests to folds
        return dataframe

    @staticmethod
    def fit_and_transform(dataframe, tags, out, extractor):
        mandatory = ['data', 'label', 'fold']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception('Not a dict or missing tag: data, label, group.')
        dataframe[out] = dataframe.apply(lambda x: extractor.transform(tags['data']), axis=1)
        return dataframe

    @staticmethod
    def transform(dataframe, tags, out, extractor):
        mandatory = ['data', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception('Not a dict or missing tag: data, label, group.')
        features = extractor.transform(dataframe[tags['data']].to_numpy())
        dataframe[out] = [f for f in features]
        return dataframe

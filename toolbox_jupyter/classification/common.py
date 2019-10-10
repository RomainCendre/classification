import warnings
from copy import deepcopy
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, GroupKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


class Data:

    @staticmethod
    def collapse(self, filters, on, filters_collapse, on_collapse, data_tag=None):
        pd.options.mode.chained_assignment = None
        if data_tag is None:
            data_tag = self.tags['datum']

        # Filters
        filters.update(self.filters)
        filters_collapse.update(self.filters)

        # Query
        query = self.to_query(filters)
        query_collapse = self.to_query(filters_collapse)

        # Data
        data = self.data.query(query)
        data_collapse = self.data.query(query_collapse)

        # Collapse data
        rows = []
        for name, group in data.groupby(on):
            # Get features by group
            raw_row = group.iloc[0]
            group_collapse = data_collapse[data_collapse[on_collapse] == name]
            raw_row[data_tag] = np.array(group_collapse[data_tag].tolist())
            rows.append(raw_row)

        # Now set new data
        input = copy(self)
        input.data = pd.DataFrame(rows)
        return input


class Folds:

    @staticmethod
    def build_folds(dataframe, tags, split=5):
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')

        # Inputs
        data = dataframe[tags['datum']]
        labels = dataframe[tags['label_encode']]

        # Rule to create folds
        split_rule = StratifiedKFold(n_splits=split)

        # Make folds
        folds = np.zeros(len(labels), dtype=int)
        current_folds = list(split_rule.split(X=data, y=labels))
        for index, fold in enumerate(current_folds, 1):
            folds[fold[1]] = index
        dataframe['Fold'] = folds.tolist()  # Add tests to folds
        return dataframe

    @staticmethod
    def build_group_folds(dataframe, tags, split=5):
        mandatory = ['datum', 'label_encode', 'group']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')
        # Inputs
        data = dataframe[tags['datum']]
        labels = dataframe[tags['label_encode']]
        groups = dataframe[tags['group']]

        # Rule to create folds
        split_rule = GroupKFold(n_splits=split)

        # Make folds
        folds = np.zeros(len(labels), dtype=int)
        current_folds = list(split_rule.split(X=data, y=labels, groups=groups))
        for index, fold in enumerate(current_folds, 1):
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
    def evaluate(dataframe, tags, model, out, mask=None, grid=None, distribution=None):
        # Fold needed for evaluation
        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)
        mask = pd.Series(mask)

        # Check valid labels, at least several classes
        if not Tools.__check_labels(dataframe[mask], {'label_encode': tags['label_encode']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Out fields
        out_preds = f'{out}_Predictions'
        out_probas = f'{out}_Probabilities'
        out_features = f'{out}_Features'
        out_params = f'{out}_Parameters'

        # Browse folds
        folds = dataframe.loc[mask, 'Fold']
        for fold in np.unique(folds):
            # Out fields
            fold_preds = f'{out_preds}_{fold}'
            fold_probas = f'{out_probas}_{fold}'
            fold_features = f'{out_features}_{fold}'
            fold_params = f'{out_params}_{fold}'

            # Create mask
            test_mask = folds == fold
            print(f'Fold {fold} performed...', end='\r')

            # Check that current fold respect labels
            if not Tools.__check_labels(dataframe[mask], {'label_encode': tags['label_encode']}, ~test_mask):
                warnings.warn(f'Invalid fold, missing labels for fold {fold}')
                continue

            # Clone model
            fitted_model = Tools.fit(dataframe[mask], tags, deepcopy(model), mask=~test_mask,
                                     grid=grid, distribution=distribution)

            # Predict
            dataframe[fold_preds] = Tools.predict(dataframe[mask], {'datum': tags['datum']}, fold_preds, fitted_model)[fold_preds]
            dataframe[fold_probas] = Tools.predict_proba(dataframe[mask], {'datum': tags['datum']}, fold_probas, fitted_model)[fold_probas]
            dataframe[fold_features] = Tools.number_of_features(dataframe[mask], fitted_model, fold_params)[fold_params]
            dataframe[fold_params] = [fitted_model.best_params] * len(dataframe)

        print(f'Evaluation achieved!', end='\r')
        return dataframe

    @staticmethod
    def fit(dataframe, tags, model, mask=None, grid=None, distribution=None):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)

        # Check valid labels, at least several classes
        if not Tools.__check_labels(dataframe[mask], {'label_encode': tags['label_encode']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        data = np.array(dataframe.loc[mask, tags['datum']].to_list())
        labels = np.array(dataframe.loc[mask, tags['label_encode']].to_list())

        if grid is not None:
            grid_search = GridSearchCV(model, param_grid=grid, cv=2, iid=False)
            grid_search.fit(data, y=labels)
            model = grid_search.best_estimator_
            model.best_params = grid_search.best_params_
            return model
        elif distribution is not None:
            random_search = RandomizedSearchCV(model, param_distributions=distribution, cv=2, iid=False)
            random_search.fit(data, y=labels)
            model = random_search.best_estimator_
            model.best_params = random_search.best_params_
            return model
        else:
            model.fit(data, y=labels)
            model.best_params = {}
            return model

    @staticmethod
    def fit_transform(dataframe, tags, model, out, mask=None, grid=None, distribution=None):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)
        mask = pd.Series(mask)

        # Clone model
        fitted_model = Tools.fit(dataframe[mask], tags, deepcopy(model), mask=mask, grid=None, distribution=None)

        # Transform
        dataframe.loc[mask, out] = Tools.transform(dataframe[mask], {'datum': tags['datum']}, fitted_model, out)[out]

        return dataframe

    @staticmethod
    def misclassified(inputs, tags):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode', 'result']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Prediction tag
        tag_pred = f'{tags["result"]}_Predictions'

        # Mask
        mask = (inputs[tags['label_encode']] == inputs[tag_pred])
        inputs = inputs[mask]
        data = {'datum': inputs[tags['datum']],
                'labels': inputs[tags['label_encode']],
                'predictions': inputs[tag_pred]}
        return pd.DataFrame(data)

    @staticmethod
    def number_of_features(dataframe, model, out, mask=None):
        # Mask creation (see pandas view / copy mechanism)
        if mask is None:
            mask = [True] * len(dataframe.index)
        mask = pd.Series(mask)
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
        mandatory = ['label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        labels = dataframe[tags['label_encode']]
        if mask_sub is None:
            return len(np.unique(labels)) > 1
        return len(np.unique(labels)) > 1 and np.array_equal(np.unique(labels),
                                                             np.unique(dataframe.loc[mask_sub, tags['label_encode']]))

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

import pickle
import warnings
from copy import deepcopy
import numpy as np
from pandas.errors import PerformanceWarning
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from toolbox.models.models import KerasBatchClassifier
warnings.filterwarnings("ignore", category=PerformanceWarning)
pd.options.mode.chained_assignment = None


class Data:

    @staticmethod
    def build_bags(dataframe, mask_1, on_tag_1, mask_2, on_tag_2, datum_tag, out_tag=None):
        # Check elements
        if on_tag_1 not in dataframe:
            raise Exception(f'{on_tag_1} not in dataframe.')

        if on_tag_2 not in dataframe:
            raise Exception(f'{on_tag_2} not in dataframe.')

        # Init target if doesn't set
        if out_tag is None:
            out_tag = datum_tag

        # Collapse data
        if out_tag not in dataframe:
            dataframe[out_tag] = np.nan

        # Mask dataframe
        sub_1 = dataframe[mask_1]
        sub_2 = dataframe[mask_2]

        def bag_maker(row):
            group_collapse = sub_2[sub_2[on_tag_2] == row[on_tag_1]]
            return np.array(group_collapse[datum_tag].tolist())

        dataframe.loc[mask_1, out_tag] = sub_1.apply(bag_maker, axis=1)


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

    FEATURES = 'Features'
    PARAMETERS = 'Parameters'
    PREDICTION = 'Prediction'
    PROBABILITY = 'Probability'

    @staticmethod
    def evaluate(dataframe, tags, model, out, mask=None, grid=None, distribution=None, unbalanced=None, cpu=-1, path=None):
        # Fold needed for evaluation
        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Check valid labels, at least several classes
        if not Tools.__check_labels(dataframe, {'label_encode': tags['label_encode']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Out fields
        out_preds = f'{out}_{Tools.PREDICTION}'
        out_probas = f'{out}_{Tools.PROBABILITY}'
        out_features = f'{out}_{Tools.FEATURES}'
        out_params = f'{out}_{Tools.PARAMETERS}'

        # Create missing fields
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        folds = sub['Fold']
        for fold in np.unique(folds):
            # Out fields
            fold_preds = f'{out_preds}_{fold}'
            fold_probas = f'{out_probas}_{fold}'
            fold_features = f'{out_features}_{fold}'
            fold_params = f'{out_params}_{fold}'
            if fold_preds not in dataframe:
                dataframe[fold_preds] = np.nan
            if fold_probas not in dataframe:
                dataframe[fold_probas] = np.nan
            if fold_features not in dataframe:
                dataframe[fold_features] = np.nan
            if fold_params not in dataframe:
                dataframe[fold_params] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Browse folds
        folds = sub['Fold']
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
            if not Tools.__check_labels(sub[~test_mask], {'label_encode': tags['label_encode']}):
                warnings.warn(f'Invalid fold, missing labels for fold {fold}')
                continue

            # Clone model
            fitted_model = Tools.fit(sub[~test_mask], tags, deepcopy(model),
                                     grid=grid, distribution=distribution, unbalanced=unbalanced, cpu=cpu)

            # Save if needed
            if path is not None:
                file = path / f'{out}_{fold}.hdf5'
                if isinstance(fitted_model, KerasBatchClassifier):
                    fitted_model.save(str(file))
                else:
                    pickle.dumps(fitted_model, str(file))

            # Predict
            Tools.predict(sub, {'datum': tags['datum']}, fold_preds, fitted_model)
            Tools.predict_proba(sub, {'datum': tags['datum']}, fold_probas, fitted_model)
            Tools.number_of_features(sub, fitted_model, fold_features)
            Tools.__best_params(sub, fitted_model, fold_params)

        if mask is not None:
            dataframe[mask] = sub
        print(f'Evaluation achieved!', end='\r')

    @staticmethod
    def fit(dataframe, tags, model, grid=None, distribution=None, unbalanced=None, cpu=-1):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Check valid labels, at least several classes
        if not Tools.__check_labels(dataframe, {'label_encode': tags['label_encode']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        data = np.array(dataframe[tags['datum']].to_list())
        labels = np.array(dataframe[tags['label_encode']].to_list())

        if unbalanced is not None:
            if callable(getattr(unbalanced, 'fit_resample', None)):
                data, labels = unbalanced.fit_resample(data, labels)
            else:
                Exception(f'Expected valid unbalanced property {unbalanced}.')

        # Check for supervised stuff if in supervised mode
        if not hasattr(model, 'is_semi'):
            labelled = labels != -1
            data = data[labelled]
            labels = labels[labelled]

        if grid is not None:
            grid_search = GridSearchCV(model, param_grid=grid, cv=2, iid=False, n_jobs=cpu)
            grid_search.fit(data, y=labels)
            model = grid_search.best_estimator_
            model.best_params = grid_search.best_params_
            return model
        elif distribution is not None:
            random_search = RandomizedSearchCV(model, param_distributions=distribution, cv=2, iid=False, n_jobs=cpu)
            random_search.fit(data, y=labels)
            model = random_search.best_estimator_
            model.best_params = random_search.best_params_
            return model
        else:
            model.fit(data, y=labels)
            model.best_params = {}
            return model

    @staticmethod
    def fit_transform(dataframe, tags, model, out, mask=None, grid=None, distribution=None, unbalanced=None, cpu=-1):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Clone model
        fitted_model = Tools.fit(sub, tags, deepcopy(model), grid=grid, distribution=distribution, unbalanced=unbalanced, cpu=cpu)

        # Transform
        sub[out] = Tools.transform(sub, {'datum': tags['datum']}, fitted_model, out)[out]

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def fit_predict(dataframe, tags, model, out, mask=None, grid=None, distribution=None, unbalanced=None, cpu=-1):
        # Fold needed for evaluation
        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Check valid labels, at least several classes
        if not Tools.__check_labels(dataframe, {'label_encode': tags['label_encode']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Out fields
        out_preds = f'{out}_{Tools.PREDICTION}'
        out_probas = f'{out}_{Tools.PROBABILITY}'

        # Manage columns
        if out_preds not in dataframe:
            dataframe[out_preds] = np.nan
        if out_probas not in dataframe:
            dataframe[out_probas] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Browse folds
        folds = sub['Fold']
        for fold in np.unique(folds):
            # Create mask
            test_mask = folds == fold

            # Check that current fold respect labels
            if not Tools.__check_labels(sub[~test_mask], {'label_encode': tags['label_encode']}):
                warnings.warn(f'Invalid fold, missing labels for fold {fold}')
                continue

            # Clone model
            fitted_model = Tools.fit(sub[~test_mask], tags, model=deepcopy(model), grid=grid, distribution=distribution,
                                     unbalanced=unbalanced, cpu=cpu)

            # Fill new data
            Tools.predict(sub, {'datum': tags['datum']}, out_preds, fitted_model, mask=test_mask)
            Tools.predict_proba(sub, {'datum': tags['datum']}, out_probas, fitted_model, mask=test_mask)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def number_of_features(dataframe, model, out, mask=None):
        # Create column if doesnt exist
        if out not in dataframe:
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        sub[out] = Tools.__number_of_features(model)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def predict(dataframe, tags, out, model, mask=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Create column if doesnt exist
        if out not in dataframe:
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Check predict_proba field
        if not hasattr(model, 'predict'):
            warnings.warn('No method predict found.')
            sub[out] = np.nan
            if mask is not None:
                dataframe[mask] = sub
            return

        # Set de predict values
        data = np.array(sub[tags['datum']].to_list())
        predictions = model.predict(data)
        sub[out] = pd.Series([p for p in predictions], index=sub.index)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def predict_proba(dataframe, tags, out, model, mask=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Create column if doesnt exist
        if out not in dataframe:
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Check predict_proba field
        if not hasattr(model, 'predict_proba'):
            warnings.warn('No method predict_proba found.')
            sub[out] = np.nan
            if mask is not None:
                dataframe[mask] = sub
            return

        # Set de predict probas values
        data = np.array(sub[tags['datum']].to_list())
        probabilities = model.predict_proba(data)
        sub[out] = pd.Series([p for p in probabilities], index=sub.index)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def transform(dataframe, tags, model, out, mask=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Create column if doesnt exist
        if out not in dataframe:
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Check predict_proba field
        if not hasattr(model, 'transform'):
            warnings.warn('No method predict_proba found.')
            sub[out] = np.nan
            if mask is not None:
                dataframe[mask] = sub
            return

        data = np.array(sub[tags['datum']].to_list())
        features = model.transform(data)
        sub[out] = pd.Series([f for f in features], index=sub.index)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def __best_params(dataframe, model, out, mask=None):
        # Create column if doesnt exist
        if out not in dataframe:
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        sub[out] = [model.best_params] * len(sub)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def __check_labels(dataframe, tags):
        mandatory = ['label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        labels = dataframe[tags['label_encode']]
        return len(np.unique(labels)) > 1

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

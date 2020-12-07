import dill as pickle
import warnings
from copy import deepcopy
import numpy as np
from pandas.errors import PerformanceWarning
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from toolbox.models.models import KerasBatchClassifier, MultimodalClassifier, CustomCalibrationCV

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

    @staticmethod
    def export_group_folds(dataframe, tags):
        mandatory = ['group']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')
        data = dataframe[[tags['group'], 'Fold']]
        data = data.drop_duplicates()
        data = data.reset_index(drop=True)
        return data

    @staticmethod
    def restore_group_folds(dataframe, folds, tags):
        mandatory = ['group']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')
        dataframe['Fold'] = None
        dataframe.set_index(tags['group'], inplace=True)
        dataframe.update(folds.set_index(tags['group']))
        dataframe.reset_index(inplace=True)


class IO:

    @staticmethod
    def load(input_file):
        return pd.read_pickle(input_file)

    @staticmethod
    def save(dataframe, save):
        dataframe.to_pickle(save)


class Tools:

    FEATURES = 'Features'
    PARAMETERS = 'Parameters'
    PREDICTION = 'Prediction'
    SCORE = 'Score'
    STEPS = 'Steps'
    VAL_RATIO = 2

    @staticmethod
    def evaluate(dataframe, tags, model, out, mask=None, grid=None, distribution=None, cpu=-1, folds=None, calibrate=None, instance=None):
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
        out_features = f'{out}_{Tools.FEATURES}'
        out_predict = f'{out}_{Tools.PREDICTION}'
        out_score = f'{out}_{Tools.SCORE}'
        out_params = f'{out}_{Tools.PARAMETERS}'
        out_steps = f'{out}_{Tools.STEPS}'
        if out_predict not in dataframe:
            dataframe[out_predict] = np.nan
        if out_score not in dataframe:
            dataframe[out_score] = np.nan
        if out_features not in dataframe:
            dataframe[out_features] = np.nan
        if out_params not in dataframe:
            dataframe[out_params] = np.nan
        if out_steps not in dataframe:
            dataframe[out_steps] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        dump = pickle.dumps(model)

        # Browse folds
        reference_folds = sub['Fold']
        if folds is None:
            folds = Tools.__default_folds(list(np.unique(reference_folds)))

        for index, current in enumerate(folds):
            # Create fit mask
            fit_mask = reference_folds.isin(current[0])

            print(f'Fold {index} performed...', end='\r')

            # Check that current fold respect labels
            if not Tools.__check_labels(sub[fit_mask], {'label_encode': tags['label_encode']}):
                warnings.warn(f'Invalid fold, missing labels for fold {index}')
                continue

            model = pickle.loads(dump)

            # Clone model
            model = Tools.fit(sub[fit_mask], tags, model, grid=grid, distribution=distribution, cpu=cpu)

            # Make evaluation of calibration if needed
            if calibrate:
                calibrate_mask = reference_folds.isin(current[1]) # Prepare mask for calibration
                model_calibration = CustomCalibrationCV(model, cv=Tools.VAL_RATIO, method=calibrate)
                model = Tools.fit(sub[calibrate_mask], tags, model_calibration, cpu=cpu)

            # Prepare the prediction mask
            predict_mask = reference_folds.isin(current[-1])

            # Remember features and params
            Tools.number_of_features(sub, model, out_features, mask=predict_mask)
            Tools.__best_params(sub, model, out_params, mask=predict_mask)

            # Predict
            if hasattr(model, 'predict'):
                Tools.predict(sub, {'datum': tags['datum']}, model, out_predict, mask=predict_mask, instance=instance)
            if hasattr(model, 'predict_steps'):
                Tools.predict_steps(sub, {'datum': tags['datum']}, model, out_steps, mask=predict_mask, instance=instance)
            if hasattr(model, 'score'):
                Tools.score(sub, {'datum': tags['datum']}, model, out_score, mask=predict_mask, instance=instance)

        if mask is not None:
            dataframe[mask] = sub
        print(f'Evaluation achieved!', end='\r')

    @staticmethod
    def fit(dataframe, tags, model, grid=None, distribution=None, cpu=-1):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Check valid labels, at least several classes
        if not Tools.__check_labels(dataframe, {'label_encode': tags['label_encode']}):
            raise ValueError('Not enough unique labels where found, at least 2.')

        # Check for unsupervised stuff during fit mode
        if not isinstance(model, LabelSpreading) or not isinstance(model, LabelPropagation):
            dataframe = dataframe[dataframe[tags['label_encode']] != -1]

        # Used in case of higher predictions levels (inconsistent data)
        if not hasattr(model, 'is_inconsistent'):
            data = np.array(dataframe[tags['datum']].to_list())
        else:
            data = dataframe[tags['datum']].to_list()
        labels = np.array(dataframe[tags['label_encode']].to_list())

        if grid is not None:
            grid_search = GridSearchCV(model, scoring='f1_weighted', param_grid=grid, cv=2, iid=False, n_jobs=cpu)
            grid_search.fit(data, y=labels)
            model = grid_search.best_estimator_
            model.best_params = grid_search.best_params_
            return model
        elif distribution is not None:
            random_search = RandomizedSearchCV(model, scoring='f1_weighted', param_distributions=distribution, cv=Tools.VAL_RATIO, iid=False, n_jobs=cpu)
            random_search.fit(data, y=labels)
            model = random_search.best_estimator_
            model.best_params = random_search.best_params_
            return model
        else:
            model.fit(data, y=labels)
            model.best_params = {}
            return model

    @staticmethod
    def fit_predict(dataframe, tags, model, out, mask=None, folds=None, grid=None, distribution=None, cpu=-1, calibrate=None):
        # Fold needed for evaluation
        if 'Fold' not in dataframe:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Out fields
        out_predict = f'{out}_{Tools.PREDICTION}'
        out_score = f'{out}_{Tools.SCORE}'

        # Manage columns
        if out not in dataframe and hasattr(model, 'transform'):
            dataframe[out] = np.nan
        if out_predict not in dataframe and hasattr(model, 'predict'):
            dataframe[out_predict] = np.nan
        if out_score not in dataframe and hasattr(model, 'score'):
            dataframe[out_score] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Browse folds
        reference_folds = sub['Fold']
        if folds is None:
            folds = Tools.__default_folds(list(np.unique(reference_folds)))

        for index, current in enumerate(folds):
            # Create mask
            fit_mask = reference_folds.isin(current[0])

            # Check that current fold respect labels
            if not Tools.__check_labels(sub[fit_mask], {'label_encode': tags['label_encode']}):
                warnings.warn(f'Invalid fold, missing labels for fold {index}')
                continue

            # Clone model
            fitted_model = Tools.fit(sub[fit_mask], tags, model=deepcopy(model), grid=grid, distribution=distribution, cpu=cpu)

            # Make evaluation of calibration if needed
            if calibrate:
                calibrate_mask = reference_folds.isin(current[1]) # Prepare mask for calibration
                model_calibration = CustomCalibrationCV(fitted_model, cv=Tools.VAL_RATIO, method=calibrate)
                fitted_model = Tools.fit(sub[calibrate_mask], tags, model_calibration, cpu=cpu)

            # Prepare the prediction mask
            predict_mask = reference_folds.isin(current[-1])

            # Fill new data
            if hasattr(model, 'transform'):
                Tools.transform(sub, {'datum': tags['datum']}, fitted_model, out, mask=predict_mask)
            if hasattr(model, 'predict'):
                Tools.predict(sub, {'datum': tags['datum']}, fitted_model, out_predict, mask=predict_mask)
            if hasattr(model, 'score'):
                Tools.score(sub, {'datum': tags['datum']}, fitted_model, out_score, mask=predict_mask)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def generate_folds(pattern, upto):
        outputs = []
        for ind in np.arange(upto):
            temp = []
            for current in pattern:
                temp.append(list(((np.array(current) + ind - 1) % (upto)) + 1))
            outputs.append(temp)
        return outputs

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
    def predict(dataframe, tags, model, out, mask=None, instance=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        method = 'predict'
        if not hasattr(model, method):
            raise Exception(f'No method {method} found.')

        # Create column if doesnt exist
        if out not in dataframe and hasattr(model, 'predict'):
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Used in case of higher predictions levels (inconsistent data)
        if not hasattr(model, 'is_inconsistent'):
            data = np.array(sub[tags['datum']].to_list())
        else:
            data = sub[tags['datum']].to_list()

        # Set de predict values
        if instance is None:
            predictions = model.predict(data)
        else:
            predictions = model.predict_instance(data)

        sub[out] = pd.Series([p for p in predictions], index=sub.index)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def score(dataframe, tags, model, out, mask=None, instance=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        method = 'score'
        if not hasattr(model, method):
            raise Exception(f'No method {method} found.')

        # Create column if doesnt exist
        if out not in dataframe:
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Set values
        data = np.array(sub[tags['datum']].to_list())

        if instance is None:
            scores = np.argmax(model.score(data), axis=1)
        else:
            scores = model.score(data)

        sub[out] = pd.Series([s for s in scores], index=sub.index)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def predict_steps(dataframe, tags, model, out, mask=None, instance=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        method = 'predict_steps'
        if not hasattr(model, method):
            raise Exception(f'No method {method} found.')

        # Create column if doesnt exist
        if out not in dataframe:
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

        # Set values
        data = np.array(sub[tags['datum']].to_list())

        if instance is None:
            steps = model.predict_steps(data)
        else:
            steps = model.predict_steps(data)

        sub[out] = pd.Series([p for p in steps], index=sub.index)

        if mask is not None:
            dataframe[mask] = sub

    @staticmethod
    def transform(dataframe, tags, model, out, mask=None):
        # Check mandatory fields
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Check transform field
        method = 'transform'
        if not hasattr(model, method):
            raise Exception(f'No method {method} found.')

        # Create column if doesnt exist
        if out not in dataframe:
            dataframe[out] = np.nan

        # Mask dataframe
        if mask is None:
            sub = dataframe
        else:
            sub = dataframe[mask]

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

        if isinstance(model, MultimodalClassifier):
            sub[out] = [model.thresholds] * len(sub)
        else:
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

    @staticmethod
    def __default_folds(uniques):
        folds = []
        for fold in uniques:
            train = uniques.copy()
            train.remove(fold)
            folds.append((train, [fold]))
        return folds
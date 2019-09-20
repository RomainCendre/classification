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
        if not isinstance(tags, dict) or ['data', 'label', 'group'] not in tags.keys:
            raise Exception('Not a dict or missing tag: data, label, group.')
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
        if not isinstance(tags, dict) or ['data', 'label', 'group'] not in tags.keys:
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
    def fit_and_transform(dataframe, data_in, data_out, extractor):
        # Extract needed data
        references = dataframe['reference']

    @staticmethod
    def transform(dataframe, data_in, data_out, extractor):


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
                            features_file.create_dataset(reference, data=feature)
        else:
            features = self.__feature_extraction(prefix, inputs)

        # Update input
        inputs.update(prefix, features, references, 'data')

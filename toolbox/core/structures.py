import warnings

import pandas as pd
from collections import Iterable
from copy import copy, deepcopy
from itertools import chain
import numpy as np
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils.multiclass import unique_labels


class Settings:

    def __init__(self, data=None):
        if data is None:
            self.data = {}
        else:
            self.data = data

    def get_color(self, key):
        default_color = (0, 0, 0)
        colors = self.data.get('colors', None)
        if colors is None:
            return default_color
        return colors.get(key, default_color)

    def get_line(self, key):
        lines = self.data.get('lines', None)
        if lines is None:
            return None
        return lines.get(key, None)

    def is_in_data(self, check):
        for key, value in check.items():
            if key not in self.data or self.data[key] not in value:
                return False
        return True

    def update(self, data):
        self.data.update(data)


class Data:

    def __init__(self, data, filters=None):
        if data is not None and not isinstance(data, pd.DataFrame):
            raise Exception('Invalid data support!')
        self.data = data

        if filters is None:
            self.filters = {}
        else:
            self.filters = filters

    def get_number_samples(self, filters=None):
        cur_filters = copy(self.filters)
        if filters is not None:
            cur_filters.update(filters)

        query = self.to_query(cur_filters)
        if query:
            return len(self.data.query(query))
        else:
            return len(self.data)

    def get_from_key(self, key, filters=None, flatten=False):
        cur_filters = copy(self.filters)
        if filters is not None:
            cur_filters.update(filters)

        if key not in self.data.columns:
            return None

        query = self.to_query(cur_filters)
        if query:
            datas = self.data.query(query)[key].to_list()
        else:
            datas = self.data[key].to_list()

        if any(isinstance(el, Iterable) for el in datas) and flatten:
            datas = list(chain.from_iterable(datas))
        return np.array(datas)

    def get_unique_from_key(self, key, filters=None):
        return unique_labels(self.get_from_key(key, filters, flatten=True))

    def is_valid_keys(self, keys):
        return set(keys).issubset(set(self.data.columns))

    def set_filters(self, filters):
        self.filters = filters

    def to_query(self, filter_by):
        filters = []
        for key, values in filter_by.items():
            if not isinstance(values, list):
                values = [values]
            # Now make it list
            filters.append('{key} in {values}'.format(key=key, values=values))
        return ' & '.join(filters)


class Inputs(Data):

    def __init__(self, data, tags, filters={}):
        super().__init__(None, filters)
        self.data = data
        self.tags = tags
        self.encoders = None
        self.name = 'default'
        self.working_folder = None

    def build_folds(self, by_patients=True):
        # Data
        datas = self.get('datum')
        # Labels
        labels = self.get('label')
        # References
        references = self.get('reference')
        # Groups
        groups = self.get('group')

        # Rule to create folds
        if by_patients:
            split_rule = GroupKFold(n_splits=4)
        else:
            split_rule = KFold(n_splits=4)

        folds = np.zeros(groups.shape, dtype=np.int64)
        # Make folds
        current_folds = list(split_rule.split(X=datas, y=labels, groups=groups))
        for index, fold in enumerate(current_folds):
            folds[fold[1]] = index  # Add tests to folds

        # Make final folds
        self.update('Fold', folds, references)
        self.tags.update({'fold': 'Fold'})

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

    def copy_and_change(self, substitute):
        inputs = deepcopy(self)
        for key, value in substitute.items():
            inputs.data[key] = inputs.data[key].apply(lambda x: value[1] if x in value[0] else x)
        return inputs

    def decode(self, key, indices):
        is_list = isinstance(indices, np.ndarray)
        if not is_list:
            indices = np.array([indices])

        encoder = self.encoders.get(key, None)
        if encoder is None:
            result = indices
        else:
            try:
                result = encoder.inverse_transform(indices)
            except NotFittedError as e:
                encoder.fit(indices)
                result = encoder.inverse_transform(indices)

        if not is_list:
            result = result[0]

        return result

    def encode(self, key, data):
        is_list = isinstance(data, np.ndarray)
        if not is_list:
            data = np.array([data])

        encoder = self.encoders.get(key, None)
        if encoder is None:
            result = data
        else:
            try:
                result = encoder.transform(data)
            except NotFittedError as e:
                encoder.fit(data)
                result = encoder.transform(data)

        if not is_list:
            result = result[0]

        return result

    def get(self, tag, encode=True):
        if tag not in self.tags:
            warnings.warn('Invalid tag: {tag} not in {tags}'.format(tag=tag, tags=self.tags))
            return None

        # Filter and get groups
        metas = self.get_from_key(self.tags[tag])
        if encode:
            return self.encode(key=tag, data=metas)
        else:
            return metas

    def get_working_folder(self):
        return self.working_folder

    def set_encoders(self, encoders):
        self.encoders = encoders

    def set_working_folder(self, folder):
        self.working_folder = folder

    def sub_inputs(self, filters=None):
        inputs = deepcopy(self)
        cur_filters = copy(self.filters)
        if filters is not None:
            cur_filters.update(filters)
        inputs.data = inputs.data.query(super().to_query(cur_filters))
        inputs.data = inputs.data.reset_index(drop=True)
        return inputs

    def update(self, key, datas, references, field=None):
        if 'reference' not in self.tags:
            return None
        # As list for update of Dataframe
        references = [ref for ref in references]
        datas = [d for d in datas]
        # Now update, if already exist just update values
        temp = pd.DataFrame({key: datas, self.tags['reference']: references})
        if temp[key].dtype == 'int64':
            temp[key] = temp[key].astype('Int64')

        if key in self.data.columns:
            self.data = self.data.set_index(self.tags['reference'])
            temp = temp.set_index(self.tags['reference'])
            self.data.update(temp)
            self.data = self.data.reset_index()
        else:
            self.data = self.data.join(temp.set_index(self.tags['reference']), on=self.tags['reference'])

        if field is not None:
            self.tags.update({field: key})

    def read(self, name):
        self.data.read_csv(name)

    def write(self, folder):
        self.data.to_csv(folder/'{name}.csv'.format(name=self.name))


class Spectra(Inputs):

    def apply_average_filter(self, size):
        """This method allow user to apply an average filter of 'size'.

        Args:
            size: The size of average window.

        """
        self.data['datum'] = self.data['datum'].apply(lambda x: np.correlate(x, np.ones(size) / size, mode='same'))

    def apply_scaling(self, method='default'):
        """This method allow to normalize spectra.

            Args:
                method(:obj:'str') : The kind of method of scaling ('default', 'max', 'minmax' or 'robust')
            """
        if method == 'max':
            self.data['datum'] = self.data['datum'].apply(lambda x: preprocessing.maxabs_scale(x))
        elif method == 'minmax':
            self.data['datum'] = self.data['datum'].apply(lambda x: preprocessing.minmax_scale(x))
        elif method == 'robust':
            self.data['datum'] = self.data['datum'].apply(lambda x: preprocessing.robust_scale(x))
        else:
            self.data['datum'] = self.data['datum'].apply(lambda x: preprocessing.scale(x))

    def change_wavelength(self, wavelength):
        """This method allow to change wavelength scale and interpolate along new one.

        Args:
            wavelength(:obj:'array' of :obj:'float'): The new wavelength to fit.

        """
        self.data['datum'] = self.data.apply(lambda x: np.interp(wavelength, x['wavelength'], x['datum']), axis=1)
        self.data['wavelength'] = self.data['wavelength'].apply(lambda x: wavelength)

    def integrate(self):
        self.data['datum'] = self.data.apply(lambda x: [0, np.trapz(x['datum'], x['wavelength'])], axis=1)

    def norm_patient_by_healthy(self):
        query = self.to_query(self.filters)
        if query:
            data = self.data.query(query)
        else:
            data = self.data

        for name, group in data.groupby(self.tags['group']):
            # Get features by group
            row_ref = group[group.label == 'Sain']
            if len(row_ref) == 0:
                data.drop(group.index)
                continue
            mean = np.mean(row_ref.iloc[0]['datum'])
            std = np.std(row_ref.iloc[0]['datum'])
            for index, current in group.iterrows():
                data.iat[index, data.columns.get_loc('datum')] = (current['datum'] - mean) / std

    def norm_patient(self):
        query = self.to_query(self.filters)
        if query:
            data = self.data.query(query)
        else:
            data = self.data

        for name, group in data.groupby(self.tags['group']):
            # Get features by group
            group_data = np.array([current['datum'] for index, current in group.iterrows()])
            mean = np.mean(group_data)
            std = np.std(group_data)
            for index, current in group.iterrows():
                data.iat[index, data.columns.get_loc('datum')] = (current['datum'] - mean) / std

    def ratios(self):
        for name, current in self.data.iterrows():
            wavelength = current['wavelength']
            data_1 = current['datum'][np.logical_and(540 < wavelength, wavelength < 550)]
            data_2 = current['datum'][np.logical_and(570 < wavelength, wavelength < 580)]
            data_1 = np.mean(data_1)
            data_2 = np.mean(data_2)
            self.data.iloc[name, self.data.columns.get_loc('datum')] = data_1/data_2


class Outputs(Data):
    """Class that manage a result spectrum files.

    In this class we afford to manage spectrum results to write it on files.

    Attributes:

    """

    def __init__(self, results, encoders={}, name=''):
        """Make an initialisation of SpectrumReader object.

        Take a string that represent delimiter

        Args:

        """
        if isinstance(results, list):
            results = pd.DataFrame(results)

        super().__init__(results)
        self.name = name
        self.encoders = encoders

    def decode(self, key, indices):
        is_list = isinstance(indices, np.ndarray)
        if not is_list:
            indices = np.array([indices])

        encoder = self.encoders.get(key, None)
        if encoder is None:
            result = indices
        else:
            try:
                result = encoder.inverse_transform(indices)
            except NotFittedError as e:
                encoder.fit(indices)
                result = encoder.inverse_transform(indices)

        if not is_list:
            result = result[0]

        return result

    def encode(self, key, data):
        is_list = isinstance(data, np.ndarray)
        if not is_list:
            data = np.array([data])

        encoder = self.encoders.get(key, None)
        if encoder is None:
            result = data
        else:
            try:
                result = encoder.transform(data)
            except NotFittedError as e:
                encoder.fit(data)
                result = encoder.transform(data)

        if not is_list:
            result = result[0]

        return result

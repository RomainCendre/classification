import pandas as pd
from collections import Iterable
from copy import copy, deepcopy
from itertools import chain
import numpy as np
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError
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

    def check_load(self):
        if self.data is None:
            raise Exception('Data not loaded !')

    def get_from_key(self, key, filters=None, flatten=False):
        cur_filters = copy(self.filters)
        if filters is not None:
            cur_filters.update(filters)

        self.check_load()
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

    def __init__(self, folders, instance, loader, tags, filters={}, name=''):
        super().__init__(None, filters)
        self.folders = folders
        self.instance = instance
        self.loader = loader
        self.name = name
        self.tags = tags
        self.encoders = None

    def collapse(self, reference_tag, data_tag=None):
        if data_tag is None:
            data_tag = self.tags['data']

        query = self.to_query(self.filters)
        if query:
            data = self.data.query(query)
        else:
            data = self.data

        rows = []
        for name, group in data.groupby(reference_tag):
            # Get features by group
            # features = group[data_tag].tolist()
            raw_row = group.to_dict('list')
            row = {}
            for key, values in raw_row.items():
                if key == data_tag:
                    row.update({key: np.array(values)})
                    continue
                try:
                    test = list(set(values))
                    if len(test) == 1:
                        row.update({key: values[0]})
                except:
                    row.update({key: np.array(values)})

            rows.append(row)
        self.data = pd.DataFrame(rows)
        self.tags.update({'reference': reference_tag,
                          'data': data_tag})

    def aggregate(self, x):
        try:
            uniq = x.unique()
            if len(uniq) == 1:
                return uniq[0]
            else:
                return list(uniq)
        except:
            return list(x)

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

    def get_datas(self):
        self.check_load()
        if 'data' not in self.tags:
            return None

        return self.get_from_key(self.tags['data'])

    def get_groups(self):
        self.check_load()
        if 'groups' not in self.tags:
            return None

        # Filter and get groups
        groups = self.get_from_key(self.tags['groups'])
        return self.encode(key='groups', data=groups)

    def get_labels(self, encode=True):
        self.check_load()
        if 'label' not in self.tags:
            return None

        # Filter and get groups
        labels = self.get_from_key(self.tags['label'])
        if encode:
            return self.encode(key='label', data=labels)
        else:
            return labels

    def get_reference(self):
        self.check_load()
        if 'reference' not in self.tags:
            return None

        return self.get_from_key(self.tags['reference'])

    def get_unique_labels(self, encode=True):
        return unique_labels(self.get_labels(encode=encode))

    def load(self):
        if 'data' not in self.tags:
            print('data is needed at least.')
            return

        # Load data and set loading property to True
        self.data = pd.concat([self.loader(self.instance, folder) for folder in self.folders], sort=False, ignore_index=True)
        # self.data.reset_index()
        self.load_state = True

    def set_encoders(self, encoders):
        self.encoders = encoders

    def set_temporary_folder(self, temp_folder):
        self.temporary = temp_folder

    def sub_inputs(self, filters=None):
        inputs = deepcopy(self)
        cur_filters = copy(self.filters)
        if filters is not None:
            cur_filters.update(filters)
        inputs.data = inputs.data.query(super().to_query(cur_filters))
        inputs.data = inputs.data.reset_index(drop=True)
        return inputs

    def copy_and_change(self, substitute):
        inputs = deepcopy(self)
        for key, value in substitute.items():
            inputs.data[key] = inputs.data[key].apply(lambda x: value[1] if x in value[0] else x)
        return inputs

    def update(self, key, datas, references):
        self.check_load()
        if 'reference' not in self.tags:
            return None
        # As list for update of Dataframe
        references = [ref for ref in references]
        datas = [d for d in datas]
        # Now update, if already exist just update values
        if key in self.data.columns:
            self.data = self.data.set_index(self.tags['reference'])
            temp = pd.DataFrame({key: datas, self.tags['reference']: references}).set_index(self.tags['reference'])
            self.data.update(temp)
            self.data = self.data.reset_index()
        else:
            temp = pd.DataFrame({key: datas, self.tags['reference']: references})
            self.data = self.data.merge(temp)
        self.tags.update({'data': key})


class Spectra(Inputs):

    def apply_average_filter(self, size):
        """This method allow user to apply an average filter of 'size'.

        Args:
            size: The size of average window.

        """
        self.data['data'] = self.data['data'].apply(lambda x: np.correlate(x, np.ones(size) / size, mode='same'))

    def apply_scaling(self, method='default'):
        """This method allow to normalize spectra.

            Args:
                method(:obj:'str') : The kind of method of scaling ('default', 'max', 'minmax' or 'robust')
            """
        if method == 'max':
            self.data['data'] = self.data['data'].apply(lambda x: preprocessing.maxabs_scale(x))
        elif method == 'minmax':
            self.data['data'] = self.data['data'].apply(lambda x: preprocessing.minmax_scale(x))
        elif method == 'robust':
            self.data['data'] = self.data['data'].apply(lambda x: preprocessing.robust_scale(x))
        else:
            self.data['data'] = self.data['data'].apply(lambda x: preprocessing.scale(x))

    def change_wavelength(self, wavelength):
        """This method allow to change wavelength scale and interpolate along new one.

        Args:
            wavelength(:obj:'array' of :obj:'float'): The new wavelength to fit.

        """
        self.data['data'] = self.data.apply(lambda x: np.interp(wavelength, x['wavelength'], x['data']), axis=1)
        self.data['wavelength'] = self.data['wavelength'].apply(lambda x: wavelength)

    def integrate(self):
        self.data['data'] = self.data.apply(lambda x: [0, np.trapz(x['data'], x['wavelength'])], axis=1)

    def norm_patient(self):
        query = self.to_query(self.filters)
        if query:
            data = self.data.query(query)
        else:
            data = self.data

        for name, group in data.groupby(self.tags['group']):
            print(name)
            # Get features by group
            row_ref = group[group.label == "Sain"]
            if len(row_ref) == 0:
                data.drop(group.index)
                continue
            spectra_ref = row_ref.iloc[0]['data']
            # group['data'] = group['data'].apply(lambda x: x/spectra_ref)
            group['data'] = group['data'].apply(lambda x: (x - np.mean(spectra_ref))/np.std(spectra_ref))


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

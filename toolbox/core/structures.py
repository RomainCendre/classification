from collections import Iterable
from copy import copy
from itertools import chain

import pandas as pd
from numpy import correlate, ones, interp, array, ndarray
from numpy.ma import arange
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import unique_labels


class Settings:

    def __init__(self, data=None):
        if data is None:
            self.data = {}
        else:
            self.data = data

    def is_in_data(self, check):
        for key, value in check.items():
            if key not in self.data or self.data[key] not in value:
                return False
        return True

    def update(self, data):
        self.data.update(data)

    def get_color(self, key):
        return self.data.get(key, None)


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
        return array(datas)

    def get_unique_from_key(self, key, filters=None):
        return unique_labels(self.get_from_key(key, filters, flatten=True))

    def is_valid_keys(self, keys):
        return set(keys).issubset(set(self.data.columns))

    def to_query(self, filter_by):
        filters = []
        for key, values in filter_by.items():
            if not isinstance(values, list):
                values = [values]
            # Now make it list
            filters.append('{key} in {values}'.format(key=key, values=values))
        return ' & '.join(filters)


class Inputs(Data):

    def __init__(self, folders, instance, loader, tags, encoders={}, filters={}):
        super().__init__(None, filters)
        self.folders = folders
        self.instance = instance
        self.loader = loader
        self.tags = tags
        self.encoders = encoders

    def collapse(self, reference_tag, data_tag, flatten=True):
        references = list(set(self.get_from_key(reference_tag)))
        datas = []
        for reference in references:
            entities = self.sub_inputs({reference_tag: reference})
            features = entities[self.tags['data']].tolist()
            if flatten:
                features = array(features).flatten()
            data = entities.iloc[0].to_dict() #.agg(self.test).to_dict()
            data[data_tag] = features
            datas.append(data)
        self.data = pd.DataFrame(datas)
        self.tags.update({'reference': reference_tag,
                          'data': data_tag})

    def decode(self, key, indices):
        is_list = isinstance(indices, ndarray)
        if not is_list:
            indices = array([indices])

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
        is_list = isinstance(data, ndarray)
        if not is_list:
            data = array([data])

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

    def sub_inputs(self, filters=None):
        cur_filters = copy(self.filters)
        if filters is not None:
            cur_filters.update(filters)
        return self.data.query(super().to_query(cur_filters))

    def update_data(self, key, datas, references):
        self.check_load()
        if 'reference' not in self.tags:
            return None
        self.data = self.data.merge(pd.DataFrame({key: datas, self.tags['reference']: references}))
        self.tags.update({'data': key})


class Spectra(Inputs):

    def apply_average_filter(self, size):
        """This method allow user to apply an average filter of 'size'.

        Args:
            size: The size of average window.

        """
        self.data['data'] = self.data['data'].apply(lambda x: correlate(x, ones(size) / size, mode='same'))

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
        self.data['data'] = self.data.apply(lambda x: interp(wavelength, x['wavelength'], x['data']), axis=1)
        self.data['wavelength'] = self.data['wavelength'].apply(lambda x: wavelength)


class Outputs(Data):
    """Class that manage a result spectrum files.

    In this class we afford to manage spectrum results to write it on files.

    Attributes:

    """

    def __init__(self, results, name=''):
        """Make an initialisation of SpectrumReader object.

        Take a string that represent delimiter

        Args:

        """
        if isinstance(results, list):
            results = pd.DataFrame(results)

        super().__init__(results)
        self.name = name

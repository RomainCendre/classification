import warnings
from copy import copy

from numpy import correlate, ones, interp, asarray, zeros, array, ndarray, mean
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


class Data:

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


class DataSet:

    def __init__(self, data_set=None):
        if data_set is None:
            self.data_set = []
        else:
            self.data_set = data_set

    def __add__(self, other):
        """Add two spectra object

        Args:
             other (:obj:'Spectra'): A second spectra object to add.
        """
        data_set = []
        data_set.extend(self.data_set)
        data_set.extend(other.data_set)
        return DataSet(data_set)

    def apply_method(self, name, parameters={}):
        for data in self.data_set:
            getattr(data, name)(**parameters)

    def get_data(self, key: object, filter_by: object = {}) -> object:
        return asarray([data.data[key] for data in self.filter_by(filter_by)])

    def get_keys(self, filter_by={}):
        filtered_gen = self.filter_by(filter_by)
        # Init keys
        try:
            valid_keys = next(filtered_gen).data.keys()
        except StopIteration:
            return None

        # Check all keys exist
        for data in filtered_gen:
            keys = data.data.keys()
            valid_keys = list(set(valid_keys) & set(keys))
        return valid_keys

    def is_valid_keys(self, check_keys, filter_by={}):
        for key in check_keys:
            set_keys = self.get_keys(filter_by=filter_by)
            if set_keys is None or key not in set_keys:
                return False
        return True

    def get_unique_values(self, key: object, filter_by: object = {}):
        return unique_labels(self.get_data(key, filter_by))

    def methods(self):
        # If nothing in list
        if not self.data_set:
            return None

        # Init keys
        data = self.data_set[0]
        valid_methods = [method for method in dir(data) if callable(getattr(data, method))]

        # Check all keys exist
        for data in self.data_set:
            methods = [method for method in dir(data) if callable(getattr(data, method))]
            valid_methods = list(set(valid_methods) & set(methods))

        return valid_methods

    def update_data(self, key, datas, key_ref, references):
        if len(datas) != len(references):
            raise Exception('Data and references have mismatching sizes.')

        for data, reference in zip(datas, references):
            try:
                dataset = list(self.filter_by({key_ref: [reference]}))[0]
                dataset.data.update({key: data})
            except:
                warnings.warn('Reference {ref} not found.'.format(ref=reference))
                continue

    def filter_by(self, filter_by):
        for data in self.data_set:
            if data.is_in_data(filter_by):
                yield data


class Spectrum(Data):

    def __init__(self, data, wavelength, meta={}):
        meta.update({'Data': data,
                     'Wavelength': wavelength})
        super().__init__(meta)

    def apply_average_filter(self, size):
        """This method allow user to apply an average filter of 'size'.

        Args:
            size: The size of average window.

        """
        self.data['Data'] = correlate(self.data['Data'], ones(size) / size, mode='same')

    def apply_scaling(self, method='default'):
        """This method allow to normalize spectra.

            Args:
                method(:obj:'str') : The kind of method of scaling ('default', 'max', 'minmax' or 'robust')
            """
        if method == 'max':
            self.data['Data'] = preprocessing.maxabs_scale(self.data['Data'])
        elif method == 'minmax':
            self.data['Data'] = preprocessing.minmax_scale(self.data['Data'])
        elif method == 'robust':
            self.data['Data'] = preprocessing.robust_scale(self.data['Data'])
        else:
            self.data['Data'] = preprocessing.scale(self.data['Data'])

    def change_wavelength(self, wavelength):
        """This method allow to change wavelength scale and interpolate along new one.

        Args:
            wavelength(:obj:'array' of :obj:'float'): The new wavelength to fit.

        """
        self.data['Data'] = interp(wavelength, self.data['Wavelength'], self.data['Data'])
        self.data['Wavelength'] = wavelength


# Manage data for input on machine learning pipes
class Inputs:

    def __init__(self, folders, instance, loader, tags, encoders={}, style={}, filter_by={}):
        self.data = DataSet()
        self.folders = folders
        self.load_state = False
        self.instance = instance
        self.loader = loader
        self.tags = tags
        self.encoders = encoders
        self.style = style
        self.filter_by = filter_by

    def change_data(self, folders,  filter_by={}, encoders={}, tags={}, loader=None, keep=False):
        if loader is not None:
            self.loader = loader

        self.filter_by = filter_by
        self.encoders.update(encoders)
        self.tags.update(tags)

        if keep:
            self.folders.extend(folders)
        else:
            self.folders = folders

        # Set load property to true
        self.load_state = False

    def check_load(self):
        if not self.load_state:
            raise Exception('Data not loaded !')

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

    def get_from_key(self, key):
        self.check_load()
        if not self.data.is_valid_keys([key], filter_by=self.filter_by):
            return None

        return self.data.get_data(key=key, filter_by=self.filter_by)

    def get_datas(self):
        self.check_load()
        if 'data' not in self.tags:
            return None

        return self.data.get_data(key=self.tags['data'], filter_by=self.filter_by)

    def get_groups(self):
        self.check_load()
        if 'groups' not in self.tags:
            return None

        groups = self.data.get_data(key=self.tags['groups'], filter_by=self.filter_by)

        return self.encode(key='groups', data=groups)

    def get_labels(self, encode=True):
        self.check_load()
        if 'label' not in self.tags:
            return None

        labels = self.data.get_data(key=self.tags['label'], filter_by=self.filter_by)
        if encode:
            return self.encode(key='label', data=labels)
        else:
            return labels

    def get_reference(self):
        self.check_load()
        if 'reference' not in self.tags:
            return None

        return self.data.get_data(key=self.tags['reference'], filter_by=self.filter_by)

    def get_style(self, key, label):
        return self.style[key][label]

    def get_unique_labels(self):
        self.check_load()
        labels = self.data.get_unique_values(key=self.tags['label'], filter_by=self.filter_by)
        return self.encode(key='label', data=labels)

    def load(self):
        if 'data' not in self.tags:
            print('data is needed at least.')
            return

        self.data = DataSet()
        for folder in self.folders:
            self.data += self.loader(self.instance, folder)

        # Set load property to true
        self.load_state = True

    def update_data(self, key, datas, references):
        self.check_load()
        if 'reference' not in self.tags:
            return None

        self.data.update_data(key=key, datas=datas, key_ref=self.tags['reference'], references=references)
        self.tags.update({'data': key})

    def to_sub_input(self, filter=None):
        current_filter = copy(self.filter_by)
        if filter is not None:
            current_filter.update(filter)

        inputs = copy(self)
        inputs.data = DataSet(list(self.data.filter_by(current_filter)))
        return inputs

    def patch_method(self, use_mean=True):
        references = list(set(self.get_from_key('Reference')))
        new_data = []
        for reference in references:
            entities = self.to_sub_input({'Reference': [reference]})
            predictions = entities.get_datas().tolist()
            predictions = array(predictions).flatten()
            data = entities.data.data_set[0]
            data.update({'Data': predictions})
            new_data.append(data)
        self.tags.update({'reference': 'Reference',
                          'data': 'Data'})
        self.data = DataSet(new_data)


class Result(Data):

    def __init__(self, result=None):
        super().__init__(result)


class Results(DataSet):
    """Class that manage a result spectrum files.

    In this class we afford to manage spectrum results to write it on files.

    Attributes:

    """

    def __init__(self, results, name=''):
        """Make an initialisation of SpectrumReader object.

        Take a string that represent delimiter

        Args:

        """
        super().__init__(results)
        self.name = name


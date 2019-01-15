from numpy import correlate, ones, interp, asarray
from sklearn import preprocessing
from sklearn.utils.multiclass import unique_labels


class Data:

    def __init__(self, data={}):
        self.data = data

    def is_in_data(self, check):
        for key, value in check.items():
            if self.data[key] not in value:
                return False
        return True


class DataSet:

    def __init__(self, data_set):
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
        return asarray([data.data[key] for data in self.__filter_by(filter_by)])

    def get_keys(self, filter_by={}):
        filtered_gen = self.__filter_by(filter_by)
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

    def is_valid_keys(self, check_keys):
        for key in self.get_keys():
            if key not in check_keys:
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

    def __filter_by(self, filter_by):
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


class Result(Data):

    def __init__(self, result={}):
        super().__init__(result)


class Results(DataSet):
    """Class that manage a result spectrum files.

    In this class we afford to manage spectrum results to write it on files.

    Attributes:

    """

    def __init__(self, results, map_index, name=''):
        """Make an initialisation of SpectrumReader object.

        Take a string that represent delimiter

        Args:

        """
        super().__init__(results)
        self.name = name
        self.map_index = map_index

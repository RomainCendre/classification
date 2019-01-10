from numpy import correlate, ones, interp, asarray
from sklearn import preprocessing


class Data:

    def __init__(self, data, meta={}):
        self.data = data
        self.meta = meta

    def is_in_meta(self, check):
        for key, value in check.items():
            if self.meta[key] not in value:
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

    def get_data(self, filter_by={}):
        return asarray([data.data for data in self.__filter_by(filter_by)])

    def get_meta(self, meta: object, filter_by: object = {}) -> object:
        return asarray([data.meta[meta] for data in self.__filter_by(filter_by)])

    def meta(self, filter_by={}):
        filtered_gen = self.__filter_by(filter_by)
        # Init keys
        try:
            valid_keys = next(filtered_gen).meta.keys()
        except StopIteration:
            return None

        # Check all keys exist
        for data in filtered_gen:
            keys = data.meta.keys()
            valid_keys = list(set(valid_keys) & set(keys))
        return valid_keys

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
            if data.is_in_meta(filter_by):
                yield data


class Spectrum(Data):

    def __init__(self, data, wavelength, meta={}):
        super(Spectrum, self).__init__(data, meta)
        self.wavelength = wavelength

    def apply_average_filter(self, size):
        """This method allow user to apply an average filter of 'size'.

        Args:
            size: The size of average window.

        """
        self.data = correlate(self.data, ones(size) / size, mode='same')

    def apply_scaling(self, method='default'):
        """This method allow to normalize spectra.

            Args:
                method(:obj:'str') : The kind of method of scaling ('default', 'max', 'minmax' or 'robust')
            """
        if method == 'max':
            self.data = preprocessing.maxabs_scale(self.data)
        elif method == 'minmax':
            self.data = preprocessing.minmax_scale(self.data)
        elif method == 'robust':
            self.data = preprocessing.robust_scale(self.data)
        else:
            self.data = preprocessing.scale(self.data)

    def change_wavelength(self, wavelength):
        """This method allow to change wavelength scale and interpolate along new one.

        Args:
            wavelength(:obj:'array' of :obj:'float'): The new wavelength to fit.

        """
        self.data = interp(wavelength, self.wavelength, self.data)
        self.wavelength = wavelength

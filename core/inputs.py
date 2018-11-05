from numpy import correlate, ones, interp
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


class Dataset:

    def __init__(self, dataset):
        self.dataset = dataset

    def __add__(self, other):
        """Add two spectra object

        Args:
             other (:obj:'Spectra'): A second spectra object to add.
        """
        dataset = []
        dataset.extend(self.dataset)
        dataset.extend(other.dataset)
        return Dataset(dataset)

    def apply_method(self, name, parameters=[]):
        for data in self.dataset:
            getattr(data, name)(*parameters)

    def get(self, label, filter={}, groups=None):
        if groups is None:
            return [(data.data, data.meta[label]) for data in self.dataset if data.is_in_meta(filter)]
        else:
            return [(data.data, data.meta[label], data.meta[groups]) for data in self.dataset if data.is_in_meta(filter)]

    def meta(self):
        # If nothing in list
        if not self.dataset:
            return None
        # Init keys
        valid_keys = self.dataset[0].meta.keys()
        # Check all keys exist
        for data in self.dataset:
            keys = data.meta.keys()
            valid_keys = list(set(valid_keys) & set(keys))
        return valid_keys

    def methods(self):
        # If nothing in list
        if not self.dataset:
            return None

        # Init keys
        data = self.dataset[0]
        valid_methods = [ method for method in dir(data) if callable(getattr(data, method))]

        # Check all keys exist
        for data in self.dataset:
            methods = [ method for method in dir(data) if callable(getattr(data, method))]
            valid_methods = list(set(valid_methods) & set(methods))

        return valid_methods

class Spectrum(Data):

    def __init__(self, data, wavelength, meta={}):
        super(Spectrum, self).__init__(data,meta)
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



# class Patient:
#     def __init__(self):
#         self.meta = {}
#         self.images = []
#         self.name = ''
#
#     def filter_modality(self, modalities):
#         self.images = [image for image in self.images if image.modality in modalities]
#
#     def datas(self, key):
#         datas = [image.data for image in self.images]
#         labels = [self.meta[key] for index in range(0, len(self.images))]
#         names = [self.name for index in range(0, len(self.images))]
#         return datas, labels, names

# class Spectrum:
#     """Class that manage a spectrum representation.
#
#      In this class we afford an object that represent a spectrum and all
#      needed method to manage it.
#
#      Attributes:
#          spectrum (:obj:'list' of :obj:'float'): A set of spectrum values.
#          wavelength (:obj:'list' of :obj:'float'): A set of wavelength values.
#          label (:obj:'str'): The corresponding label.
#          patient_name (:obj:'str'): The corresponding patient name.
#          patient_label (:obj:'str'): The corresponding patient label.
#          spectrum_id (:obj:'int'): The corresponding column.
#          device (:obj:'str'): The version of device/operator combination
#          location (:obj:'str'): The place where occur acquisition
#
#      """
#
#     def __init__(self, spectrum, wavelength, label, spectrum_id):
#         """Make an initialisation of Spectrum object.
#
#         Take three arguments : spectrum values, wavelength values (of same size than spectrum)
#         and a label.
#
#         Args:
#             spectrum (:obj:'list' of :obj:'float'): A set of spectrum values.
#             wavelength (:obj:'list' of :obj:'float'): A set of wavelength values.
#             label (:obj:'str'): The corresponding label.
#             spectrum_id (:obj:'int'): The corresponding column.
#
#         """
#         self.spectrum = spectrum
#         self.wavelength = wavelength
#
#         self.label = label
#
#         self.spectrum_id = spectrum_id
#         self.patient_name = ''
#         self.patient_label = ''
#         self.device = ''
#         self.location = ''
#
#     def apply_average_filter(self, size):
#         """This method allow user to apply an average filter of 'size'.
#
#         Args:
#             size: The size of average window.
#
#         """
#         self.spectrum = correlate(self.spectrum, ones(size) / size, mode='same')
#
#     def apply_scaling(self, method='default'):
#         """This method allow to normalize spectra.
#
#             Args:
#                 method(:obj:'str') : The kind of method of scaling ('default', 'max', 'minmax' or 'robust')
#             """
#         if method == 'max':
#             self.spectrum = preprocessing.maxabs_scale(self.spectrum)
#         elif method == 'minmax':
#             self.spectrum = preprocessing.minmax_scale(self.spectrum)
#         elif method == 'robust':
#             self.spectrum = preprocessing.robust_scale(self.spectrum)
#         else:
#             self.spectrum = preprocessing.scale(self.spectrum)
#
#     def change_wavelength(self, wavelength):
#         """This method allow to change wavelength scale and interpolate along new one.
#
#         Args:
#             wavelength(:obj:'array' of :obj:'float'): The new wavelength to fit.
#
#         """
#         self.spectrum = interp(wavelength, self.wavelength, self.spectrum)
#         self.wavelength = wavelength



# class Patients:
#
#     def __init__(self):
#         self.patients = []
#
#     def filter_modality(self, modalities):
#         for patient in self.patients:
#             patient.filter_modality(modalities)
#
#     def datas(self, key):
#         global_datas = []
#         global_labels = []
#         global_names = []
#         for patient in self.patients:
#             datas, labels, patient = patient.datas(key)
#             global_datas.extend(datas)
#             global_labels.extend(labels)
#             global_names.extend(patient)
#         return asarray(global_datas), asarray(global_labels), asarray(global_names)
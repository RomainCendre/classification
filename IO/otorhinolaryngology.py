from numpy import genfromtxt
from os.path import join, splitext

from core.input import Spectrum, Dataset


class Reader:
    """Class that manage a spectrum files.

     In this class we afford to manage spectrum files and get data from them.

     Attributes:
         delimiter (:obj:'list' of :obj:'float'): A delimiter used in spectrum files.

     """
    COLUMN_WAVELENGTH = 0
    COLUMN_FIRST = 1
    ROW_LABEL = 0
    ROW_WAVELENGTH = 6
    FILE_EXTENSION = '*.csv'

    def __init__(self, delimiter):
        """Make an initialisation of SpectrumReader object.

        Take a string that represent delimiter

        Args:
             delimiter (:obj:'str'): The delimiter string.
        """
        self.delimiter = delimiter

    def read_file(self, file_path):
        """Read a spectrum file and return spectra

        Args:
             file_path (:obj:'str'): The file to read spectrum from.

        Returns:
            A spectra object.
        """
        # Read csv
        csv = genfromtxt(file_path, dtype='str', delimiter=self.delimiter)
        spectra = []
        # Build spectrum
        for x in range(self.COLUMN_FIRST, csv.shape[1]):
            meta = {'label': csv[self.ROW_LABEL, x],
                    'spectrum_id': x - self.COLUMN_FIRST}
            spectrum = Spectrum(
                data=csv[self.ROW_WAVELENGTH:csv.shape[0], x].astype("float"),
                wavelength=csv[self.ROW_WAVELENGTH:csv.shape[0], self.COLUMN_WAVELENGTH].astype("float"),
                meta=meta)
            spectra.append(spectrum)
        return spectra

    def read_table(self, table_path):
        """Read a specific file that map meta data and spectrum files

        Args:
             table_path (:obj:'str'): The matching file.

        Returns:
            A spectra object.
        """
        # Read csv
        base_folder = splitext(table_path)[0]
        csv = genfromtxt(table_path, dtype='str', delimiter=self.delimiter, skip_header=1)
        spectra = []
        for row_index in range(1, len(csv)):
            current_file = csv[row_index, 5]
            if not current_file:
                continue
            # Get patient meta data
            meta = {'patient_name': csv[row_index, 5],
                    'patient_label': csv[row_index, 2],
                    'device': csv[row_index, 3],
                    'location': csv[row_index, 4]}

            patient_datas = self.read_file(join(base_folder, current_file))
            [data.meta.update(meta) for data in patient_datas]
            spectra.extend(patient_datas)
        return Dataset(spectra)

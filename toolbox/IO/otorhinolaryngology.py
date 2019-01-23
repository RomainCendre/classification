import pandas
from os.path import join, splitext
from numpy.ma import array
from toolbox.core.structures import Spectrum, DataSet


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

    def __init__(self):
        """Make an initialisation of SpectrumReader object.

        Take a string that represent delimiter

        Args:
        """

    def read_file(self, file_path):
        """Read a spectrum file and return spectra

        Args:
             file_path (:obj:'str'): The file to read spectrum from.

        Returns:
            A spectra object.
        """
        # Read csv
        csv = array(pandas.read_csv(file_path, header=None, dtype=str).values)
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
        table = pandas.read_csv(table_path, dtype=str).fillna('')
        spectra = []
        for ind, row in table.iterrows():
            current_file = row['fichier']
            if not current_file:
                continue
            # Get patient meta data
            meta = {'patient_name': row['fichier'],
                    'patient_label': row['pathologie'],
                    'operator': row['operateur'],
                    'location': row['provenance']}

            patient_datas = self.read_file(join(base_folder, current_file))
            [data.data.update(meta) for data in patient_datas]
            spectra.extend(patient_datas)
        return DataSet(spectra)

from pathlib import Path

import pandas
from numpy.ma import array
from toolbox.core.structures import Spectra


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

    @staticmethod
    def read_file(file_path):
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
        for x in range(Reader.COLUMN_FIRST, csv.shape[1]):
            spectrum = {'label': csv[Reader.ROW_LABEL, x],
                        'spectrum_id': x - Reader.COLUMN_FIRST}
            spectrum.update({'data': csv[Reader.ROW_WAVELENGTH:csv.shape[0], x].astype("float"),
                             'wavelength': csv[Reader.ROW_WAVELENGTH:csv.shape[0], Reader.COLUMN_WAVELENGTH].astype(
                                 "float")})
            spectra.append(spectrum)
        return pandas.DataFrame(spectra)

    def read_table(self, table_path):
        """Read a specific file that map meta data and spectrum files

        Args:
             table_path (:obj:'str'): The matching file.

        Returns:
            A spectra object.
        """
        # Read csv
        meta_patient = pandas.read_csv(table_path, dtype=str).fillna('')
        meta_patient['Reference'] = meta_patient.apply(lambda row: '{id}_{patient}'.format(id=row['identifier'],
                                                                                           patient=row['patient']),
                                                       axis=1)
        spectra = []
        for ind, row in meta_patient.iterrows():
            current_file = row['fichier']
            if not current_file:
                continue

            patient_datas = Reader.read_file(table_path.parent/table_path.stem/current_file)
            patient_datas['Reference'] = row['Reference']
            patient_datas = patient_datas.merge(meta_patient, on='Reference')

            patient_datas['Reference_spectrum'] = patient_datas.apply(
                lambda row: '{reference}_{spectrum}'.format(reference=row['Reference'],
                                                            spectrum=row['spectrum_id']), axis=1)
            spectra.append(patient_datas)
        return pandas.concat(spectra, sort=False, ignore_index=True)

import pandas
import numpy as np
from numpy.ma import array


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

            patient_datas = Reader.read_file(table_path.parent / table_path.stem / current_file)
            patient_datas['Reference'] = row['Reference']
            patient_datas = patient_datas.merge(meta_patient, on='Reference')

            patient_datas['Reference_spectrum'] = patient_datas.apply(
                lambda row: '{reference}_{spectrum}'.format(reference=row['Reference'],
                                                            spectrum=row['spectrum_id']), axis=1)
            spectra.append(patient_datas)
        return pandas.concat(spectra, sort=False, ignore_index=True)


class Generator:

    def __init__(self, nb_spectra, nb_patients):
        self.nb_spectra = nb_spectra
        self.nb_patients = nb_patients

    def generate_study(self):
        random_patient = list(np.random.randint(3, size=self.nb_patients))
        patients = []
        for index, random in enumerate(random_patient):
            patient = self.generate_patient(random)
            patient['patient'] = index
            patient['operateur'] = 'V0'
            patients.append(patient)

        patients = pandas.concat(patients, sort=False, ignore_index=True)
        patients['Reference'] = patients.apply(lambda row: '{patient}'.format(patient=row['patient']),
                                               axis=1)

        patients['Reference_spectrum'] = patients.apply(
            lambda row: '{reference}_{spectrum}'.format(reference=row['Reference'],
                                                        spectrum=row['spectrum_id']), axis=1)
        return patients

    def generate_patient(self, mode):
        random_data = list(np.random.randint(mode + 1, size=np.random.randint(self.nb_spectra[0], self.nb_spectra[1])))
        data = []
        for index, random in enumerate(random_data):
            datum = self.generate_spectrum(random)
            datum['spectrum_id'] = index
            data.append(datum)

        if mode == 2:
            label = 'Cancer'
        elif mode == 1:
            label = 'Precancer'
        else:
            label = 'Sain'

        patient = pandas.concat(data)
        patient['pathologie'] = label
        return patient

    def generate_spectrum(self, mode):
        wavelength = np.arange(start=445, stop=962, step=1)
        indices = np.linspace(0, 1, len(wavelength), endpoint=False)

        if mode == 2:
            data = np.sin(2 * np.pi * indices)
            label = 'Cancer'
        elif mode == 1:
            data = np.square(2 * np.pi * indices)
            label = 'Precancer'
        else:
            indices[:] = 0
            data = indices
            label = 'Sain'

        return pandas.DataFrame({'data': [data], 'wavelength': [wavelength], 'label': label})

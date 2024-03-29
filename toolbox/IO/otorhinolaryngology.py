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
            spectrum = {'Diagnosis': csv[Reader.ROW_LABEL, x],
                        'ID_Spectrum': x - Reader.COLUMN_FIRST}
            spectrum.update({'Datum': csv[Reader.ROW_WAVELENGTH:csv.shape[0], x].astype('float'),
                             'Wavelength': csv[Reader.ROW_WAVELENGTH:csv.shape[0], Reader.COLUMN_WAVELENGTH].astype('float')})
            spectra.append(spectrum)
        return pandas.DataFrame(spectra)

    def read_table(self, table_paths):
        """Read a specific file that map meta data and spectrum files

        Args:
             table_path (:obj:'str'): The matching file.

        Returns:
            A spectra object.
        """
        # Read csv
        spectra = []
        for table_path in table_paths:
            meta_patient = pandas.read_csv(table_path, dtype=str).fillna('')
            meta_patient['Reference'] = meta_patient.apply(lambda row: '{id}_{patient}'.format(id=row['Identifier'], patient=row['Patient']), axis=1)
            for ind, row in meta_patient.iterrows():
                current_file = row['File']
                if not current_file:
                    continue

                patient_datas = Reader.read_file(table_path.parent / table_path.stem / current_file)
                patient_datas['Category'] = table_path.stem
                patient_datas['Reference'] = row['Reference']
                patient_datas = patient_datas.merge(meta_patient, on='Reference')

                patient_datas['Ref_Spectrum'] = patient_datas.apply(
                    lambda row: '{reference}_{spectrum}'.format(reference=row['Reference'],
                                                                spectrum=row['ID_Spectrum']), axis=1)
                spectra.append(patient_datas)
        dataframe = pandas.concat(spectra, sort=False, ignore_index=True)
        # Set pathological label
        dataframe['Pathological'] = dataframe['Diagnosis']
        mask = ~(dataframe['Diagnosis'] == 'Sain')
        dataframe.loc[mask, 'Pathological'] = 'Pathological'  # Set new label
        # Set malignant label
        dataframe['Cancerous'] = dataframe['Diagnosis']
        mask = ~(dataframe['Diagnosis'] == 'Cancer')  # Set new label
        dataframe.loc[mask, 'Cancerous'] = 'Rest'
        return dataframe

    @staticmethod
    def change_wavelength(inputs, tags, wavelength):
        """This method allow to change wavelength scale and interpolate along new one.

        Args:
            wavelength(:obj:'array' of :obj:'float'): The new wavelength to fit.

        """
        mandatory = ['datum', 'wavelength']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')
        inputs[tags['datum']] = inputs.apply(lambda x: np.interp(wavelength, x[tags['wavelength']], x[tags['datum']]), axis=1)
        inputs[tags['wavelength']] = [wavelength] * len(inputs)
        return inputs

    @staticmethod
    def remove_negative(inputs, tags):
        """This method allow to change wavelength scale and interpolate along new one.

        Args:
            wavelength(:obj:'array' of :obj:'float'): The new wavelength to fit.

        """
        mandatory = ['datum']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Not a dict or missing tag: {mandatory}.')
        inputs[tags['datum']] = inputs.apply(lambda x: Reader.__numpy_remove_negative(x[tags['datum']]), axis=1)
        return inputs

    @staticmethod
    def __numpy_remove_negative(x):
        x[x < 0] = 0
        return x


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

        return pandas.DataFrame({'datum': [data], 'wavelength': [wavelength], 'label': label})

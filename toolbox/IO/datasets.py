from os.path import normpath, expanduser, dirname, join
from sklearn.preprocessing import LabelEncoder
from toolbox.IO import dermatology, otorhinolaryngology
from toolbox.core.structures import Inputs, Spectra, Settings
from toolbox.core.transforms import OrderedEncoder


class Dataset:

    @staticmethod
    def full_images():
        home_path = expanduser('~')
        input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                         normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
        return Dataset.__full_images(input_folders)

    @staticmethod
    def patches_images(folder, size):
        home_path = expanduser('~')
        input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                         normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
        return Dataset.__patches_images(input_folders, folder, size)

    @staticmethod
    def spectras():
        home_path = expanduser('~')
        location = normpath('{home}/Data/Neck/'.format(home=home_path))
        input_folders = [join(location, 'Patients.csv'), join(location, 'Temoins.csv')]
        return Dataset.__spectras(input_folders)

    @staticmethod
    def thumbnails():
        home_path = expanduser('~')
        input_folders = [normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))]
        return Dataset.__thumbnails(input_folders)

    @staticmethod
    def test_full_images():
        here_path = dirname(__file__)
        input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                         normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
        return Dataset.__full_images(input_folders)

    @staticmethod
    def test_patches_images(folder, size):
        here_path = dirname(__file__)
        input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                         normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
        return Dataset.__patches_images(input_folders, folder, size)

    @staticmethod
    def test_spectras():
        here_path = dirname(__file__)
        input_folders = [normpath('{here}/data/spectroscopy/Patients.csv'.format(here=here_path))]
        return Dataset.__spectras(input_folders)

    @staticmethod
    def test_thumbnails():
        here_path = dirname(__file__)
        input_folders = [normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))]
        return Dataset.__thumbnails(input_folders)

    @staticmethod
    def __thumbnails(folders):
        inputs = Inputs(folders=folders, instance=dermatology.Reader(),
                        loader=dermatology.Reader.scan_folder_for_images,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'})
        inputs.load()
        return inputs

    @staticmethod
    def __full_images(folders):
        filters = {'Modality': 'Microscopy',
                   'Label': ['Malignant', 'Benign', 'Normal']}
        inputs = Inputs(folders=folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'}, filters=filters,
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs

    @staticmethod
    def __patches_images(folders, extraction_folder, size):
        filters = {'Modality': 'Microscopy',
                   'Label': ['Malignant', 'Benign', 'Normal']}
        inputs = Inputs(folders=folders, instance=dermatology.Reader(extraction_folder),
                        loader=dermatology.Reader.scan_folder_for_patches,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Patch_Reference'}, filters=filters,
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs

    @staticmethod
    def __spectras(folders):
        inputs = Spectra(folders=folders, instance=otorhinolaryngology.Reader(),
                         loader=otorhinolaryngology.Reader.read_table,
                         tags={'data': 'data', 'label': 'label', 'group': 'Reference',
                               'reference': 'Reference_spectrum'})
        inputs.load()
        return inputs


class DefinedSettings:

    @staticmethod
    def get_default_orl():
        colors = dict(Cancer=(1, 0, 0), Precancer=(0.5, 0.5, 0), Sain=(0, 1, 0), Luck=(0, 0, 1))
        lines = {'Cancer': {'linestyle': '-'},
                 'Precancer': {'linestyle': '-.'},
                 'Sain': {'linestyle': ':'},
                 'Luck': {'linestyle': '--'}}
        return Settings({'colors': colors, 'lines': lines})

    @staticmethod
    def get_default_dermatology():
        colors = dict(Malignant=(1, 0, 0), Benign=(0.5, 0.5, 0), Normal=(0, 1, 0), Luck=(0, 0, 1))
        lines = {'Malignant': {'linestyle': '-'},
                 'Benign': {'linestyle': '-.'},
                 'Normal': {'linestyle': ':'},
                 'Luck': {'linestyle': '--'}}
        return Settings({'colors': colors, 'lines': lines})

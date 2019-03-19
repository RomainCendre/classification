from os.path import normpath, expanduser, dirname
from sklearn.preprocessing import LabelEncoder
from toolbox.IO import dermatology
from toolbox.core.structures import Inputs
from toolbox.core.transforms import OrderedEncoder


class Dataset:

    @staticmethod
    def thumbnails():
        home_path = expanduser('~')
        input_folders = [normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))]
        return Dataset.__thumbnails(input_folders)

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
    def test_thumbnails():
        here_path = dirname(__file__)
        input_folders = [normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))]
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
    def __thumbnails(folders):
        inputs = Inputs(folders=folders, instance=dermatology.Reader(),
                        loader=dermatology.Reader.scan_folder_for_images,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'})
        inputs.load()
        return inputs

    @staticmethod
    def __full_images(folders):
        filter_by = {'Modality': 'Microscopy',
                     'Label': ['Malignant', 'Benign', 'Normal']}
        inputs = Inputs(folders=folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'}, filter_by=filter_by,
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs

    @staticmethod
    def __patches_images(folders, extraction_folder, size):
        filter_by = {'Modality': 'Microscopy',
                     'Label': ['Malignant', 'Benign', 'Normal']}
        inputs = Inputs(folders=folders, instance=dermatology.Reader(extraction_folder),
                        loader=dermatology.Reader.scan_folder_for_patches,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Patch_Reference'}, filter_by=filter_by,
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs


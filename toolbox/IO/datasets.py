from os.path import normpath, expanduser, dirname
from sklearn.preprocessing import LabelEncoder
from toolbox.IO import dermatology
from toolbox.core.structures import Inputs
from toolbox.core.transforms import OrderedEncoder


class Dataset:

    @staticmethod
    def thumbnails():
        home_path = expanduser('~')
        input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
        inputs = Inputs(folders=[input_folder], instance=dermatology.Reader(),
                        loader=dermatology.Reader.scan_folder_for_images,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'},
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs

    @staticmethod
    def full_images():
        home_path = expanduser('~')
        filter_by = {'Modality': 'Microscopy',
                     'Label': ['Malignant', 'Benign', 'Normal']}
        input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                         normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
        inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                        filter_by=filter_by,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference', 'groups': 'ID'},
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs

    @staticmethod
    def patches_images(folder, size):
        home_path = expanduser('~')
        filter_by = {'Modality': 'Microscopy',
                     'Label': ['Malignant', 'Benign', 'Normal']}
        input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                         normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
        inputs = Inputs(folders=input_folders, instance=dermatology.Reader(folder),
                        filter_by=filter_by, loader=dermatology.Reader.scan_folder_for_patches,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference', 'groups': 'ID'},
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs

    @staticmethod
    def test_thumbnails():
        here_path = dirname(__file__)
        inputs_folder = normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))
        inputs = Inputs(folders=[inputs_folder], instance=dermatology.Reader(),
                        loader=dermatology.Reader.scan_folder_for_images,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'})
        inputs.load()
        return inputs

    @staticmethod
    def test_full_images():
        here_path = dirname(__file__)
        filter_by = {'Modality': 'Microscopy',
                     'Label': ['Malignant', 'Benign', 'Normal']}
        input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                         normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
        inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'}, filter_by=filter_by)
        inputs.load()
        return inputs

    @staticmethod
    def test_patches_images(folder, size):
        here_path = dirname(__file__)
        filter_by = {'Modality': 'Microscopy',
                     'Label': ['Malignant', 'Benign', 'Normal']}
        input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                         normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
        inputs = Inputs(folders=input_folders, instance=dermatology.Reader(folder),
                        loader=dermatology.Reader.scan_folder_for_patches(),
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'}, filter_by=filter_by)
        inputs.load()
        return inputs
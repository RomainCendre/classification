from os import path as ospath, makedirs
from tempfile import gettempdir
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from toolbox.IO import dermatology, otorhinolaryngology
from toolbox.core.structures import Inputs, Spectra, Settings
from toolbox.core.transforms import OrderedEncoder


class Dataset:

    def __init__(self, temporary=None, patch_parameters=None, multi_coefficients=None):
        self.temporary = temporary
        self.patch_parameters = patch_parameters
        self.multi_coefficients = multi_coefficients

    @staticmethod
    def images():
        home_path = ospath.expanduser('~')
        input_folders = [ospath.normpath('{home}/Data/Skin/Saint_Etienne/Patients'.format(home=home_path))]
        return Dataset.__images(input_folders)

    @staticmethod
    def multiresolution(coefficients):
        home_path = ospath.expanduser('~')
        input_folders = [ospath.normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                         ospath.normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
        multi_folder = ospath.join(home_path, 'Multi')
        return Dataset.__multi_images(input_folders, multi_folder, coefficients)

    @staticmethod
    def patches_images(size, overlap):
        home_path = ospath.expanduser('~')
        input_folders = [ospath.normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                         ospath.normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
        patch_folder = ospath.join(home_path, 'Patch')
        return Dataset.__patches_images(input_folders, patch_folder, size, overlap)

    @staticmethod
    def spectras():
        home_path = ospath.expanduser('~')
        location = ospath.normpath('{home}/Data/Neck/'.format(home=home_path))
        input_folders = [ospath.join(location, 'Patients.csv'), ospath.join(location, 'Temoins.csv')]
        return Dataset.__spectras(input_folders)

    @staticmethod
    def test_full_images():
        here_path = ospath.dirname(__file__)
        input_folders = [ospath.normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                         ospath.normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
        return Dataset.__full_images(input_folders)

    @staticmethod
    def test_multiresolution(coefficients):
        here_path = ospath.dirname(__file__)
        input_folders = [ospath.normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                         ospath.normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]

        multi_folder = ospath.join(gettempdir(), 'Multi')
        return Dataset.__multi_images(input_folders, multi_folder, coefficients)

    @staticmethod
    def test_patches_images(size, overlap):
        home_path = ospath.expanduser('~')
        input_folders = [ospath.normpath('{home}/Desktop/Patients_test'.format(home=home_path))]

        patch_folder = ospath.join(gettempdir(), 'Patch')
        return Dataset.__patches_images(input_folders, patch_folder, size, overlap)

    @staticmethod
    def test_spectras():
        here_path = ospath.dirname(__file__)
        input_folders = [ospath.normpath('{here}/data/spectroscopy/Patients.csv'.format(here=here_path))]
        return Dataset.__spectras(input_folders)

    @staticmethod
    def __images(folders):
        inputs = Inputs(folders=folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                        tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'},
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs

    @staticmethod
    def __multi_images(folders, extraction_folder, coefficients):
        inputs = Inputs(folders=folders, instance=Dataset(temporary=extraction_folder,
                                                          multi_coefficients=coefficients),
                        loader=Dataset.__scan_for_confocal_multi,
                        tags={'data': 'Multi_Path', 'label': 'Label', 'reference': 'Multi_Reference'},
                        encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                  'groups': LabelEncoder()})
        inputs.load()
        return inputs

    @staticmethod
    def __patches_images(folders, extraction_folder, size, overlap):
        parameters = {'Size': size,
                      'Overlap': overlap}
        inputs = Inputs(folders=folders, instance=Dataset(temporary=extraction_folder,
                                                          patch_parameters=parameters),
                        loader=Dataset.__scan_for_confocal_patches,
                        tags={'data': 'Patch_Path', 'label': 'Label', 'reference': 'Patch_Reference'},
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

    def __scan_for_confocal_multi(self, folder_path):
        # Browse data
        data_set = dermatology.Reader.scan_folder(folder_path)
        data_set = data_set[data_set.Modality == 'Microscopy']

        # Get into patches
        multis_data = []
        Dataset.__print_progress_bar(0, len(data_set), prefix='Progress:')
        for index, (df_index, data) in zip(np.arange(len(data_set.index)), data_set.iterrows()):
            Dataset.__print_progress_bar(index, len(data_set), prefix='Progress:')
            multi = Dataset.__multi_resolution(data['Full_path'], data['Reference'], self.multi_coefficients, self.temporary)
            multi = multi.merge(data_set, on='Reference')
            multis_data.append(multi)

        return pd.concat(multis_data, sort=False)

    @staticmethod
    def __multi_resolution(filename, reference, coefficients, multi_folder):
        if not ospath.exists(multi_folder):
            makedirs(multi_folder)

        image = Image.open(filename).convert('L')
        metas = []
        for index, coefficient in enumerate(coefficients):
            new_size = np.multiply(image.size, coefficient)

            # Create patch informations
            meta = dict()
            meta.update(
                {'Multi_Path': ospath.normpath(
                    '{ref}_{coef}.png'.format(ref=ospath.join(multi_folder, reference), coef=coefficient))})
            meta.update({'Multi_Index': index})
            meta.update({'Multi_Reference': '{ref}_{index}'.format(ref=reference, index=index)})
            meta.update({'Reference': reference})
            metas.append(meta)

            # Check if need to write patch
            if not ospath.isfile(meta['Multi_Path']):
                new_image = image.copy()
                new_image.thumbnail(size=new_size, resample=3)
                new_image.save(meta['Multi_Path'])

        return pd.DataFrame(metas)

    def __scan_for_confocal_patches(self, folder_path):
        # Browse data
        data_set = dermatology.Reader().scan_folder(folder_path)
        data_set = data_set[data_set.Modality == 'Microscopy']

        # Get into patches
        patches_data = []
        Dataset.__print_progress_bar(0, len(data_set), prefix='Progress:')
        for index, (df_index, data) in zip(np.arange(len(data_set.index)), data_set.iterrows()):
            Dataset.__print_progress_bar(index, len(data_set), prefix='Progress:')
            patches = Dataset.__patchify(data['Full_path'], data['Reference'], self.patch_parameters['Size'],
                                         self.patch_parameters['Overlap'], self.temporary)
            patches = patches.merge(data_set, on='Reference')
            patches_data.append(patches)

        return pd.concat(patches_data, sort=False)

    @staticmethod
    def __patchify(filename, reference, window_size, overlap, patch_folder):
        patch_folder = ospath.join(patch_folder, '{size}_{overlap}'.format(size=window_size, overlap=overlap))
        if not ospath.exists(patch_folder):
            makedirs(patch_folder)

        image = np.ascontiguousarray(np.array(Image.open(filename).convert('L')))
        stride = int(window_size - (window_size * overlap))  # Overlap of images
        image_shape = np.array(image.shape)
        window_shape = np.asanyarray((window_size, window_size))
        stride_shape = np.asanyarray((stride, stride))
        nbl = (image_shape - window_shape) // stride_shape + 1
        strides = np.r_[image.strides * stride_shape, image.strides]
        dims = np.r_[nbl, window_shape]
        patches = np.lib.stride_tricks.as_strided(image, strides=strides, shape=dims, writeable=False)
        patches = patches.reshape(-1, *window_shape)

        locations = np.ascontiguousarray(np.arange(0, image_shape[0]*image_shape[1]).reshape(image_shape))
        strides = np.r_[locations.strides * stride_shape, locations.strides]
        dims = np.r_[nbl, window_shape]
        patches_loc = np.lib.stride_tricks.as_strided(locations, strides=strides, shape=dims, writeable=False)
        patches_loc = patches_loc.reshape(-1, *window_shape)

        metas = []
        for index, (patch, location) in enumerate(zip(patches, patches_loc)):
            # Create patch informations
            meta = dict()
            meta.update(
                {'Patch_Path': ospath.normpath('{ref}_{id}.png'.format(ref=ospath.join(patch_folder, reference), id=index))})
            meta.update({'Patch_Index': index})
            start = location[0, 0]
            meta.update({'Patch_Start': (start%image_shape[0], start//image_shape[0])})

            end = location[-1, -1]
            meta.update({'Patch_End': (end%image_shape[0], end//image_shape[0])})
            meta.update({'Patch_Reference': '{ref}_{index}_{sta}_{end}'.format(ref=reference, index=index,
                                                                               sta=start, end=end)})
            meta.update({'Reference': reference})
            metas.append(meta)

            # Check if need to write patch
            if not ospath.isfile(meta['Patch_Path']):
                Image.fromarray(patch).save(meta['Patch_Path'])

        return pd.DataFrame(metas)

    @staticmethod
    def __print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print('\r{prefix} |{bar}| {percent} {suffix}\r'.format(prefix=prefix, bar=bar, percent=percent, suffix=suffix))
        # Print New Line on Complete
        if iteration == total:
            print()


class DefinedSettings:

    @staticmethod
    def get_default_orl():
        colors = dict(Cancer=(1, 0, 0), Precancer=(0.5, 0.5, 0), Sain=(0, 1, 0), Pathology=(0.75, 0.75, 0),
                      Rest=(0.25, 0.25, 0), Luck=(0, 0, 1))
        lines = {'Cancer': {'linestyle': '-'},
                 'Precancer': {'linestyle': '-.'},
                 'Sain': {'linestyle': ':'},
                 'Pathology': {'linestyle': ':'},
                 'Rest': {'linestyle': '-.'},
                 'Luck': {'linestyle': '--'}}
        return Settings({'colors': colors, 'lines': lines})

    @staticmethod
    def get_default_dermatology():
        colors = dict(Malignant=(1, 0, 0), Benign=(0.5, 0.5, 0), Normal=(0, 1, 0), Pathology=(0.75, 0.75, 0),
                      Rest=(0.25, 0.25, 0), Luck=(0, 0, 1))
        lines = {'Malignant': {'linestyle': '-'},
                 'Benign': {'linestyle': '-.'},
                 'Normal': {'linestyle': '-'},
                 'Pathology': {'linestyle': ':'},
                 'Rest': {'linestyle': '-.'},
                 'Luck': {'linestyle': '--'}}
        return Settings({'colors': colors, 'lines': lines})

from pathlib import Path
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from tempfile import gettempdir
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, GroupKFold

from toolbox.IO import dermatology, otorhinolaryngology
from toolbox.core.structures import Inputs, Spectra, Settings


class ORLDataset:

    @staticmethod
    def get_results_location(is_test=False):
        if is_test:
            base_folder = Path(gettempdir())
        else:
            base_folder = Path().home()
        result_folder = base_folder/'Results/ORL'
        if not result_folder.is_dir():
            result_folder.mkdir(parents=True)
        return result_folder

    @staticmethod
    def spectras():
        home_path = Path().home()
        location = home_path/'Data/Neck/'
        input_folders = [location/'Patients.csv', location/'Temoins.csv']
        return ORLDataset.__spectras(input_folders)

    @staticmethod
    def test_spectras():
        here_path = Path(__file__)
        input_folders = [here_path.parent/'../data_test/spectroscopy/Patients.csv']
        return ORLDataset.__spectras(input_folders)

    @staticmethod
    def __spectras(folders):
        inputs = Spectra(folders=folders, instance=otorhinolaryngology.Reader(),
                         loader=otorhinolaryngology.Reader.read_table,
                         tags={'data': 'data', 'label': 'label', 'group': 'Reference',
                               'reference': 'Reference_spectrum', 'group_label': 'pathologie'})
        inputs.load()
        return inputs


class DermatologyDataset:

    def __init__(self, work_folder=None, modality=None, patch_parameters=None, multi_coefficients=None):
        self.work_folder = work_folder
        self.modality = modality
        self.patch_parameters = patch_parameters
        self.multi_coefficients = multi_coefficients

    @staticmethod
    def get_results_location(is_test=False):
        if is_test:
            base_folder = Path(gettempdir())
        else:
            base_folder = Path().home()
        result_folder = base_folder/'Results/Dermatology'
        if not result_folder.is_dir():
            result_folder.mkdir(parents=True)
        return result_folder

    @staticmethod
    def images(modality=None):
        home_path = Path().home()
        input_folders = [home_path/'Data/Skin/Saint_Etienne/Patients']
        work_folder = home_path/'.research'
        if not work_folder.exists():
            work_folder.mkdir()
        return DermatologyDataset.__images(input_folders, work_folder, modality)

    @staticmethod
    def multiresolution(coefficients, modality=None):
        home_path = Path().home()
        input_folders = [home_path/'Data/Skin/Saint_Etienne/Patients']
        work_folder = home_path/'.research'
        if not work_folder.exists():
            work_folder.mkdir()
        return DermatologyDataset.__multi_images(input_folders, work_folder, coefficients, modality)

    @staticmethod
    def sliding_images(size, overlap, modality=None):
        home_path = Path().home()
        input_folders = [home_path/'Data/Skin/Saint_Etienne/Patients']
        work_folder = home_path/'.research'
        if not work_folder.exists():
            work_folder.mkdir()
        return DermatologyDataset.__sliding_images(input_folders, work_folder, size, overlap, modality)

    @staticmethod
    def test_images():
        here_path = Path(__file__)
        input_folders = [here_path.parent/'../data_test/dermatology/Test']
        work_folder = Path(gettempdir())/'.research'
        if not work_folder.exists():
            work_folder.mkdir()
        return DermatologyDataset.__images(input_folders, work_folder, 'Microscopy')

    @staticmethod
    def test_multiresolution(coefficients):
        here_path = Path(__file__)
        input_folders = [here_path.parent/'../data_test/dermatology/Test']
        work_folder = Path(gettempdir())/'.research'
        if not work_folder.exists():
            work_folder.mkdir()
        return DermatologyDataset.__multi_images(input_folders, work_folder, coefficients, 'Microscopy')

    @staticmethod
    def test_sliding_images(size, overlap):
        here_path = Path(__file__)
        input_folders = [here_path.parent/'../data_test/dermatology/Test']
        work_folder = Path(gettempdir())/'.research'
        if not work_folder.exists():
            work_folder.mkdir()
        return DermatologyDataset.__sliding_images(input_folders, work_folder, size, overlap, 'Microscopy')

    @staticmethod
    def __images(folders, work_folder, modality):
        inputs = Inputs(folders=folders, instance=DermatologyDataset(work_folder=work_folder, modality=modality),
                        loader=DermatologyDataset.__scan,
                        tags={'data': 'Full_Path', 'label': 'Label', 'reference': 'Reference', 'group': 'ID',
                              'group_label': 'Binary_Diagnosis'})
        inputs.load()
        inputs.set_temporary_folder(work_folder)
        return inputs

    @staticmethod
    def __multi_images(folders, work_folder, coefficients, modality):
        inputs = Inputs(folders=folders,
                        instance=DermatologyDataset(work_folder=work_folder, multi_coefficients=coefficients, modality=modality),
                        loader=DermatologyDataset.__scan_and_multi,
                        tags={'data': 'Full_Path', 'label': 'Label', 'reference': 'Reference', 'group': 'ID',
                              'group_label': 'Binary_Diagnosis'})
        inputs.load()
        inputs.set_temporary_folder(work_folder)
        return inputs

    @staticmethod
    def __sliding_images(folders, work_folder, size, overlap, modality):
        parameters = {'Size': size,
                      'Overlap': overlap}
        inputs = Inputs(folders=folders,
                        instance=DermatologyDataset(work_folder=work_folder, patch_parameters=parameters, modality=modality),
                        loader=DermatologyDataset.__scan_and_patchify,
                        tags={'data': 'Full_Path', 'label': 'Label', 'reference': 'Reference', 'group': 'ID',
                              'group_label': 'Binary_Diagnosis'})
        inputs.load()
        inputs.set_temporary_folder(work_folder)
        return inputs

    def __scan(self, folder_path):
        # Browse data
        return dermatology.Reader().scan_folder(folder_path, parameters={'patches': True,
                                                                         'modality': self.modality})

    def __scan_and_multi(self, folder_path):
        # Browse data
        data_set = dermatology.Reader().scan_folder(folder_path=folder_path, parameters={'patches': False,
                                                                                         'modality': self.modality})

        # Get into patches
        multis_data = []
        DermatologyDataset.__print_progress_bar(0, len(data_set), prefix='Progress:')
        for index, (df_index, data) in zip(np.arange(len(data_set.index)), data_set.iterrows()):
            DermatologyDataset.__print_progress_bar(index, len(data_set), prefix='Progress:')
            multi = DermatologyDataset.__multi_resolution(data['Full_Path'], data['Reference'], self.multi_coefficients,
                                                          self.work_folder)
            multi['Type'] = 'Multi'
            multi = pd.concat([multi, pd.DataFrame(data).T], sort=False)
            multi = multi.fillna(data)
            multis_data.append(multi)

        return pd.concat(multis_data, sort=False)

    def __scan_and_patchify(self, folder_path):
        # Browse data
        dataset = dermatology.Reader().scan_folder(folder_path, parameters={'patches': True,
                                                                            'modality': self.modality})
        images = dataset[dataset.Type == 'Full']
        patches = dataset[dataset.Type == 'Patch']

        # Get into patches
        windows_data = [patches]
        DermatologyDataset.__print_progress_bar(0, len(dataset), prefix='Progress:')
        for index, (df_index, data) in zip(np.arange(len(images.index)), images.iterrows()):
            DermatologyDataset.__print_progress_bar(index, len(images), prefix='Progress:')
            windows = DermatologyDataset.__patchify(data['Full_Path'], data['Reference'], self.patch_parameters['Size'],
                                                    self.patch_parameters['Overlap'], self.work_folder)
            windows['Type'] = 'Window'
            windows = pd.concat([windows, pd.DataFrame(data).T], sort=False)
            windows = windows.fillna(data)
            windows_data.append(windows)

        return pd.concat(windows_data, sort=False)

    @staticmethod
    def __multi_resolution(filename, reference, coefficients, work_folder):
        multi_folder = work_folder/'Multi'
        if not multi_folder.exists():
            multi_folder.mkdir()

        image = Image.open(filename).convert('L')
        metas = []
        for index, coefficient in enumerate(coefficients):
            new_size = np.multiply(image.size, coefficient)

            # Location of file
            filepath = multi_folder/'{ref}_{coef}.png'.format(ref=reference, coef=coefficient)

            # Create patch informations
            meta = dict()
            meta.update({'Full_Path': str(filepath)})
            meta.update({'Multi_Index': index})
            meta.update({'Coefficient': coefficient})
            meta.update({'Reference': '{ref}_{index}_M'.format(ref=reference, index=index)})
            meta.update({'Source': reference})
            metas.append(meta)

            # Check if need to write patch
            if not filepath.is_file():
                new_image = image.copy()
                new_image.thumbnail(size=new_size, resample=3)
                new_image.save(filepath)

        return pd.DataFrame(metas)

    @staticmethod
    def __patchify(filename, reference, window_size, overlap, work_folder):
        # Manage folder that contains root of patches
        work_folder = work_folder/'Patches'
        if not work_folder.exists():
            work_folder.mkdir()

        # Manage patches folder
        patch_folder = work_folder/'{size}_{overlap}'.format(size=window_size, overlap=overlap)
        if not patch_folder.exists():
            patch_folder.mkdir()

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

        locations = np.ascontiguousarray(np.arange(0, image_shape[0] * image_shape[1]).reshape(image_shape))
        strides = np.r_[locations.strides * stride_shape, locations.strides]
        dims = np.r_[nbl, window_shape]
        patches_loc = np.lib.stride_tricks.as_strided(locations, strides=strides, shape=dims, writeable=False)
        patches_loc = patches_loc.reshape(-1, *window_shape)

        metas = []
        for index, (patch, location) in enumerate(zip(patches, patches_loc)):

            # Location of file
            filepath = patch_folder/'{ref}_{id}.png'.format(ref=reference, id=index)

            # Create patch informations
            meta = dict()
            meta.update({'Full_Path': str(filepath)})
            meta.update({'Window_Index': int(index)})
            meta.update({'Label': 'Unknown'})
            start = location[0, 0]
            start = (start % image_shape[0], start // image_shape[0])
            end = location[-1, -1]
            end = (end % image_shape[0], end // image_shape[0])
            center = (int((start[0] + end[0]) / 2), int((start[1] + end[1]) / 2))
            meta.update({'Center_X': int(center[0])})
            meta.update({'Center_Y': int(center[1])})
            meta.update({'Height': int(window_size)})
            meta.update({'Width': int(window_size)})
            meta.update({'Reference': '{ref}_{index}_{x}_{y}'.format(ref=reference, index=index,
                                                                     x=center[0], y=center[1])})
            meta.update({'Source': reference})
            metas.append(meta)

            # Check if need to write patch
            if not filepath.is_file():
                Image.fromarray(patch).save(filepath)

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


class BuiltInSettings:

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


class LocalParameters:

    @staticmethod
    def get_cpu_number():
        return 4

    @staticmethod
    def get_dermatology_filters():
        return [('All', {'Label': ['Normal', 'Benign', 'Malignant', 'Unknown'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']},
                 ['Normal', 'Benign', 'Malignant'], {}),
                ('NvsP', {'Label': ['Normal', 'Pathology', 'Unknown'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']},
                 ['Normal', 'Pathology'], {'Label': (['Benign', 'Malignant'], 'Pathology')}),
                ('MvsR', {'Label': ['Rest', 'Malignant', 'Unknown'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']},
                 ['Rest', 'Malignant'], {'Label': (['Normal', 'Benign'], 'Rest')})]

    @staticmethod
    def get_orl_filters():
        return [('All', {'label': ['Sain', 'Precancer', 'Cancer']}, {}),
                ('NvsP', {'label': ['Sain', 'Pathology']}, {'label': (['Precancer', 'Cancer'], 'Pathology')}),
                ('MvsR', {'label': ['Rest', 'Cancer']}, {'label': (['Sain', 'Precancer'], 'Rest')})]

    @staticmethod
    def get_statistics_keys():
        return ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    @staticmethod
    def get_validation_test():
        return GroupKFold(n_splits=3), StratifiedKFold(n_splits=5)  # GroupKFold(n_splits=5)

    @staticmethod
    def set_gpu(percent_gpu=1, allow_growth=True):
        if not K.backend() == 'tensorflow':
            return

        # Change GPU usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_growth
        config.gpu_options.per_process_gpu_memory_fraction = percent_gpu
        set_session(tf.Session(config=config))

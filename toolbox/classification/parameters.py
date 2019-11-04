from pathlib import Path
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from tempfile import gettempdir
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, make_scorer
from toolbox.IO import otorhinolaryngology, dermatology


class ORL:

    @staticmethod
    def get_spectra(wavelength):
        home_path = Path().home()
        location = home_path / 'Data/Neck/'
        input_folders = [location / 'Patients.csv', location / 'Temoins.csv']
        return ORL.__spectra(input_folders, wavelength)


    @staticmethod
    def get_test_spectra():
        generator = otorhinolaryngology.Generator((4, 7), 30)
        return generator.generate_study()

    @staticmethod
    def __spectra(files, wavelength):
        spectra = otorhinolaryngology.Reader().read_table(files)
        spectra = otorhinolaryngology.Reader.change_wavelength(spectra, {'datum': 'Datum', 'wavelength': 'Wavelength'}, wavelength)
        spectra = otorhinolaryngology.Reader.remove_negative(spectra, {'datum': 'Datum'})
        return spectra

    @staticmethod
    def get_filters():
        return [('All', {'label': ['Sain', 'Precancer', 'Cancer']}, {}),
                ('NvsP', {'label': ['Sain', 'Pathology']}, {'label': (['Precancer', 'Cancer'], 'Pathology')}),
                ('MvsR', {'label': ['Rest', 'Cancer']}, {'label': (['Sain', 'Precancer'], 'Rest')})]

    @staticmethod
    def get_statistics_keys():
        return ['Pathology', 'Practitioner', 'Location', 'Label']


class Dermatology:

    @staticmethod
    def images(modality=None):
        home_path = Path().home()
        location = home_path / 'Data/Skin/'
        input_folders = [location / 'Elisa.csv', location / 'JeanLuc.csv']
        return Dermatology.__images(input_folders, modality)

    @staticmethod
    def multiple_resolution(coefficients, modality=None):
        home_path = Path().home()
        location = home_path / 'Data/Skin/'
        input_folders = [location / 'Elisa.csv', location / 'JeanLuc.csv']
        # Create temporary folder
        work_folder = home_path / '.research'
        work_folder.mkdir(exist_ok=True)
        return Dermatology.__multi_images(input_folders, work_folder, coefficients, modality)

    @staticmethod
    def sliding_images(size, overlap, modality=None):
        home_path = Path().home()
        location = home_path / 'Data/Skin/'
        input_folders = [location / 'Elisa.csv', location / 'JeanLuc.csv']
        # Create temporary folder
        work_folder = home_path / '.research'
        work_folder.mkdir(exist_ok=True)
        return Dermatology.__sliding_images(input_folders, work_folder, size, overlap, modality)

    @staticmethod
    def test_images(modality=None):
        work_folder = Path(gettempdir()) / '.research'
        work_folder.mkdir(exist_ok=True)
        return Dermatology.__images(None, modality)

    @staticmethod
    def test_multiresolution(coefficients, modality=None):
        work_folder = Path(gettempdir()) / '.research'
        work_folder.mkdir(exist_ok=True)
        return Dermatology.__multi_images(None, work_folder, coefficients, modality)

    @staticmethod
    def test_sliding_images(size, overlap, modality=None):
        work_folder = Path(gettempdir()) / '.research'
        work_folder.mkdir(exist_ok=True)
        return Dermatology.__sliding_images(None, work_folder, size, overlap, modality)

    @staticmethod
    def __images(folder, modality, patches=True):
        if folder is None:
            generator = dermatology.Generator((5, 10), 20)
            dataframe = generator.generate_study(patches=patches)
        else:
            dataframe = Dermatology.__scan(folder, patches=patches, modality=modality)
        return dataframe

    @staticmethod
    def __multi_images(folder, work_folder, coefficients, modality):
        dataframe = Dermatology.__images(folder, modality=modality, patches=False)
        return Dermatology.__to_multi(dataframe, coefficients, work_folder)

    @staticmethod
    def __sliding_images(folder, work_folder, size, overlap, modality):
        dataframe = Dermatology.__images(folder, modality=modality, patches=False)
        return Dermatology.__to_patch(dataframe, size, overlap, work_folder)

    @staticmethod
    def __scan(folder_path, patches=True, modality=None):
        # Browse data
        return dermatology.Reader().read_table(folder_path, parameters={'patches': patches, 'modality': modality})

    @staticmethod
    def __to_multi(dataframe, coefficients, work_folder):
        # Get into patches
        multis_data = []
        Dermatology.__print_progress_bar(0, len(dataframe), prefix='Progress:')
        for index, (df_index, data) in zip(np.arange(len(dataframe.index)), dataframe.iterrows()):
            Dermatology.__print_progress_bar(index, len(dataframe), prefix='Progress:')
            multi = Dermatology.__multi_resolution(data['Datum'], data['Reference'], coefficients, work_folder)
            multi['Type'] = 'Instance'
            multi = pd.concat([multi, pd.DataFrame(data).T], sort=False)
            multi = multi.fillna(data)
            multis_data.append(multi)

        return pd.concat(multis_data, sort=False)

    @staticmethod
    def __to_patch(dataframe, size, overlap, work_folder):
        # Browse data
        images = dataframe[dataframe.Type == 'Full']
        patches = dataframe[dataframe.Type == 'Patch']

        # Get into patches
        windows_data = [patches]
        Dermatology.__print_progress_bar(0, len(dataframe), prefix='Progress:')
        for index, (df_index, data) in zip(np.arange(len(images.index)), images.iterrows()):
            Dermatology.__print_progress_bar(index, len(images), prefix='Progress:')
            windows = Dermatology.__patchify(data['Datum'], data['Reference'], size, overlap, work_folder)
            windows['Type'] = 'Instance'
            windows = pd.concat([windows, pd.DataFrame(data).T], sort=False)
            windows = windows.fillna(data)
            windows_data.append(windows)

        return pd.concat(windows_data, sort=False)

    @staticmethod
    def __multi_resolution(filename, reference, coefficients, work_folder):
        multi_folder = work_folder / 'Multi'
        multi_folder.mkdir(exist_ok=True)

        image = Image.open(filename).convert('L')
        metas = []
        for index, coefficient in enumerate(coefficients):
            new_size = np.multiply(image.size, coefficient)

            # Location of file
            filepath = multi_folder / '{ref}_{coef}.png'.format(ref=reference, coef=coefficient)

            # Create patch informations
            meta = dict()
            meta.update({'Datum': str(filepath)})
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
        work_folder = work_folder / 'Patches'
        work_folder.mkdir(exist_ok=True)

        # Manage patches folder
        patch_folder = work_folder / '{size}_{overlap}'.format(size=window_size, overlap=overlap)
        patch_folder.mkdir(exist_ok=True)

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
            filepath = patch_folder / '{ref}_{id}.png'.format(ref=reference, id=index)

            # Create patch informations
            meta = dict()
            meta.update({'Datum': str(filepath)})
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

    @staticmethod
    def get_statistics_keys():
        return ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    @staticmethod
    def get_filters():
        return [('All', {'Label': ['Normal', 'Benign', 'Malignant', 'Unknown']},
                 ['Normal', 'Benign', 'Malignant'], {}),
                ('NvsP', {'Label': ['Normal', 'Pathology', 'Unknown']},
                 ['Normal', 'Pathology'], {'Label': (['Benign', 'Malignant'], 'Pathology')}),
                ('MvsR', {'Label': ['Rest', 'Malignant', 'Unknown']},
                 ['Rest', 'Malignant'], {'Label': (['Normal', 'Benign'], 'Rest')})]


class Settings:

    def __init__(self, data=None):
        if data is None:
            self.data = {}
        else:
            self.data = data

    def get_color(self, key):
        default_color = (0, 0, 0)
        colors = self.data.get('colors', None)
        if colors is None:
            return default_color
        return colors.get(key, default_color)

    def get_line(self, key):
        lines = self.data.get('lines', None)
        if lines is None:
            return None
        return lines.get(key, None)

    def is_in_data(self, check):
        for key, value in check.items():
            if key not in self.data or self.data[key] not in value:
                return False
        return True

    def update(self, data):
        self.data.update(data)

    @staticmethod
    def get_default_orl():
        colors = dict(Cancer=(0.93, 0.5, 0.2), Precancer=(1, 0.75, 0), Sain=(0.44, 0.68, 0.28),
                      Pathological=(0.75, 0.75, 0), Rest=(0.90, 0.47, 0), Luck=(0, 0, 1))
        lines = {'Cancer': {'linestyle': '-'},
                 'Precancer': {'linestyle': '-.'},
                 'Sain': {'linestyle': ':'},
                 'Pathological': {'linestyle': ':'},
                 'Rest': {'linestyle': '-.'},
                 'Luck': {'linestyle': '--'}}
        return Settings({'colors': colors, 'lines': lines})

    @staticmethod
    def get_default_dermatology():
        colors = dict(Malignant=(0.93, 0.5, 0.2), Benign=(1, 0.75, 0), Normal=(0.44, 0.68, 0.28),
                      Pathological=(0.75, 0.75, 0), Rest=(0.90, 0.47, 0), Luck=(0, 0, 1), Draw=(1, 0.25, 0.25))
        lines = {'Malignant': {'linestyle': '-'},
                 'Benign': {'linestyle': '-.'},
                 'Normal': {'linestyle': '-'},
                 'Pathological': {'linestyle': ':'},
                 'Rest': {'linestyle': '-.'},
                 'Luck': {'linestyle': '--'}}
        return Settings({'colors': colors, 'lines': lines})


class LocalParameters:

    @staticmethod
    def get_result_dir(is_test=False):
        if is_test:
            base_folder = Path(gettempdir())
        else:
            base_folder = Path().home()
        result_folder = base_folder / 'Results'
        result_folder.mkdir(parents=True, exist_ok=True)
        return result_folder

    @staticmethod
    def get_temp_dir(is_test=False):
        if is_test:
            base_folder = Path(gettempdir())
        else:
            base_folder = Path().home()
        work_folder = base_folder / '.research'
        work_folder.mkdir(exist_ok=True)
        return work_folder

    @staticmethod
    def get_scorer():
        return make_scorer(lambda yt, yp: f1_score(yt, yp, average='weighted'))

    @staticmethod
    def set_gpu(percent_gpu=1, allow_growth=True):
        if not K.backend() == 'tensorflow':
            return

        # Change GPU usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = allow_growth
        config.gpu_options.per_process_gpu_memory_fraction = percent_gpu
        set_session(tf.Session(config=config))

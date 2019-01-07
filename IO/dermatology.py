import glob
import pyocr
import shutil
from os import listdir, makedirs, path
from os.path import isdir, join
from PIL import Image
from numpy import genfromtxt, asarray, savetxt
from pyocr import builders
from core.inputs import Data, DataSet


class Reader:

    def __init__(self, delimiter):
        """Make an initialisation of SpectrumReader object.

        Take a string that represent delimiter

        Args:
             delimiter (:obj:'str'): The delimiter string.
        """
        self.delimiter = delimiter

    def __read_images_file(self, parent_folder, subdir):

        # Patient file
        images_file = join(parent_folder, subdir, 'images.csv')

        # Read csv
        csv = genfromtxt(images_file, dtype='str', delimiter=self.delimiter)

        # Build spectrum
        images = []
        for row in range(1, csv.shape[0]):
            meta = {'patient_name': subdir,
                    'modality': csv[row, 0]}
            image = Data(data=join(parent_folder, csv[row, 1]),meta=meta)
            images.append(image)
        return images

    def __read_patient_file(self, folder_path):
        # Patient file
        patient_file = join(folder_path, 'patient.csv')

        # Read csv
        csv = genfromtxt(patient_file, dtype='str', delimiter=self.delimiter)

        # Size of csv meta
        if len(csv.shape) == 1:
            return {csv[0]: csv[1]}

        csv_size = csv.shape[1]

        # Build spectrum
        dict = {}
        for col in range(0, csv_size):
            dict[csv[0, col]] = csv[1, col]

        return dict

    def scan_folder(self, folder_path):
        # Subdirectories
        subdirs = [name for name in listdir(folder_path) if isdir(join(folder_path, name))]

        # Browse subdirectories
        datas = []
        for subdir in subdirs:
            patient_datas=[]
            try:
                meta = self.__read_patient_file(join(folder_path, subdir))
                patient_datas=self.__read_images_file(folder_path, subdir)
                # Update all meta data
                [data.meta.update(meta) for data in patient_datas]
            except OSError:
                print('Patient {}'.format(subdir))
            datas.extend(patient_datas)

        return DataSet(datas)


class ConfocalBuilder(builders.TextBuilder):

    def __init__(self):
        super().__init__()
        self.tesseract_configs += ["-c", "tessedit_char_whitelist=0123456789.-um"]


class DataManager:

    @staticmethod
    def launch_converter(root_dir, out_dir):

        # In resources
        table_file = path.join(root_dir, 'Table.csv')
        rcm_file = path.join(root_dir, 'RCM.csv')
        microscopy_dir = path.join(root_dir, 'Microscopy')
        dermoscopy_dir = path.join(root_dir, 'Dermoscopy')
        photography_dir = path.join(root_dir, 'Photography')

        # Create output dir
        if not path.exists(out_dir):
            makedirs(out_dir)

        # Read csv
        table_csv = genfromtxt(table_file, dtype='str', delimiter=';')
        rcm_csv = genfromtxt(rcm_file, dtype='str', delimiter=';')
        patient_length = table_csv.shape[0]
        # Print progress bar
        DataManager.print_progress_bar(0, patient_length, prefix='Progress:')
        for index in range(1, patient_length):
            # Print progress bar
            DataManager.print_progress_bar(index, patient_length, prefix='Progress:')

            # Store current line
            row = table_csv[index, :]

            # Construct folder reference
            outSubDir = path.join(out_dir, str(index))

            # Create folder if necessary
            if not path.exists(outSubDir):
                makedirs(outSubDir)

            # Save metadata
            out_patient = asarray(
                [['Sex', 'Age', 'Area', 'Diagnosis', 'Malignant'], [row[5], row[2], row[6], row[10], row[9]]])
            savetxt(path.join(outSubDir, 'patient.csv'), out_patient, fmt='%s', delimiter=';')

            # Get photography files
            out_images = [['Modality', 'Path', 'Depth(um)']]
            dir_modality = path.join(path.join(outSubDir, 'Photography'))
            if not path.exists(dir_modality):
                makedirs(dir_modality)
            files = glob.glob(photography_dir + "\\" + str(row[1]) + " (*.jpg", recursive=True)
            for file in files:
                new_path = path.join(dir_modality, path.basename(file))
                shutil.copy(file, new_path)
                out_images.append(['Photography', path.relpath(new_path, out_dir), 'NaN'])

            # Get dermoscopy files
            dir_modality = path.join(path.join(outSubDir, 'Dermoscopy'))
            if not path.exists(dir_modality):
                makedirs(dir_modality)
            files = glob.glob(dermoscopy_dir + "\\" + str(row[1]) + " (*.jpg", recursive=True)
            for file in files:
                new_path = path.join(dir_modality, path.basename(file))
                shutil.copy(file, new_path)
                out_images.append(['Dermoscopy', path.relpath(new_path, out_dir), 'NaN'])

            # Get microscopy files
            dir_modality = path.join(path.join(outSubDir, 'Microscopy'))
            if not path.exists(dir_modality):
                makedirs(dir_modality)
            files = glob.glob(microscopy_dir + "\\" + str(row[0]) + "\\**\\*.bmp", recursive=True)
            for file in files:
                new_path = path.join(dir_modality, path.basename(file))
                raw_image = Image.open(file)
                width = raw_image.size[0]
                height = raw_image.size[1]
                if height == 1000:  # Non-OCR version
                    image = raw_image
                    digits = '0.0'
                else:  # OCR version
                    digits = DataManager.read_ocr(raw_image.crop((18, height - 25, 100, height)))
                    image = raw_image.crop((0, 0, width, height - 45))

                image.save(new_path, "BMP")
                out_images.append(['Microscopy', path.relpath(new_path, out_dir), digits])

            out_images = asarray(out_images)
            savetxt(path.join(outSubDir, 'images.csv'), out_images, fmt='%s', delimiter=';')

    @staticmethod
    def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
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
    def read_ocr(image):
        # Find OCR tool
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("No OCR tool found")
        tool = tools[0]

        langs = tool.get_available_languages()
        lang = langs[0]

        w = image.size[0]
        h = image.size[1]
        im_OCR = image.resize((w * 20, h * 20), Image.BILINEAR)
        digits = tool.image_to_string(
            im_OCR,
            lang=lang,
            builder=ConfocalBuilder()
        )
        digits = digits.replace(' ', '')
        digits = digits.replace('um', '')
        return digits

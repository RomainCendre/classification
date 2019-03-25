from glob import glob
import pandas as pd
import pyocr
import shutil
from os import listdir, makedirs, path
from os.path import isdir, join, normpath, isfile, basename, splitext
from PIL import Image
from numpy.ma import array
from pyocr import builders


class Reader:

    def __init__(self, temp_folder=None):
        self.temp_folder = temp_folder

    def scan_folder(self, folder_path):
        # Subdirectories
        subdirs = [name for name in listdir(folder_path) if isdir(join(folder_path, name))]

        # Browse subdirectories
        datas = []
        for subdir in subdirs:
            try:
                # Read patient and images data
                meta_patient = Reader.__read_patient_file(join(folder_path, subdir))
                patient_datas = Reader.__read_images_file(folder_path, subdir)
                # Merge both
                patient_datas['ID'] = meta_patient['ID'][0]
                patient_datas = patient_datas.merge(meta_patient)
                patient_datas['Reference'] = patient_datas.apply(
                    lambda row: '{patient}_{image}'.format(patient=row['ID'], image=row['ID_Image']), axis=1)
                datas.append(patient_datas)
            except OSError:
                print('Patient {}'.format(subdir))

        return pd.concat(datas)

    def scan_folder_for_images(self, folder_path):
        # Subdirectories
        sub_dirs = [name for name in listdir(folder_path) if isdir(join(folder_path, name))]

        # Browse subdirectories
        data_set = []
        for subdir in sub_dirs:
            sub_folder = join(folder_path, subdir)
            sub_files = glob(join(sub_folder, '*.bmp'))

            for sub_file in sub_files:
                reference = '{label}-{file}'.format(label=subdir, file=splitext(basename(sub_file))[0])
                data_set.append({'Full_path': sub_file,
                                 'Label': subdir,
                                 'Reference': reference})
        return pd.DataFrame(data_set)

    def scan_folder_for_patches(self, folder_path, patch_size=250):
        # Browse data
        data_set = self.scan_folder(folder_path)
        data_set = data_set[data_set.Modality == 'Microscopy']

        # Get into patches
        patches_data = []
        DataManager.print_progress_bar(0, len(data_set), prefix='Progress:')
        for index, data in data_set.iterrows():
            DataManager.print_progress_bar(index, len(data_set), prefix='Progress:')
            patches = Reader.__get_patches(data['Full_path'], data['Reference'], patch_size, self.temp_folder)
            patches = patches.merge(data_set, on='Reference')
            patches_data.append(patches)

        return pd.concat(patches_data)

    @staticmethod
    def __get_patches(filename, reference, patch_size, temp_folder):
        X = array(Image.open(filename).convert('L'))
        patches = []
        patch_index = 0
        for r in range(0, X.shape[0] - patch_size + 1, patch_size):
            for c in range(0, X.shape[1] - patch_size + 1, patch_size):
                # Create patch informations
                patch = dict()
                patch.update(
                    {'Patch_Path': normpath('{reference}_{size}_{id}.png'.format(reference=join(temp_folder, reference),
                                                                           size=patch_size, id=patch_index))})
                patch.update({'Patch_Index': patch_index})
                patch.update({'Patch_Position_X': r})
                patch.update({'Patch_Position_Y': c})
                patch.update({'Patch_Reference': '{ref}_{index}'.format(ref=reference, index=patch_index)})
                patch.update({'Reference': reference})
                patch_index += 1
                patches.append(patch)

                # Check if need to write patch
                if not isfile(patch['Patch_Path']):
                    Image.fromarray(X[r:r + patch_size, c:c + patch_size]).save(patch['Patch_Path'])

        return pd.DataFrame(patches)

    @staticmethod
    def __read_images_file(parent_folder, subdir):
        # Patient file
        images_file = join(parent_folder, subdir, 'images.csv')

        # Read csv and add tag for path
        images = pd.read_csv(images_file, dtype=str)
        images['Full_path'] = images.apply(lambda row: join(parent_folder, subdir, row['Modality'], row['Path']),
                                           axis=1)
        return images

    @staticmethod
    def __read_patient_file(folder_path):
        # Patient file
        patient_file = join(folder_path, 'patient.csv')
        return pd.read_csv(patient_file, dtype=str)


class ConfocalBuilder(builders.TextBuilder):

    def __init__(self):
        super().__init__()
        self.tesseract_configs += ["-c", "tessedit_char_whitelist=0123456789.-um"]


class DataManager:

    def __init__(self, root_folder):
        self.table_file = path.join(root_folder, 'Table.csv')
        self.rcm_file = path.join(root_folder, 'RCM.csv')
        self.microscopy_folder = path.join(root_folder, 'Microscopy')
        self.dermoscopy_folder = path.join(root_folder, 'Dermoscopy')
        self.photography_folder = path.join(root_folder, 'Photography')
        self.labels = ['Malignant', 'Benign', 'Normal', 'Doubtful', 'Draw']

    def compute_dermoscopy(self, source_id, destination):
        destination_folder = path.join(destination, 'Dermoscopy')
        if not path.exists(destination_folder):
            makedirs(destination_folder)

        images = []
        folder = path.join(self.dermoscopy_folder, source_id)
        files = glob(folder + " (*.jpg", recursive=True)
        for source_file in files:
            image = Image.open(source_file)
            width, height = image.size
            destination_file = path.join(destination_folder, path.basename(source_file))
            shutil.copy(source_file, destination_file)
            images.append({'Modality': 'Dermoscopy',
                           'Path': path.relpath(destination_file, destination_folder),
                           'Height': height,
                           'Width': width})

        return images

    def compute_photography(self, source_id, destination):
        destination_folder = path.join(destination, 'Photography')
        if not path.exists(destination_folder):
            makedirs(destination_folder)

        images = []
        folder = path.join(self.photography_folder, source_id)
        files = glob(folder + " (*.jpg", recursive=True)
        for source_file in files:
            image = Image.open(source_file)
            width, height = image.size
            destination_file = path.join(destination_folder, path.basename(source_file))
            shutil.copy(source_file, destination_file)
            images.append({'Modality': 'Photography',
                           'Path': path.relpath(destination_file, destination_folder),
                           'Height': height,
                           'Width': width})

        return images

    def compute_microscopy(self, source_id, destination):
        # Read microscopy file for each patient
        rcm_data = pd.read_csv(self.rcm_file, dtype=str)
        # Folder where are send new data
        destination_folder = path.join(destination, 'Microscopy')
        if not path.exists(destination_folder):
            makedirs(destination_folder)

        images = []
        microscopy_labels = rcm_data[rcm_data['ID_RCM'] == source_id]
        microscopy_folder = path.join(self.microscopy_folder, source_id)
        for ind, row_label in microscopy_labels.iterrows():
            if pd.isna(row_label['Folder']):
                microscopy_subfolder = microscopy_folder
                destination_subfolder = destination_folder
            else:
                microscopy_subfolder = path.join(microscopy_folder, row_label['Folder'])
                destination_subfolder = path.join(destination_folder, row_label['Folder'])
                if not path.exists(destination_subfolder):
                    makedirs(destination_subfolder)

            # Browse different labels...
            for label in self.labels:
                if pd.isna(row_label[label]):
                    continue
                images_refs = DataManager.ref_to_images(row_label[label])

                # .. then images
                for images_ref in images_refs:
                    # Construct source and destination file path
                    source_file = path.join(microscopy_subfolder, images_ref)
                    if not path.isfile(source_file):
                        print("Not existing:" + source_file)
                        continue
                    destination_file = path.join(destination_subfolder, images_ref)

                    # Open image
                    raw_image = Image.open(source_file)
                    width = raw_image.size[0]
                    height = raw_image.size[1]

                    # Try to read optical informations
                    if height == 1000:  # Non-OCR version
                        image = raw_image
                        digits = '0.0'
                    else:  # OCR version
                        # digits = DataManager.read_ocr(raw_image.crop((18, height - 25, 100, height)))
                        digits = '0.0'
                        image = raw_image.crop((0, 0, width, height - 45))

                    image.save(destination_file, "BMP")
                    width, height = image.size
                    images.append({'Modality': 'Microscopy',
                                   'Path': path.relpath(destination_file, destination_folder),
                                   'Label': label,
                                   'Depth(um)': digits,
                                   'Height': height,
                                   'Width': width})
        return images

    def launch_converter(self, output_folder, excluded_meta):
        # Create output dir
        if not path.exists(output_folder):
            makedirs(output_folder)

        id_modality = ['ID_Dermoscopy', 'ID_RCM']
        # Read csv
        table = pd.read_csv(self.table_file, dtype=str)
        table = table.drop(excluded_meta, axis=1)
        nb_patients = table.shape[0]

        # Print progress bar
        DataManager.print_progress_bar(0, nb_patients, prefix='Progress:')

        # Iterate table
        for index, row in table.iterrows():
            # Print progress bar
            DataManager.print_progress_bar(index, nb_patients, prefix='Progress:')

            # Construct folder reference
            out_patient_folder = path.join(output_folder, str(index))

            # Create folder if necessary
            if not path.exists(out_patient_folder):
                makedirs(out_patient_folder)

            # Write patient meta
            pd.DataFrame([row.drop(id_modality, errors='ignore').to_dict()]).to_csv(
                path.join(out_patient_folder, 'patient.csv'), index=False)

            images = []

            if 'ID_Dermoscopy' in row.index:
                # Get photography files
                images.extend(self.compute_photography(row['ID_Dermoscopy'], out_patient_folder))
                # Get dermoscopy files
                images.extend(self.compute_dermoscopy(row['ID_Dermoscopy'], out_patient_folder))

            if 'ID_RCM' in row.index:
                # Get microscopy files
                images.extend(self.compute_microscopy(row['ID_RCM'], out_patient_folder))

            # Write images list
            dataframe = pd.DataFrame(images)
            dataframe.index.name = 'ID_Image'
            dataframe.to_csv(path.join(out_patient_folder, 'images.csv'))

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

    @staticmethod
    def ref_to_images(references):
        images = []
        references = references.split(';')
        for reference in references:
            if '-' in reference:
                refs = reference.split('-')
                images.extend(range(int(refs[0]), int(refs[1]) + 1))
            else:
                images.append(int(reference))
        return ['v{number}.bmp'.format(number=str(image).zfill(7)) for image in images]

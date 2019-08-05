import numpy as np
import pandas as pandas
import pyocr
from PIL import Image
from pyocr import builders
from pathlib import Path
from natsort import index_natsorted, order_by_index


class Reader:

    def scan_folder(self, folder_path, parameters={}):
        # Go through pathlib
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError('{folder} not a folder'.format(folder=folder_path))

        # Browse subdirectories
        datas = []
        for subdir in folder_path.iterdir():
            try:
                datas.append(self.scan_subfolder(subdir, parameters))
            except OSError:
                print('Patient {}'.format(subdir))

        return pandas.concat(datas, sort=False, ignore_index=True).drop(columns='Path')

    @staticmethod
    def scan_subfolder(subdir, parameters={}):
        # Read patient and images data
        metas = Reader.read_patient_file(subdir)
        metas = metas.drop(columns='ID_JLP', errors='ignore')
        images = Reader.read_images_file(subdir, modality=parameters.get('modality', None))
        images = images.drop(columns='Depth(um)', errors='ignore')

        # Patch filter
        if parameters.get('patches', True):
            patches = Reader.read_patches_file(subdir, modality=parameters.get('modality', None))
        else:
            patches = None

        # Merge both
        images['ID'] = metas['ID'][0]
        images = images.merge(metas)
        images['Reference'] = images.apply(
            lambda row: '{patient}_{image}_F'.format(patient=row['ID'], image=row.name), axis=1)
        images['Source'] = images['Reference']

        if patches is not None:
            patches['ID'] = metas['ID'][0]
            patches = patches.merge(metas)
            patches['Reference'] = patches.apply(
                lambda row: '{patient}_{image}_P'.format(patient=row['ID'], image=row.name), axis=1)
            patches['Source'] = patches.apply(lambda row: images[images.Path == row['Source']]['Reference'].iloc[0],
                                              axis=1)
        return pandas.concat([images, patches], sort=False)

    @staticmethod
    def read_images_file(subdir, modality=None):
        # Patient file
        images_file = subdir/'images.csv'
        # Folder name
        folder_name = subdir.name
        # Read csv and add tag for path
        images = pandas.read_csv(images_file, dtype=str)
        if len(images) == 0:
            return None
        images = images.reindex(index=order_by_index(images.index, index_natsorted(images.Path)))
        images['Full_Path'] = images.apply(lambda row: str(subdir/row['Modality']/row['Path']), axis=1)
        images['Type'] = 'Full'
        images['Reference2'] = images.apply(
            lambda row: Reader.filter_str('{id}_{image}'.format(id=folder_name, image=row['Path'])), axis=1)
        if modality is not None:
            images = images[images.Modality == modality]
        return images

    @staticmethod
    def read_patches_file(subdir, modality=None):
        # Patient file
        patch_file = subdir/'patches.csv'
        if not patch_file.is_file():
            return None

        # Read csv and add tag for path
        patches = pandas.read_csv(patch_file, dtype=str)
        if len(patches) == 0:
            return None
        # Customize patch table
        patches['Full_Path'] = patches.apply(lambda row: str(subdir/'patches'/row['Path']), axis=1)
        patches['Type'] = 'Patch'
        if modality is not None:
            patches = patches[patches.Modality == modality]
        return patches

    @staticmethod
    def read_patient_file(folder_path):
        # Patient file
        return pandas.read_csv(folder_path/'patient.csv', dtype=str)

    @staticmethod
    def filter_str(my_string):
        return ''.join([c if c.isalnum() else '_' for c in my_string])


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


class ConfocalBuilder(builders.TextBuilder):

    def __init__(self):
        super().__init__()
        self.tesseract_configs += ["-c", "tessedit_char_whitelist=0123456789.-um"]


class DataManager:

    def __init__(self, root_folder):
        # Go through pathlib
        root_folder = Path(root_folder)
        if not root_folder.is_dir():
            raise ValueError('{folder} not a folder'.format(folder=root_folder))

        self.table_file = root_folder/'Table.csv'
        self.rcm_file = root_folder/'RCM.csv'
        self.microscopy_folder = root_folder/'Microscopy'
        self.dermoscopy_folder = root_folder/'Dermoscopy'
        self.photography_folder = root_folder/'Photography'
        self.labels = ['Malignant', 'Benign', 'Normal', 'Doubtful', 'Draw']

    def compute_dermoscopy(self, source_id, label, output_folder):
        # Check output folder
        output_folder = Path(output_folder)
        output_folder = output_folder/'Dermoscopy'
        output_folder.mkdir(exist_ok=True)

        # Browse source files
        images = []
        for source_file in self.dermoscopy_folder.glob('{id} (*.jpg'.format(id=source_id)):
            image = Image.open(source_file)
            width, height = image.size
            output_file = output_folder/'{base}.bmp'.format(base=source_file.stem)
            raw_image = Image.open(source_file)
            raw_image.save(output_file, "BMP")
            images.append({'Modality': 'Dermoscopy',
                           'Path': output_file.relative_to(output_folder).as_posix(),
                           'Label': label,
                           'Height': height,
                           'Width': width})
        return images

    def compute_photography(self, source_id, label, output_folder):
        # Check output folder
        output_folder = Path(output_folder)
        output_folder = output_folder/'Photography'
        output_folder.mkdir(exist_ok=True)

        # Browse source files
        images = []
        for source_file in self.photography_folder.glob('{id} (*.jpg'.format(id=source_id)):
            image = Image.open(source_file)
            width, height = image.size
            output_file = output_folder/'{base}.bmp'.format(base=source_file.stem)
            raw_image = Image.open(source_file)
            raw_image.save(output_file, "BMP")
            images.append({'Modality': 'Photography',
                           'Path': output_file.relative_to(output_folder).as_posix(),
                           'Label': label,
                           'Height': height,
                           'Width': width})
        return images

    def compute_microscopy(self, source_id, output_folder):
        # Check output folder
        output_folder = Path(output_folder)
        output_folder = output_folder/'Microscopy'
        output_folder.mkdir(exist_ok=True)

        # Read microscopy file for each patient
        rcm_data = pandas.read_csv(self.rcm_file, dtype=str)

        images = []
        microscopy_labels = rcm_data[rcm_data['ID_RCM'] == source_id]
        microscopy_folder = self.microscopy_folder/source_id
        # Get all microscopy location
        remains_images = [str(file.relative_to(microscopy_folder)) for file in microscopy_folder.glob('**/*.bmp')]
        sorted_images = {'Draw': []}
        # Identify images and their label
        for ind, row_label in microscopy_labels.iterrows():
            # Browse different labels
            for label in self.labels:# If label doesn't contains images
                if pandas.isna(row_label[label]):
                    continue
                referenced_images = DataManager.ref_to_images(row_label[label])
                sorted_images[label] = referenced_images
                remains_images = [item for item in remains_images if item not in referenced_images]
        sorted_images['Draw'].extend(remains_images)
        # Now browse all categories
        for label, images_references in sorted_images.items():
            for image_reference in images_references:
                # Construct source and destination file path
                source_file = microscopy_folder/image_reference
                if not source_file.is_file():
                    print('Not existing {source}'.format(source=source_file))
                    continue
                output_file = output_folder/image_reference
                output_file.parent.mkdir(parents=True, exist_ok=True)

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

                image.save(output_file, "BMP")
                width, height = image.size
                images.append({'Modality': 'Microscopy',
                               'Path': output_file.relative_to(output_folder).as_posix(),
                               'Label': label,
                               'Depth(um)': digits,
                               'Height': height,
                               'Width': width})
        return images

    def launch_converter(self, output_folder, excluded_meta):
        # Check output folder
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        id_modality = ['ID_Dermoscopy', 'ID_RCM']
        # Read csv
        table = pandas.read_csv(self.table_file, dtype=str)
        table = table.drop(excluded_meta, axis=1)
        nb_patients = table.shape[0]

        # Print progress bar
        DataManager.print_progress_bar(0, nb_patients, prefix='Progress:')

        # Iterate table
        for index, row in table.iterrows():
            # Print progress bar
            DataManager.print_progress_bar(index, nb_patients, prefix='Progress:')

            # Construct folder reference
            output_patient = output_folder/row['ID']
            output_patient.mkdir(exist_ok=True)

            # Write patient meta
            pandas.DataFrame([row.drop(id_modality, errors='ignore').to_dict()]).to_csv(output_patient/'patient.csv', index=False)

            images = []
            if 'ID_Dermoscopy' in row.index:
                # Get photography files
                images.extend(self.compute_photography(row['ID_Dermoscopy'], row['Binary_Diagnosis'], output_patient))
                # Get dermoscopy files
                images.extend(self.compute_dermoscopy(row['ID_Dermoscopy'], row['Binary_Diagnosis'], output_patient))

            if 'ID_RCM' in row.index:
                # Get microscopy files
                images.extend(self.compute_microscopy(row['ID_RCM'], output_patient))

            # Write images list
            dataframe = pandas.DataFrame(images)
            dataframe.to_csv(output_patient/'images.csv', index=False)

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

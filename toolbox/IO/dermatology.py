import pandas as pd
import pyocr
from PIL import Image
from pyocr import builders
from pathlib import Path


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
                # Read patient and images data
                metas = Reader.__read_patient_file(subdir)
                metas = metas.drop(columns='ID_JLP', errors='ignore')
                images = Reader.__read_images_file(subdir)
                images = images.drop(columns='Depth(um)', errors='ignore')

                # Patch filter
                if parameters.get('patches', True):
                    patches = Reader.__read_patches_file(subdir)
                else:
                    patches = None

                # Modality filter
                modality = parameters.get('modality', None)

                # Merge both
                images['ID'] = metas['ID'][0]
                images = images.merge(metas)
                images['Reference'] = images.apply(
                    lambda row: '{patient}_{image}_F'.format(patient=row['ID'], image=row.name), axis=1)
                images['Source'] = images['Reference']
                # Filter images
                if modality is not None:
                    images = images[images.Modality == modality]

                if patches is not None:
                    patches['ID'] = metas['ID'][0]
                    patches = patches.merge(metas)
                    patches['Reference'] = patches.apply(
                        lambda row: '{patient}_{image}_P'.format(patient=row['ID'], image=row.name), axis=1)
                    patches['Source'] = patches.apply(lambda row: images[images.Path == row['Source']]['Reference'].iloc[0], axis=1)
                    # Filter patches
                    if modality is not None:
                        patches = patches[patches.Modality == modality]

                datas.append(pd.concat([images, patches], sort=False))
            except OSError:
                print('Patient {}'.format(subdir))

        return pd.concat(datas, sort=False, ignore_index=True).drop(columns='Path')

    @staticmethod
    def __read_images_file(subdir):
        # Patient file
        images_file = subdir/'images.csv'

        # Read csv and add tag for path
        images = pd.read_csv(images_file, dtype=str)
        images['Full_Path'] = images.apply(lambda row: str(subdir/row['Modality']/row['Path']), axis=1)
        images['Type'] = 'Full'
        return images

    @staticmethod
    def __read_patches_file(subdir):
        # Patient file
        patch_file = subdir/'patches.csv'
        if not patch_file.is_file():
            return None

        # Read csv and add tag for path
        patches = pd.read_csv(patch_file, dtype=str)
        patches['Full_Path'] = patches.apply(lambda row: str(subdir/'patches'/row['Path']), axis=1)
        patches['Type'] = 'Patch'
        return patches

    @staticmethod
    def __read_patient_file(folder_path):
        # Patient file
        return pd.read_csv(folder_path/'patient.csv', dtype=str)


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
        rcm_data = pd.read_csv(self.rcm_file, dtype=str)

        images = []
        microscopy_labels = rcm_data[rcm_data['ID_RCM'] == source_id]
        microscopy_folder = self.microscopy_folder/source_id
        for ind, row_label in microscopy_labels.iterrows():
            # Prepare if needed sub folders
            if pd.isna(row_label['Folder']):
                microscopy_subfolder = microscopy_folder
                output_subfolder = output_folder
            else:
                microscopy_subfolder = microscopy_folder/row_label['Folder']
                output_subfolder = output_folder/row_label['Folder']
                output_subfolder.mkdir(exist_ok=True)

            # Browse different labels...
            for label in self.labels:
                # If label doesn't contains images
                if pd.isna(row_label[label]):
                    continue

                images_refs = DataManager.ref_to_images(row_label[label])
                # .. then images
                for images_ref in images_refs:
                    # Construct source and destination file path
                    source_file = microscopy_subfolder/images_ref
                    if not source_file.is_file():
                        print('Not existing {source}'.format(source=source_file))
                        continue
                    output_file = output_subfolder/images_ref

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
            output_patient = output_folder/row['ID']
            output_patient.mkdir(exist_ok=True)

            # Write patient meta
            pd.DataFrame([row.drop(id_modality, errors='ignore').to_dict()]).to_csv(output_patient/'patient.csv', index=False)

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
            dataframe = pd.DataFrame(images)
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

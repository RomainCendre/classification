import pandas as pandas
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

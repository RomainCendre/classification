import pandas as pandas
from pathlib import Path


class Reader:

    images_types = ['png', 'bmp']

    def scan_folder(self, folder_path):
        # Go through pathlib
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError('{folder} not a folder'.format(folder=folder_path))

        # Browse subdirectories
        datas = []
        for subdir in folder_path.iterdir():
            try:
                datas.append(self.scan_subfolder(subdir))
            except OSError:
                print('Patient {}'.format(subdir))

        return pandas.concat(datas, sort=False, ignore_index=True)

    @staticmethod
    def scan_subfolder(subdir):
        images = []
        for image_type in Reader.images_types:
            paths = list(subdir.glob(f'*.{image_type}'))
            images.extend([str(path) for path in paths])
        sub_frame = pandas.DataFrame({'Data': images})
        sub_frame['Label'] = subdir.stem
        return sub_frame

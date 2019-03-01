from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.core.pipes import DermatologyPipes
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = filename
    validation = StratifiedKFold(n_splits=2, shuffle=True)

    # Output dir
    output_folder = normpath('{temp}/dermatology/{filename}'.format(temp=temp_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Feature folder
    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Input data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                     normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
    inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'}, filter_by=filter_by)
    inputs.load()

    # Initiate model and params
    process = Process()
    process.begin(validation, validation, n_jobs=2)
    process.end(inputs=inputs, model=DermatologyPipes.get_wavelet(cache_folder=features_folder),
                output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Processes
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters

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

    # Input data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'LB', 'Normal']}
    input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                     normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
    inputs = Inputs(folders=input_folders, loader=dermatology.Reader.scan_folder,
                    tags={'data_tag': 'Data', 'label_tag': 'Label'}, filter_by=filter_by)
    inputs.load()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    model = KerasBatchClassifier(DeepModels.get_dummy_model)
    params = {'epochs': 2,
              'batch_size': 10,
              'preprocessing_function': None,
              'callbacks': DeepModels.get_callbacks(output_folder),
              'inner_cv': validation,
              'outer_cv': validation}

    # Launch process
    Processes.dermatology(inputs, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)

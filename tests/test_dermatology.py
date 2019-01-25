from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname
from sklearn.model_selection import StratifiedKFold
from experiences.processes import Processes
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":

    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = 'DermatologyDeepTest'
    epochs = 1
    batch_size = 10
    validation = StratifiedKFold(n_splits=2, shuffle=True)

    # Output dir
    output_folder = normpath('{temp}/dermatology/'.format(temp=temp_path))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    input_folder = normpath('{here}/data/dermatology/Patients'.format(here=here_path))
    input_folders = [input_folder]

    # Filters
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    model = KerasBatchClassifier(DeepModels.get_dummy_model)
    params = {'epochs': [epochs],
              'batch_size': [batch_size],
              'preprocessing_function': [None],
              'inner_cv': validation,
              'outer_cv': validation}

    # Launch process
    Processes.dermatology(input_folders, filter_by, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)



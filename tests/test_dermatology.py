from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname
from sklearn.model_selection import StratifiedKFold
from experiences.processes import Processes
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":

    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = 'DermatologyDeepTest'
    validation = StratifiedKFold(n_splits=2, shuffle=True)

    # Output dir
    output_folder = normpath('{temp}/dermatology/'.format(temp=temp_path))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}
    input_folder = normpath('{here}/data/dermatology/Patients'.format(here=here_path))
    inputs = Inputs(folders=[input_folder], loader=dermatology.Reader.scan_folder,
                    tags={'data_tag': 'Data', 'label_tag': 'Label'}, filter_by=filter_by)

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    model = KerasBatchClassifier(DeepModels.get_dummy_model)
    params = {'epochs': [1],
              'batch_size': [10],
              'preprocessing_function': [None],
              'inner_cv': validation,
              'outer_cv': validation}

    # Launch process
    Processes.dermatology(inputs, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)



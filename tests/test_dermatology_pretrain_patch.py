from copy import deepcopy
from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename
from sklearn.model_selection import StratifiedKFold
from experiences.processes import Processes
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
    epochs = 1
    batch_size = 10
    validation = StratifiedKFold(n_splits=2, shuffle=True)

    # Output dir
    output_folder = normpath('{temp}/dermatology/{filename}'.format(temp=temp_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Pretrain data
    pretrain_folder = normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data_tag': 'Data', 'label_tag': 'Label'})
    pretrain_inputs.load()

    # Input data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}
    input_folder = normpath('{here}/data/dermatology/Patients'.format(here=here_path))
    inputs = deepcopy(pretrain_inputs)
    inputs.change_data(folders=[input_folder], filter_by=filter_by, loader=dermatology.Reader.scan_folder,
                       tags={'data_tag': 'Data', 'label_tag': 'Label', 'groups': 'Patient'})
    inputs.load()

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
    Processes.dermatology_pretrain(pretrain_inputs, inputs, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)



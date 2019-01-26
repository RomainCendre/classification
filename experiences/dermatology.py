from os import makedirs, startfile
from sklearn.model_selection import StratifiedKFold
from os.path import exists, expanduser, normpath
from experiences.processes import Processes
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters

if __name__ == '__main__':

    # Parameters
    home_path = expanduser("~")
    name = 'Dermatology'
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Skin/Saint_Etienne/Deep/'.format(home=home_path))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}
    input_folder = normpath('{home}/Data/Skin/Saint_Etienne/Patients'.format(home=home_path))
    inputs = Inputs(folders=[input_folder], loader=dermatology.Reader.scan_folder,
                    tags={'data_tag': 'Data', 'label_tag': 'Label'}, filter_by=filter_by)
    inputs.load()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    model = KerasBatchClassifier(DeepModels.get_confocal_model)
    params = {'epochs': [100],
              'batch_size': [10],
              'preprocessing_function': [DeepModels.get_confocal_preprocessing()],
              'inner_cv': validation,
              'outer_cv': validation}

    # Launch process
    Processes.dermatology(inputs, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)

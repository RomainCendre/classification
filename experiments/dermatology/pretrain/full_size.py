from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Processes
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters

if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser("~")
    name = filename
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Dermatology/Pretrain/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Pretrain data
    pretrain_folder = normpath('{home}/Data/Skin/Thumbnails/'.format(home=home_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data_tag': 'Data', 'label_tag': 'Label'})
    pretrain_inputs.load()

    # Input data
    inputs = deepcopy(pretrain_inputs)
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'LB', 'Normal']}

    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs.change_data(folders=input_folders, filter_by=filter_by, loader=dermatology.Reader.scan_folder,
                       tags={'data_tag': 'Data', 'label_tag': 'Label', 'groups': 'Patient'})
    inputs.load()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=1)

    # Initiate model and params
    model = KerasBatchClassifier(DeepModels.get_confocal_model)
    params = {'epochs': [100],
              'batch_size': [50],
              'preprocessing_function': [DeepModels.get_confocal_preprocessing()],
              'inner_cv': validation,
              'outer_cv': validation}

    # Launch process
    Processes.dermatology_pretrain(pretrain_inputs, inputs, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)

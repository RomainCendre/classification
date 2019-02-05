from os import makedirs, startfile

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from os.path import exists, expanduser, normpath, basename, splitext, join
from experiments.processes import Processes
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters

if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    name = 'Results'
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Dermatology/Transfer_learning/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Temporary folder
    temp_folder = join(output_folder, 'Temp')
    if not exists(temp_folder):
        makedirs(temp_folder)

    # Input data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}

    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs = Inputs(folders=input_folders, loader=dermatology.Reader.scan_folder,
                    tags={'data_tag': 'Data', 'label_tag': 'Label'}, filter_by=filter_by)
    inputs.load()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    model = KerasBatchClassifier(DeepModels.get_application_model)
    model.init_model()
    model_params = {'preprocessing_function': DeepModels.get_application_preprocessing(),
                    'batch_size': 50}

    # Initiate model and params
    predictor = KerasClassifier(DeepModels.get_prediction_model)
    predictor_params = {'epochs': 100,
                        'output_classes': 3,
                        'batch_size': 50,
                        'callbacks': DeepModels.get_callbacks(output_folder),
                        'inner_cv': validation,
                        'outer_cv': validation}

    # Launch process
    Processes.dermatology_bottleneck(inputs, temp_folder, output_folder, model, model_params,
                                     predictor, predictor_params, name)

    # Open result folder
    startfile(output_folder)

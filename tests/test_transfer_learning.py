from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename, join

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
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
                    tags={'data_tag': 'Full_path', 'label_tag': 'Label', 'reference_tag': 'Reference'}, filter_by=filter_by)
    inputs.load()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    extractor = KerasBatchClassifier(DeepModels.get_application_model)
    extractor_params = {'architecture': 'MobileNet',
                        'batch_size': 1,
                        'preprocessing_function': None}

    predictor = KerasClassifier(DeepModels.get_dummy_model)
    predictor_params = {'batch_size': 10,
                        'output_classes': 3}

    process = Process()
    process.begin(validation, validation, DeepModels.get_callbacks(output_folder))
    process.checkpoint_step(inputs=inputs, model=extractor, params=extractor_params, folder=features_folder)
    process.end(inputs=inputs, model=predictor, params=predictor_params, output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

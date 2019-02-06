from copy import deepcopy
from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from experiments.processes import Process
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels, ClassifierPatch
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

    # Pretrain data
    pretrain_folder = normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data_tag': 'Data', 'label_tag': 'Label'})
    pretrain_inputs.load()

    # Input data
    inputs = deepcopy(pretrain_inputs)
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                     normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
    inputs.change_data(folders=input_folders, filter_by=filter_by, loader=dermatology.Reader.scan_folder,
                       tags={'data_tag': 'Data', 'label_tag': 'Label', 'groups': 'Patient'})
    inputs.load()

    hierarchy = [inputs.encode_label(['Malignant'])[0],
                 inputs.encode_label(['Benign'])[0],
                 inputs.encode_label(['Normal'])[0]]
    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    model = KerasBatchClassifier(DeepModels.get_dummy_model)
    params = {'epochs': 1,
              'batch_size': 10,
              'preprocessing_function': None}

    # Launch process
    process = Process()
    process.begin(validation, validation, DeepModels.get_callbacks(output_folder))
    model, params = process.train_step(pretrain_inputs, model, params)
    classifier = KerasClassifier(DeepModels.get_dummy_model)
    classifier.model = model.model
    patch_classifier = ClassifierPatch(classifier, SVC(kernel='linear', probability=True), 250)
    process.end(inputs, patch_classifier, output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)



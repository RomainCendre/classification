from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import Transforms, Classifiers
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

    # Features folder
    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Projection folder
    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    # Pretrain data
    pretrain_folder = normpath('{home}/Data/Skin/Thumbnails/'.format(home=home_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], instance=dermatology.Reader(),
                             loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'})
    pretrain_inputs.load()

    # Input data
    inputs = deepcopy(pretrain_inputs)
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}

    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs.change_data(folders=input_folders, filter_by=filter_by, loader=dermatology.Reader.scan_folder,
                       tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'})
    inputs.load()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    extractor = KerasBatchClassifier(Transforms.get_application)
    extractor_params = {'architecture': 'InceptionV3',
                        'epochs': 100,
                        'batch_size': 10,
                        'preprocessing_function': Classifiers.get_preprocessing_application()}

    predictor = KerasClassifier(Classifiers.get_dense)
    predictor_params = {'batch_size': 1,
                        'epochs': 100,
                        'optimizer': 'adam',
                        'output_classes': 3}

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation)
    process.checkpoint_step(inputs=pretrain_inputs, model=(extractor, extractor_params), folder=features_folder)
    model, params = process.train_step(inputs=pretrain_inputs, model=(predictor, predictor_params))

    process.checkpoint_step(inputs=inputs, model=(extractor, extractor_params), folder=features_folder,
                            projection_folder=projection_folder, projection_name=name)
    process.end(inputs=inputs, model=(model, params), output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

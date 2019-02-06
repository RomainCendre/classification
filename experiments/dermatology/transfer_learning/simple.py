from os import makedirs, startfile

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from os.path import exists, expanduser, normpath, basename, splitext, join
from experiments.processes import Process
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters

if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    name_patch = 'Patch'
    name_full = 'Full'
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath(
        '{home}/Results/Dermatology/Transfer_learning/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Temporary folder
    patch_folder = join(output_folder, 'Features_patch')
    if not exists(patch_folder):
        makedirs(patch_folder)

    full_folder = join(output_folder, 'Features_full')
    if not exists(full_folder):
        makedirs(full_folder)

    # Projection folder
    projection_patch_folder = join(output_folder, 'Projection_patch')
    if not exists(projection_patch_folder):
        makedirs(projection_patch_folder)

    projection_full_folder = join(output_folder, 'Projection_full')
    if not exists(projection_full_folder):
        makedirs(projection_full_folder)

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    ################# PATCH
    # Input patch
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs_patch = Inputs(folders=[input_folder], loader=dermatology.Reader.scan_folder_for_images,
                          tags={'data_tag': 'Data', 'label_tag': 'Label'})
    inputs_patch.load()

    # Initiate model and params
    extractor = KerasBatchClassifier(DeepModels.get_application_model)
    extractor_params = {'architecture': 'InceptionV3',
                        'batch_size': 1,
                        'preprocessing_function': DeepModels.get_application_preprocessing()}

    predictor = KerasClassifier(DeepModels.get_prediction_model)
    predictor_params = {'batch_size': 1,
                        'epochs': 100,
                        'optimizer': 'adam',
                        'output_classes': 3}
    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation)
    process.checkpoint_step(inputs=inputs_patch, model=extractor, params=extractor_params, folder=patch_folder,
                            projection_folder=projection_patch_folder)
    process.end(inputs=inputs_patch, model=predictor, params=predictor_params, output_folder=output_folder, name=name_patch)

    ################# FULL
    # Input full
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs_full = Inputs(folders=input_folders, loader=dermatology.Reader.scan_folder,
                         tags={'data_tag': 'Data', 'label_tag': 'Label'}, filter_by=filter_by)
    inputs_full.load()

    # Launch process
    process.checkpoint_step(inputs=inputs_full, model=extractor, params=extractor_params, folder=patch_folder,
                            projection_folder=projection_full_folder)
    process.end(inputs=inputs_full, model=predictor, params=predictor_params, output_folder=output_folder, name=name_full)

    # Open result folder
    startfile(output_folder)

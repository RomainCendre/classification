from copy import deepcopy
from os import makedirs, startfile
from sklearn.model_selection import StratifiedKFold
from os.path import exists, expanduser, normpath, splitext, basename, join

from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.IO import dermatology
from toolbox.IO.writers import PatchWriter
from toolbox.core.models import Transforms, Classifiers, PatchClassifier
from toolbox.core.structures import Inputs
from toolbox.core.transforms import PredictorTransform, OrderedEncoder
from toolbox.tools.limitations import Parameters

if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    name = 'Results'
    validation = StratifiedKFold(n_splits=5, shuffle=True)
    # Parameters
    colors = {'patches': dict(Malignant=[255, 0, 0], Benign=[125, 125, 0], Normal=[0, 255, 0]),
              'draw': dict(Malignant=(1, 0, 0), Benign=(0.5, 0.5, 0), Normal=(0, 1, 0))}

    # Output dir
    output_folder = normpath(
        '{home}/Results/Dermatology/SVM/Transfer_learning/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Sub folders
    patch_folder = join(output_folder, 'Patch')
    if not exists(patch_folder):
        makedirs(patch_folder)

    patch_colors_folder = join(output_folder, 'Patch_colors')
    if not exists(patch_colors_folder):
        makedirs(patch_colors_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Inputs data
    pretrain_folder = normpath('{home}/Data/Skin/Thumbnails/'.format(home=home_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], instance=dermatology.Reader(), style=colors,
                             loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'},
                             encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                                       'groups': LabelEncoder()})
    pretrain_inputs.load()

    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Normal', 'Benign', 'Malignant']}

    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]

    inputs = Inputs(folders=input_folders, instance=dermatology.Reader(patch_folder),
                    loader=dermatology.Reader.scan_folder_for_patches, style=colors, filter_by=filter_by,
                    tags={'data': 'Patch_Path', 'label': 'Label', 'groups': 'Patient', 'reference': 'Patch_Reference'},
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation, n_jobs=2)

    # Patch model training
    process.checkpoint_step(inputs=pretrain_inputs, model=Transforms.get_keras_extractor(), folder=features_folder)
    model, params = process.train_step(inputs=pretrain_inputs, model=Classifiers.get_linear_svm())

    # Final model evaluation
    process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(), folder=features_folder)
    process.checkpoint_step(inputs=inputs, model=PredictorTransform(model, probabilities=True), folder=features_folder)
    # patch_writer = PatchWriter(inputs)
    # patch_writer.write_patch(patch_colors_folder)
    inputs.patch_method()
    # model = PatchClassifier(hierarchies=[inputs.encode('label', 'Malignant'),
    #                                      inputs.encode('label', 'Benign'),
    #                                      inputs.encode('label', 'Normal')])
    process.end(inputs=inputs, model=Classifiers.get_linear_svm(), output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

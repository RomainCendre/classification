from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.core.models import PatchClassifier, Transforms, Classifiers
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.core.transforms import PredictorTransform

if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    name = 'Results'
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Dermatology/Haralick/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Features folder
    patch_folder = join(output_folder, 'Patch')
    if not exists(patch_folder):
        makedirs(patch_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Inputs data
    pretrain_folder = normpath('{home}/Data/Skin/Thumbnails/'.format(home=home_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], instance=dermatology.Reader(patch_folder), loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'})
    pretrain_inputs.load()

    inputs = deepcopy(pretrain_inputs)
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}

    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]

    inputs.change_data(folders=input_folders, filter_by=filter_by, loader=dermatology.Reader.scan_folder_for_patches,
                       tags={'data': 'Full_path', 'label': 'Label', 'groups': 'Patient', 'reference': 'Patch_Reference'})
    inputs.load()

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation)

    # Patch model training
    process.checkpoint_step(inputs=pretrain_inputs, model=Transforms.get_haralick(), folder=features_folder)
    model, params = process.train_step(inputs=pretrain_inputs, model=Classifiers.get_linear_svm())

    # Final model evaluation
    process.checkpoint_step(inputs=inputs, model=Transforms.get_haralick(), folder=features_folder)
    process.checkpoint_step(inputs=inputs, model=PredictorTransform(model, probabilities=False), folder=features_folder)
    inputs.patch_method()
    model = PatchClassifier(hierarchies=[inputs.encode_label(['Malignant'])[0],
                                         inputs.encode_label(['Benign'])[0],
                                         inputs.encode_label(['Normal'])[0]])
    process.end(inputs=inputs, model=model, output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

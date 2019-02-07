from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.core.models import SimpleModels, ClassifierPatch
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology

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

    # Pretrain data
    pretrain_folder = normpath('{home}/Data/Skin/Thumbnails/'.format(home=home_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data_tag': 'Data', 'label_tag': 'Label'})
    pretrain_inputs.load()

    # Input data
    inputs = deepcopy(pretrain_inputs)
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}

    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs.change_data(folders=input_folders, filter_by=filter_by, loader=dermatology.Reader.scan_folder,
                       tags={'data_tag': 'Data', 'label_tag': 'Label', 'groups': 'Patient', 'reference_tag': ['ID', 'Path']})
    inputs.load()

    # Initiate model and params
    model, params = SimpleModels.get_haralick_process()

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation)
    # Patch model training
    model, params = process.train_step(inputs=pretrain_inputs, model=model, params=params)
    # Final model evaluation
    patch_classifier = ClassifierPatch(model, SVC(kernel='linear', probability=True), 250)
    process.end(inputs=inputs, model=patch_classifier, output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

from copy import deepcopy
from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.core.models import DeepModels, SimpleModels, PatchClassifier
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.core.transforms import HaralickDescriptorTransform, PredictorTransform

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

    # Temporary folder
    patch_folder = join(output_folder, 'Patch')
    if not exists(patch_folder):
        makedirs(patch_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    predict_folder = join(output_folder, 'Predictions')
    if not exists(predict_folder):
        makedirs(predict_folder)

    # Input data
    # Inputs data
    pretrain_folder = normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], instance=dermatology.Reader(patch_folder),
                             loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data_tag': 'Full_path', 'label_tag': 'Label', 'reference_tag': ['Reference']})
    pretrain_inputs.load()

    inputs = deepcopy(pretrain_inputs)
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                     normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
    inputs.change_data(folders=input_folders, filter_by=filter_by, loader=dermatology.Reader.scan_folder_for_patches,
                       tags={'data_tag': 'Full_path', 'label_tag': 'Label', 'groups': 'Patient', 'reference_tag': ['Patch_Reference']})
    inputs.load()

    # Initiate model and params
    model, params = SimpleModels.get_dummy_process()

    # Launch process
    process = Process()
    process.begin(validation, validation)

    # Patch model training
    process.checkpoint_step(inputs=pretrain_inputs, model=HaralickDescriptorTransform(), folder=features_folder)
    model, params = process.train_step(inputs=pretrain_inputs, model=model, params=params)

    process.checkpoint_step(inputs=inputs, model=HaralickDescriptorTransform(), folder=features_folder)
    process.checkpoint_step(inputs=inputs, model=PredictorTransform(model, probabilities=False), folder=predict_folder)
    inputs.patch_method()
    hierarchies = [inputs.encode_label(['Malignant'])[0],
                   inputs.encode_label(['Benign'])[0],
                   inputs.encode_label(['Normal'])[0]]
    process.end(inputs=inputs, model=PatchClassifier(hierarchies=hierarchies), output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

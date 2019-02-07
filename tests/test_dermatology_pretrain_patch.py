from copy import deepcopy
from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename, join

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from experiments.processes import Process
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels, ClassifierPatch, SimpleModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.core.transforms import HaralickDescriptorTransform, PatchMakerTransform
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

    # Temporary folder
    patch_folder = join(output_folder, 'Patch')
    if not exists(patch_folder):
        makedirs(patch_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Pretrain data
    pretrain_folder = normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))
    pretrain_inputs = Inputs(folders=[pretrain_folder], loader=dermatology.Reader.scan_folder_for_images,
                             tags={'data_tag': 'Data', 'label_tag': 'Label', 'reference_tag': ['Data']})
    pretrain_inputs.load()

    # Input data
    inputs = deepcopy(pretrain_inputs)
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                     normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
    inputs.change_data(folders=input_folders, filter_by=filter_by, loader=dermatology.Reader.scan_folder,
                       tags={'data_tag': 'Data', 'label_tag': 'Label', 'groups': 'Patient', 'reference_tag': ['ID', 'Path']})
    inputs.load()

    # Initiate model and params
    model, params = SimpleModels.get_linear_svm_process()

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation)

    # Patch model training
    process.checkpoint_step(inputs=pretrain_inputs, model=HaralickDescriptorTransform(), folder=features_folder)
    model, params = process.train_step(inputs=pretrain_inputs, model=model, params=params)

    # Final model evaluation
    test = Pipeline([('Patch', PatchMakerTransform(folder=patch_folder)),
                     ('Hara', HaralickDescriptorTransform()),
                     ('None', None)])
    process.checkpoint_step(inputs=inputs, model=test, folder=features_folder)
    patch_classifier = ClassifierPatch(model, SVC(kernel='linear', probability=True), 250)
    process.end(inputs=inputs, model=patch_classifier, output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)



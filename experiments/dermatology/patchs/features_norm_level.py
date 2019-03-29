import itertools
from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.transforms import OrderedEncoder
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":
    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5, shuffle=True)
    test = validation  # GroupKFold(n_splits=5)
    settings = DefinedSettings.get_default_dermatology()

    # Output folders
    output_folder = normpath('{home}/Results/Dermatology/features_norm_level/'.format(home=home_path))
    if not exists(output_folder):
        makedirs(output_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    patch_folder = join(output_folder, 'Patch')
    if not exists(patch_folder):
        makedirs(patch_folder)

    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant']}),
               ('NvsM', {'Label': ['Normal', 'Malignant']}),
               ('NvsB', {'Label': ['Normal', 'Benign']}),
               ('BvsM', {'Label': ['Benign', 'Malignant']})]

    # Inputs
    inputs = [('NoOverlap', Dataset.patches_images(folder=patch_folder, size=250, overlap=0)),
              ('Overlap25', Dataset.patches_images(folder=patch_folder, size=250, overlap=0.25))]

    # Methods
    methods = [('Haralick', Transforms.get_haralick(mean=False)),
               ('KerasAverage', Transforms.get_keras_extractor(pooling='avg')),
               ('KerasMaximum', Transforms.get_keras_extractor(pooling='max'))]

    # Launch process
    keys = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']
    process = Process(output_folder=output_folder, name=filename, settings=settings, stats_keys=keys)
    process.begin(inner_cv=validation, outer_cv=test, n_jobs=2)

    # Parameters combinations
    combinations = list(itertools.product(inputs, filters, methods))

    for combination in combinations:
        input, filter, method = combination
        image_name = 'Image_{input}_{filter}_{method}'.format(input=input, filter=filter[0], method=method[0])
        patient_name = 'Patient_{input}_{filter}_{method}'.format(input=input, filter=filter[0], method=method[0])
        input = deepcopy(input[1])
        filter = filter[1]
        method = method[1]

        # Image classification
        input.name = image_name
        input.set_filters(filter)
        input.set_encoders({'label': OrderedEncoder().fit(filter['Label']),
                            'groups': LabelEncoder()})
        process.checkpoint_step(inputs=input, model=method, folder=features_folder)
        input.collapse(reference_tag='Reference', data_tag='ImageData', flatten=False)
        process.evaluate_step(inputs=input, model=Classifiers.get_norm_model())

        # Patient classification
        input.name = patient_name
        input.collapse(reference_tag='ID', data_tag='PatientData', flatten=True)
        input.tags.update({'label': 'Binary_Diagnosis'})
        input.set_encoders({'label': OrderedEncoder().fit(['Benign', 'Malignant']),
                            'groups': LabelEncoder()})
        process.evaluate_step(inputs=input, model=Classifiers.get_norm_model(patch=False))

    process.end()

    # Open result folder
    startfile(output_folder)

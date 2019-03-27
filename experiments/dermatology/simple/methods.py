import itertools
from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.models import Transforms, Classifiers
from toolbox.core.transforms import OrderedEncoder
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":
    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5, shuffle=True)
    test = validation  # GroupKFold(n_splits=5)
    settings = DefinedSettings.get_default_dermatology()

    # Output folders
    output_folder = normpath('{home}/Results/Dermatology/simple/{filename}/'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant']}),
               ('NvsM', {'Label': ['Normal', 'Malignant']}),
               ('NvsB', {'Label': ['Normal', 'Benign']}),
               ('BvsM', {'Label': ['Benign', 'Malignant']})]

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Methods
    methods = [('Haralick', Transforms.get_haralick(mean=False)),
               ('KerasAverage', Transforms.get_keras_extractor(pooling='avg')),
               ('KerasMaximum', Transforms.get_keras_extractor(pooling='max'))]

    # Models
    models = [('Svm', Classifiers.get_linear_svm()),
              ('SvmPca', Classifiers.get_linear_svm(reduce=20))]

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=test, n_jobs=2, settings=settings)

    # Parameters combinations
    combinations = list(itertools.product(filters, methods, models))

    # Thumbnails
    for combination in combinations:
        filter, method, model = combination
        inputs = Dataset.thumbnails()
        name = 'Thumbnails_{filter}_{method}_{model}'.format(filter=filter[0], method=method[0], model=model[0])

        # Image classification
        inputs.set_filters(filter[1])
        inputs.set_encoders({'label': OrderedEncoder().fit(filter[1]['Label']),
                            'groups': LabelEncoder()})

        process.checkpoint_step(inputs=inputs, model=method[1], folder=features_folder,
                                projection_folder=projection_folder, projection_name=name)
        process.end(inputs=inputs, model=model[1], output_folder=output_folder, name=name)

    # Full images
    for combination in combinations:
        filter, method, model = combination
        inputs = Dataset.full_images()
        name = 'Full_{method}_{model}'.format(filter=filter[0], method=method[0], model=model[0])

        # Image classification
        inputs.set_filters(filter[1])
        inputs.set_encoders({'label': OrderedEncoder().fit(filter[1]['Label']),
                            'groups': LabelEncoder()})

        process.checkpoint_step(inputs=inputs, model=method[1], folder=features_folder,
                                projection_folder=projection_folder, projection_name=name)
        process.end(inputs=inputs, model=model[1], output_folder=output_folder, name=name)

        # Patient classification
        inputs.collapse(reference_tag='ID', data_tag='PatientData', flatten=True)
        inputs.tags.update({'label': 'Binary_Diagnosis'})
        inputs.set_encoders({'label': OrderedEncoder().fit(['Benign', 'Malignant']),
                             'groups': LabelEncoder()})
        process.end(inputs=inputs, model=model[1], output_folder=output_folder,
                    name='Patient_{name}'.format(name=name))

    # Open result folder
    startfile(output_folder)

import itertools
import webbrowser
import numpy as np
from os import makedirs
from os.path import exists, splitext, basename, join
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms
from toolbox.core.parameters import BuiltInSettings, LocalParameters, DermatologyDataset
from toolbox.core.transforms import OrderedEncoder


def get_linear_svm():
    # Add scaling step
    steps = []
    steps.append(('scale', StandardScaler()))
    steps.append(('clf', SVC(kernel='linear', class_weight='balanced', probability=True)))
    pipe = Pipeline(steps)
    pipe.name = 'LinearSVM'

    # Define parameters to validate through grid CV
    parameters = [
        {'clf__C': np.logspace(-2, 10, 13), 'clf__kernel': ['linear']},
        {'clf__C': np.logspace(-2, 10, 13), 'clf__gamma': np.logspace(-9, 3, 13), 'clf__kernel': ['rbf']},
    ]
    return pipe, parameters


def simple(original_inputs, folder):

    # Advanced parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation, test = LocalParameters.get_validation_test()
    settings = BuiltInSettings.get_default_dermatology()

    # Statistics expected
    statistics = LocalParameters.get_statistics_keys()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Methods
    methods = [('Wavelet', Transforms.get_image_dwt()),
               ('Haralick', Transforms.get_haralick(mean=False)),
               ('KerasAverage', Transforms.get_keras_extractor(pooling='avg')),
               ('KerasMaximum', Transforms.get_keras_extractor(pooling='max'))]

    # Models

    models = [('Svm', get_linear_svm())]

    # Parameters combinations
    combinations = list(itertools.product(methods, models))

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu)

        for extractor, model in combinations:

            name = '{method}_{model}'.format(method=extractor[0], model=model[0])

            # Name experiment and filter data
            inputs = original_inputs.copy_and_change(filter_groups)

            # Filter datasets
            inputs.set_filters(filter_datas)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'groups': LabelEncoder()})

            # Change inputs
            process.change_inputs(inputs, split_rule=test)

            # Extract features on datasets
            process.checkpoint_step(inputs=inputs, model=extractor[1])

            # Evaluate Patch
            patch_filter = {'Type': ['Patch']}
            patch_filter.update(filter_datas)
            inputs.set_filters(patch_filter)
            inputs.name = '{name}_Patch'.format(name=name)
            process.evaluate_step(inputs=inputs, model=model[1])

            # Evaluate Full
            full_filter = {'Type': ['Full']}
            full_filter.update(filter_datas)
            inputs.set_filters(full_filter)
            inputs.name = '{name}_Full'.format(name=name)
            process.evaluate_step(inputs=inputs, model=model[1])

        process.end()


if __name__ == "__main__":

    # Configure GPU consumption
    LocalParameters.set_gpu(percent_gpu=0.5)

    # Parameters
    filename = splitext(basename(__file__))[0]
    output_folder = join(LocalParameters.get_dermatology_results(), filename)
    if not exists(output_folder):
        makedirs(output_folder)

    # Input patch
    image_inputs = DermatologyDataset.images()

    # Compute data
    simple(image_inputs, output_folder)

    # Open result folder
    webbrowser.open('file:///{folder}'.format(folder=output_folder))

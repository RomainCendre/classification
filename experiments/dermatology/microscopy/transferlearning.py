import itertools
import webbrowser
from pathlib import Path

from numpy import logspace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms
from toolbox.core.parameters import BuiltInSettings, LocalParameters, DermatologyDataset
from toolbox.core.transforms import OrderedEncoder


def get_linear_svm():
    # Add scaling step
    steps = [('scale', StandardScaler()), ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]
    pipe = Pipeline(steps)
    pipe.name = 'LinearSVM'

    # Define parameters to validate through grid CV
    parameters = {'clf__C': logspace(-2, 3, 6).tolist()}
    return pipe, parameters


def transfer_learning(original_inputs, folder):

    # Advanced parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_dermatology()

    # Statistics expected
    statistics = LocalParameters.get_statistics_keys()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Types
    types = [('Thumbnails', {'Type': 'Patch'}), ('Full', {'Type': 'Full'})]

    # Methods
    methods = [('VGG16', Transforms.get_keras_extractor(architecture='VGG16')),
               ('InceptionV3', Transforms.get_keras_extractor(architecture='InceptionV3')),
               ('InceptionResNetV2', Transforms.get_keras_extractor(architecture='InceptionResNetV2'))]
               #('NASNet', Transforms.get_keras_extractor(architecture='NASNet'))]

    # Models
    models = [('Svm', get_linear_svm())]

    # Parameters combinations
    combinations = list(itertools.product(types, methods, models))

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu)

        for im_type, extractor, model in combinations:

            # Name experiment and filter data
            inputs = original_inputs.copy_and_change(filter_groups)
            inputs.name = '{type}_{method}_{model}'.format(type=im_type[0], method=extractor[0], model=model[0])

            # Filter datasets
            type_filter = im_type[1]
            type_filter.update(filter_datas)
            inputs.set_filters(type_filter)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'group': LabelEncoder()})
            inputs.build_folds()

            # Extract features on datasets
            process.checkpoint_step(inputs=inputs, model=extractor[1])

            # Evaluate
            process.evaluate_step(inputs=inputs, model=model[1])

        process.end()


if __name__ == "__main__":
    # Parameters
    current_file = Path(__file__)
    output_folder = DermatologyDataset.get_results_location()/current_file.stem
    if not output_folder.is_dir():
        output_folder.mkdir()

    # Input patch
    image_inputs = DermatologyDataset.images(modality='Microscopy')

    # Compute data
    transfer_learning(image_inputs, output_folder)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

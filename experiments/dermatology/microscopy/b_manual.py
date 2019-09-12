

import itertools
import webbrowser
from pathlib import Path

from numpy import logspace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


def get_cart():
    # Add scaling step
    steps = [('scale', StandardScaler()), ('clf', DecisionTreeClassifier())]
    pipe = Pipeline(steps)
    pipe.name = 'Cart'

    # Define parameters to validate through grid CV
    parameters = {}
    return pipe, parameters


def manual(original_inputs, folder):

    # Advanced parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_dermatology()
    scoring = LocalParameters.get_scorer()

    # Statistics expected
    statistics = LocalParameters.get_dermatology_statistics()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Methods
    methods = [('Wavelet', Transforms.get_image_dwt()),
               ('Fourier', Transforms.get_image_fft()),
               ('Haralick', Transforms.get_haralick(mean=False))]

    # Models
    models = [('CART', get_cart()), ('LinearSVM', get_linear_svm())]

    # Parameters combinations
    combinations = list(itertools.product(methods, models))

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu, scoring=scoring)

        for im_type, extractor, model in combinations:

            # Name experiment and filter data
            inputs = original_inputs.copy_and_change(filter_groups)
            inputs.name = f'{extractor[0]}_{model[0]}'

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

    # Input patch
    image_inputs = DermatologyDataset.images(modality='Microscopy')
    image_types = ['Patch', 'Full']
    # Folder
    output_folder = DermatologyDataset.get_results_location() / 'Manual'

    for image_type in image_types:
        inputs = image_inputs.copy_and_change({'Type': image_type})
        # Compute data
        output = output_folder/image_type
        output.mkdir(exist_ok=True)
        manual(image_inputs, output)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

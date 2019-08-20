import itertools
import webbrowser
from copy import copy
from pathlib import Path

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from experiments.processes import Process
from toolbox.core.builtin_models import Classifiers
from toolbox.core.models import KerasBatchClassifier
from toolbox.core.transforms import OrderedEncoder
from toolbox.core.parameters import LocalParameters, DermatologyDataset, BuiltInSettings


def get_fine_tuning(output_classes, trainable_layers=0, added_layers=0):
    model = KerasBatchClassifier(build_fn=Classifiers.get_fine_tuning)
    metrics = Metrics()
    parameters = {  # Build paramters
        'architecture': 'InceptionV3',
        'optimizer': 'adam',
        'callbacks': [metrics],
        # Parameters for fit
        'epochs': 50,
        'batch_size': 64,
    }
    parameters.update({'output_classes': output_classes,
                       'trainable_layers': trainable_layers,
                       'added_layers': added_layers})
    fit_parameters = {  # Transformations
        'rotation_range': 180,
        'horizontal_flip': True,
        'vertical_flip': True,
    }

    return model, parameters, fit_parameters


def fine_tune(original_inputs, folder):
    # Parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_dermatology()
    scoring = LocalParameters.get_scorer()

    # Statistics expected
    statistics = LocalParameters.get_dermatology_statistics()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Image filters
    types = [('Thumbnails', {'Type': 'Patch'}), ('Full', {'Type': 'Full'})]

    # Layers parameters
    layers_parameters = {'trainable_layer': [0, 1, 2]}

    # Parameters combinations
    combinations = list(itertools.product(types, ParameterGrid(layers_parameters)))

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu, scoring=scoring)

        for scale, params in combinations:

            inputs = original_inputs.copy_and_change(filter_groups)
            inputs.name = '{scale}_{params}'.format(scale=scale[0], params=params)

            # Filter datasets
            scale_filters = copy(scale[1])
            scale_filters.update(filter_datas)
            inputs.set_filters(scale_filters)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'group': LabelEncoder()})
            inputs.build_folds()

            process.evaluate_step(inputs=inputs,
                                  model=get_fine_tuning(output_classes=len(filter_datas['Label']),
                                                        trainable_layers=params['trainable_layer']))
        process.end()


if __name__ == "__main__":
    # Parameters
    current_file = Path(__file__)
    output_folder = DermatologyDataset.get_results_location()/current_file.stem
    output_folder.mkdir(exist_ok=True)

    # Input patch
    image_inputs = DermatologyDataset.images(modality='Microscopy')

    # Compute data
    fine_tune(image_inputs, output_folder)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

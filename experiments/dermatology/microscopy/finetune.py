import itertools
import webbrowser
from copy import copy
from os import makedirs
from sklearn.model_selection import ParameterGrid
from os.path import exists, basename, splitext, join
from sklearn.preprocessing import LabelEncoder
from experiments.processes import Process
from toolbox.core.builtin_models import Classifiers
from toolbox.core.models import KerasBatchClassifier
from toolbox.core.transforms import OrderedEncoder
from toolbox.core.parameters import LocalParameters, DermatologyDataset, BuiltInSettings


def get_fine_tuning(output_classes, trainable_layers=0, added_layers=0):
    model = KerasBatchClassifier(build_fn=Classifiers.get_fine_tuning)
    parameters = {  # Build paramters
        'architecture': 'InceptionV3',
        'optimizer': 'adam',
        'metrics': [['accuracy']],
        # Parameters for fit
        'epochs': 100,
        'batch_size': 32,
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
    validation, test = LocalParameters.get_validation_test()
    settings = BuiltInSettings.get_default_dermatology()

    # Statistics expected
    statistics = LocalParameters.get_statistics_keys()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Image filters
    scales = [('Thumbnails', {'Type': 'Patch'}), ('Full', {'Type': 'Full'})]

    # Layers parameters
    layers_parameters = {'trainable_layer': [0, 1, 2],
                         'added_layer': [1, 2, 3, 4]}

    # Parameters combinations
    combinations = list(itertools.product(scales, ParameterGrid(layers_parameters)))

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=1)

        for scale, params in combinations:
            inputs = original_inputs.copy_and_change(filter_groups)
            inputs.name = '{scale}_{params}'.format(scale=scale[0], params=params)

            # Filter datasets
            scale_filters = copy(scale[1])
            scale_filters.update(filter_datas)
            inputs.set_filters(scale_filters)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'groups': LabelEncoder()})

            # Change inputs
            process.change_inputs(inputs, split_rule=test)

            process.evaluate_step(inputs=inputs,
                                  model=get_fine_tuning(output_classes=len(filter_datas['Label']),
                                                        trainable_layers=params['trainable_layer'],
                                                        added_layers=params['added_layer']))
        process.end()


if __name__ == "__main__":
    # Parameters
    filename = splitext(basename(__file__))[0]
    output_folder = join(LocalParameters.get_dermatology_results(), filename)
    if not exists(output_folder):
        makedirs(output_folder)

    # Input patch
    image_inputs = DermatologyDataset.images(modality='Microscopy')

    # Compute data
    fine_tune(image_inputs, output_folder)

    # Open result folder
    webbrowser.open('file:///{folder}'.format(folder=output_folder))

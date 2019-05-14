import itertools
from os import makedirs, startfile
from sklearn.model_selection import ParameterGrid
from os.path import exists, basename, splitext, join
from sklearn.preprocessing import LabelEncoder
from experiments.processes import Process
from toolbox.core.builtin_models import BuiltInModels
from toolbox.core.transforms import OrderedEncoder
from toolbox.core.parameters import LocalParameters, DermatologyDataset, BuiltInSettings


def fine_tune(original_inputs, folder):

    # Parameters
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

    for filter_name, filter_datas, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=1)

        for scale, params in combinations:
            inputs = original_inputs.copy_and_change(filter_groups)
            inputs.name = '{scale}_{params}'.format(scale=scale[0], params=params)

            filter_datas.update(scale[1])
            inputs.set_filters(filter_datas)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']),
                                 'groups': LabelEncoder()})
            process.evaluate_step(inputs=inputs,
                                  model=BuiltInModels.get_fine_tuning(output_classes=len(filter_datas['Label']),
                                                                      trainable_layers=params['trainable_layer'],
                                                                      added_layers=params['added_layer']))
        process.end()

    # Open result folder
    startfile(output_folder)


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
    fine_tune(image_inputs, output_folder)

    # Open result folder
    startfile(output_folder)

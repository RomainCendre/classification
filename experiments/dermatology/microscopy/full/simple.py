import itertools
from os import makedirs, startfile
from os.path import exists, splitext, basename, join
from sklearn.preprocessing import LabelEncoder
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.parameters import BuiltInSettings, LocalParameters, DermatologyDataset
from toolbox.core.transforms import OrderedEncoder


def simple(inputs, folder):

    # Advanced parameters
    validation, test = LocalParameters.get_validation_test()
    settings = BuiltInSettings.get_default_dermatology()

    # Statistics expected
    statistics = LocalParameters.get_statistics_keys()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Image filters
    scales = [('Thumbnails', {'Type': 'Patch'}), ('Full', {'Type': 'Full'})]

    # Methods
    methods = [('Wavelet', Transforms.get_image_dwt()),
               ('Haralick', Transforms.get_haralick(mean=False)),
               ('KerasAverage', Transforms.get_keras_extractor(pooling='avg')),
               ('KerasMaximum', Transforms.get_keras_extractor(pooling='max'))]

    # Models
    models = [('Svm', Classifiers.get_linear_svm())]

    # Parameters combinations
    combinations = list(itertools.product(methods, models))

    # Browse combinations
    for filter_name, filter_datas, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)

        for method, model in combinations:

            for scale in scales:
                inputs = inputs.copy_and_change(filter_groups)
                inputs.name = '{input}_{method}_{model}'.format(input=scale[0], method=method[0], model=model[0])

                filter_datas.update(scale[1])
                inputs.set_filters(filter_datas)
                inputs.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']), 'groups': LabelEncoder()})
                process.checkpoint_step(inputs=inputs, model=method[1])
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
    startfile(output_folder)



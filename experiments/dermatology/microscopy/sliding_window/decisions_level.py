import itertools
from numpy import array
from os import makedirs, startfile
from os.path import exists, splitext, basename, join
from sklearn.preprocessing import LabelEncoder
from experiments.processes import Process
from toolbox.IO.writers import PatchWriter
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.models import PatchClassifier
from toolbox.core.transforms import PredictorTransform, OrderedEncoder
from toolbox.core.parameters import LocalParameters, DermatologyDataset, BuiltInSettings


def decision_level(train_inputs, original_inputs, folder):

    # Parameters
    validation, test = LocalParameters.get_validation_test()
    settings = BuiltInSettings.get_default_dermatology()

    # Statistics expected
    statistics = LocalParameters.get_statistics_keys()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # View
    view_folder = join(folder, 'View')
    if not exists(view_folder):
        makedirs(view_folder)

    # Models
    models = [('Svm', Classifiers.get_linear_svm())]

    # Methods
    methods = Transforms.get_keras_extractor(pooling='avg')

    # Parameters combinations
    combinations = list(itertools.product(original_inputs, models))

    # Browse combinations
    for filter_name, filter_datas, filter_groups in filters:

        # Launch process
        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)

        for sliding, model in combinations:
            # Name experiment and filter data
            name = '{sliding}_{model}'.format(sliding=sliding[0], model=model[0])

            # Filter on datasets, applying groups of labels
            inputs = sliding[1].copy_and_change(filter_groups)
            pre_inputs = train_inputs.copy_and_change(filter_groups)

            # Filter datasets
            pre_inputs.set_filters(filter_datas)
            inputs.set_filters(filter_datas)

            # Set encoders
            pre_inputs.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']), 'groups': LabelEncoder()})
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']), 'groups': LabelEncoder()})
            inputs.name = name

            # Extract features on datasets
            process.checkpoint_step(inputs=pre_inputs, model=methods)
            process.checkpoint_step(inputs=inputs, model=methods)

            # Compute datas
            predictor, params = process.train_step(inputs=pre_inputs, model=model[1])
            process.checkpoint_step(inputs=inputs, model=PredictorTransform(predictor, probabilities=False))
            PatchWriter(inputs, settings).write_patch(folder=view_folder)

            # Collapse informations and make predictions
            inputs.collapse(reference_tag='Reference')
            process.evaluate_step(inputs=inputs, model=Classifiers.get_linear_svm())
            hierarchies = inputs.encode('label', array(list(reversed(filter_datas['Label']))))
            process.evaluate_step(inputs=inputs, model=PatchClassifier(hierarchies))

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
    patches_inputs = DermatologyDataset.images()
    patches_inputs.sub_inputs({'Type': 'Patch'})
    slidings_inputs = [('NoOverlap', DermatologyDataset.sliding_images(size=250, overlap=0)),
                       ('Overlap50', DermatologyDataset.sliding_images(size=250, overlap=0.50))]

    # Compute data
    decision_level(patches_inputs, slidings_inputs, output_folder)

    # Open result folder
    startfile(output_folder)
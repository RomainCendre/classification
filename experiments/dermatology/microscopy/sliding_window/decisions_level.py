import webbrowser

from numpy import array
from os import makedirs
from os.path import exists, splitext, basename, join
from sklearn.preprocessing import LabelEncoder
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.models import PatchClassifier
from toolbox.core.transforms import OrderedEncoder, ArgMaxTransform
from toolbox.core.parameters import LocalParameters, DermatologyDataset, BuiltInSettings


def decision_level(slidings, folder):

    # Parameters
    nb_cpu = LocalParameters.get_cpu_number()
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

    # Extracteur
    extractor = Transforms.get_keras_extractor(pooling='max')
    extractor.need_fit = False

    # Predicteur
    predictor = Classifiers.get_linear_svm()
    predictor[0].need_fit = True

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        # Launch process
        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu)

        for sliding in slidings:

            # Name experiment and filter data
            name = '{sliding}'.format(sliding=sliding[0])
            inputs = sliding[1].copy_and_change(filter_groups)

            # Filter datasets
            slide_filters = {'Type': ['Patch', 'Window']}
            slide_filters.update(filter_datas)
            inputs.set_filters(slide_filters)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'groups': LabelEncoder()})

            # Change inputs
            process.change_inputs(inputs, split_rule=test)

            # Extract features on datasets
            process.checkpoint_step(inputs=inputs, model=extractor)

            # Extract prediction on dataset
            process.checkpoint_step(inputs=inputs, model=predictor)

            # SCORE level predictions
            # Collapse information and make predictions
            inputs.set_filters(filter_datas)
            scores = inputs.collapse({'Type': ['Full']}, 'Reference', {'Type': ['Window']}, 'Source')

            # Evaluate using svm
            inputs.name = '{name}_score_svm'.format(name=name)
            process.evaluate_step(inputs=scores, model=Classifiers.get_linear_svm())
            inputs.name = '{name}_score_classifier'.format(name=name)
            hierarchies = inputs.encode('label', array(list(reversed(filter_datas['Label']))))
            process.evaluate_step(inputs=scores, model=PatchClassifier(hierarchies))

            # DECISION level predictions
            # Extract decision from predictions
            inputs.set_filters(slide_filters)
            process.checkpoint_step(inputs=inputs, model=ArgMaxTransform())

            inputs.set_filters(filter_datas)
            decisions = inputs.collapse({'Type': ['Full']}, 'Reference', {'Type': ['Window']}, 'Source')

            # Evaluate using svm
            inputs.name = '{name}_decision_svm'.format(name=name)
            inputs.set_filters(filter_datas)
            process.evaluate_step(inputs=decisions, model=Classifiers.get_linear_svm())
            inputs.name = '{name}_decision_classifier'.format(name=name)
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
    windows_inputs = [('NoOverlap', DermatologyDataset.sliding_images(size=250, overlap=0)),
                      ('Overlap50', DermatologyDataset.sliding_images(size=250, overlap=0.50))]

    # Compute data
    decision_level(windows_inputs, output_folder)

    # Open result folder
    webbrowser.open('file:///{folder}'.format(folder=output_folder))

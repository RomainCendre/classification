import itertools
from pathlib import Path

import misvm
import webbrowser
from numpy import logspace
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms
from toolbox.core.models import SelectAtMostKBest
from toolbox.core.parameters import LocalParameters, DermatologyDataset, BuiltInSettings
from toolbox.core.transforms import OrderedEncoder, PNormTransform, FlattenTransform


def get_reduce_model():
    # Steps and parameters
    steps = [('flatten', FlattenTransform()),
             ('scale', StandardScaler()),
             ('reduction', SelectAtMostKBest(f_classif)),
             ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]

    parameters = {'reduction__k': [20, 50, 100],
                  'clf__C': logspace(-2, 3, 6).tolist()}

    # Create pipeline
    pipe = Pipeline(steps)
    pipe.name = 'Reduce'
    pipe.need_fit = True
    return pipe, parameters


def get_norm_model():
    # Steps and parameters
    steps = [('norm', PNormTransform()),
             ('scale', StandardScaler()),
             ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]

    parameters = {'norm__p': [2, 3, 5],
                  'clf__C': logspace(-2, 3, 6).tolist()}

    # Create pipeline
    pipe = Pipeline(steps)
    pipe.name = 'PNorm'
    pipe.need_fit = True
    return pipe, parameters


def get_mil_decision():
    # Steps and parameters
    steps = [('scale', StandardScaler()),
             ('clf', misvm.MISVM(kernel='linear', C=1.0, max_iters=50))]

    parameters = {'clf__C': logspace(-2, 3, 6).tolist()}

    # Create pipeline
    pipe = Pipeline(steps)
    pipe.name = 'MiSVM'
    pipe.need_fit = True

    return pipe, parameters


def sliding_features(slidings, folder):
    # Parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation, test = LocalParameters.get_validation_test()
    settings = BuiltInSettings.get_default_dermatology()

    # Statistics expected
    statistics = LocalParameters.get_statistics_keys()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Extracteur
    extractor = Transforms.get_keras_extractor(pooling='max')
    extractor.need_fit = False

    # Evaluateurs
    evaluators = [('Reduced', get_reduce_model()),
                  ('PNorm', get_norm_model()),
                  ('MultiInstance', get_mil_decision())]

    # Parameters combinations
    combinations = list(itertools.product(slidings, evaluators))

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        # Launch process
        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu)

        for sliding, evaluator in combinations:

            # Name experiment and filter data
            name = '{sliding}_{evaluator}'.format(sliding=sliding[0], evaluator=evaluator[0])
            inputs = sliding[1].copy_and_change(filter_groups)

            # Filter datasets
            slide_filters = {'Type': ['Patch', 'Window']}
            slide_filters.update(filter_datas)
            inputs.set_filters(slide_filters)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'group': LabelEncoder()})

            # Change inputs
            process.change_inputs(inputs, split_rule=test)

            # Extract features on datasets
            process.checkpoint_step(inputs=inputs, model=extractor)

            # Collapse information and make predictions
            inputs.set_filters(filter_datas)
            features = inputs.collapse({'Type': ['Full']}, 'Reference', {'Type': ['Window']}, 'Source')

            # Evaluate using svm
            inputs.name = '{name}'.format(name=name)
            process.evaluate_step(inputs=features, model=evaluator[1])

        process.end()


if __name__ == "__main__":

    # Configure GPU consumption
    LocalParameters.set_gpu(percent_gpu=0.5)

    # Parameters
    current_file = Path(__file__)
    output_folder = DermatologyDataset.get_results_location()/current_file.stem
    if not output_folder.is_dir():
        output_folder.mkdir()

    # # Input patch
    # windows_inputs = [('NoOverlap', DermatologyDataset.sliding_images(size=250, overlap=0)),
    #                   ('Overlap50', DermatologyDataset.sliding_images(size=250, overlap=0.50))]

    windows_inputs = [('NoOverlap', DermatologyDataset.test_sliding_images(size=250, overlap=0))]

    # Compute data
    sliding_features(windows_inputs, output_folder)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

import webbrowser
from pathlib import Path
from numpy import logspace
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms
from toolbox.core.models import PCAAtMost, SelectAtMostKBest
from toolbox.core.parameters import LocalParameters, DermatologyDataset, BuiltInSettings
from toolbox.core.transforms import OrderedEncoder, FlattenTransform, PNormTransform


def get_reduce_model():
    # Steps and parameters
    steps = [('flatten', FlattenTransform()),
             ('scale', StandardScaler()),
             ('pca', PCAAtMost()),
             ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]

    parameters = {'pca__n_components': [20, 50],
                  'clf__C': logspace(-2, 3, 6).tolist()}

    # Create pipeline
    pipe = Pipeline(steps)
    pipe.name = 'Reduce'
    pipe.need_fit = True
    return pipe, parameters


def get_select_model():
    # Steps and parameters
    steps = [('flatten', FlattenTransform()),
             ('scale', StandardScaler()),
             ('select', SelectAtMostKBest(f_classif)),
             ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]

    parameters = {'select__k': [20, 50],
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


def multiscale_features(multiresolution_inputs, folder):
    # Parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_dermatology()

    # Statistics expected
    statistics = LocalParameters.get_dermatology_statistics()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Extracteur
    extractor = Transforms.get_keras_extractor(pooling='max')
    extractor.need_fit = False

    # Evaluateurs
    evaluators = [('Reduced', get_reduce_model()),
                  ('Select', get_select_model()),
                  ('PNorm', get_norm_model())]

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        # Launch process
        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu)

        for evaluator in evaluators:

            # Name experiment and filter data
            name = '{evaluator}'.format(evaluator=evaluator[0])
            inputs = multiresolution_inputs.copy_and_change(filter_groups)

            # Filter datasets
            slide_filters = {'Type': ['Multi']}
            slide_filters.update(filter_datas)
            inputs.set_filters(slide_filters)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'group': LabelEncoder()})
            inputs.build_folds()

            # Extract features on datasets
            process.checkpoint_step(inputs=inputs, model=extractor)

            # Collapse information and make predictions
            inputs.set_filters(filter_datas)
            features = inputs.collapse({'Type': ['Full']}, 'Reference', {'Type': ['Multi']}, 'Source')

            # Evaluate using svm
            features.name = '{name}'.format(name=name)
            process.evaluate_step(inputs=features, model=evaluator[1])

        process.end()


if __name__ == "__main__":
    # Parameters
    current_file = Path(__file__)
    output_folder = DermatologyDataset.get_results_location()/current_file.stem
    output_folder.mkdir(exist_ok=True)

    # Input patch
    multiresolution_input = DermatologyDataset.multiresolution(coefficients=[1, 0.75, 0.5, 0.25], modality='Microscopy')

    # Compute data
    multiscale_features(multiresolution_input, output_folder)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

import webbrowser
from pathlib import Path
import misvm
from numpy import logspace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms
from toolbox.core.parameters import BuiltInSettings, LocalParameters, DermatologyDataset
from toolbox.core.transforms import OrderedEncoder, PNormTransform


def get_linear_svm():
    # Add scaling step
    steps = [('scale', StandardScaler()), ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]
    pipe = Pipeline(steps)
    pipe.name = 'LinearSVM'

    # Define parameters to validate through grid CV
    parameters = {'clf__C': logspace(-2, 3, 6).tolist()}
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


def features(bag_inputs, folder):
    # Parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_dermatology()
    scoring = LocalParameters.get_scorer()

    # Statistics expected
    statistics = LocalParameters.get_dermatology_statistics()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Extracteur
    extractor, param = Transforms.get_image_dwt()
    extractor.need_fit = False

    # Evaluateurs
    evaluators = [('PNorm', get_norm_model()),
                  ('MultiInstance', get_mil_decision())]

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        # Launch process
        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu, scoring=scoring)

        for evaluator in evaluators:
            # Name experiment and filter data
            name = '{evaluator}'.format(evaluator=evaluator[0])
            inputs = bag_inputs.copy_and_change(filter_groups)

            # Filter datasets
            slide_filters = {'Type': ['Instance']}
            slide_filters.update(filter_datas)
            inputs.set_filters(slide_filters)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'group': LabelEncoder()})
            inputs.build_folds()

            # Extract features on datasets
            process.checkpoint_step(inputs=inputs, model=extractor)

            # Collapse information and make predictions
            inputs.set_filters(filter_datas)
            features = inputs.collapse({'Type': ['Full']}, 'Reference', {'Type': ['Instance']}, 'Source')

            # Evaluate using svm
            features.name = '{name}'.format(name=name)
            process.evaluate_step(inputs=features, model=evaluator[1])

        process.end()


if __name__ == "__main__":
    # Parameters
    current_file = Path(__file__)
    # Input sliding
    all_inputs = [('Multiscale', DermatologyDataset.multiresolution(coefficients=[1, 0.75, 0.5, 0.25], modality='Microscopy')),
                  ('NoOverlap', DermatologyDataset.sliding_images(size=250, overlap=0, modality='Microscopy')),
                  ('Overlap50', DermatologyDataset.sliding_images(size=250, overlap=0.50, modality='Microscopy'))]

    # Folder
    output_folder = DermatologyDataset.get_results_location() / 'Features'

    for inputs in all_inputs:
        output = output_folder / inputs[0]
        features(inputs[1], output)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

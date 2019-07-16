import webbrowser
from pathlib import Path
from numpy import logspace
from numpy.ma import array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.models import ScoreVotingClassifier, DecisionVotingClassifier
from toolbox.core.parameters import LocalParameters, DermatologyDataset, BuiltInSettings
from toolbox.core.transforms import OrderedEncoder, ArgMaxTransform


def get_linear_svm():
    # Add scaling step
    steps = [('scale', StandardScaler()), ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]
    pipe = Pipeline(steps)
    pipe.name = 'LinearSVM'

    # Define parameters to validate through grid CV
    parameters = {'clf__C': logspace(-2, 3, 6).tolist()}
    return pipe, parameters


def multiscale_decision(multiresolution_inputs, folder, homemade=False):

    # Parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation, test = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_dermatology()

    # Statistics expected
    statistics = LocalParameters.get_statistics_keys()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Extracteur
    extractor = Transforms.get_keras_extractor(pooling='max')
    extractor.need_fit = False

    # Predicteur
    predictor = Classifiers.get_linear_svm()
    predictor[0].need_fit = True

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        # Launch process
        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu)

        # Filter on datasets, applying groups of labels
        inputs = multiresolution_inputs.copy_and_change(filter_groups)

        # Filter datasets
        slide_filters = {'Type': ['Multi']}
        slide_filters.update(filter_datas)
        inputs.set_filters(slide_filters)
        inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'group': LabelEncoder()})
        inputs.build_folds()

        # Extract features on datasets
        process.checkpoint_step(inputs=inputs, model=extractor)

        # Extract prediction on dataset
        process.checkpoint_step(inputs=inputs, model=predictor)

        # SCORE level predictions
        # Collapse information and make predictions
        inputs.set_filters(filter_datas)
        scores = inputs.collapse({'Type': ['Full']}, 'Reference', {'Type': ['Multi']}, 'Source')

        # Evaluate using svm
        inputs.name = 'Multi_resolution_score_svm'
        process.evaluate_step(inputs=scores, model=get_linear_svm())

        if homemade:
            inputs.name = 'Multi_resolution_score_classifier'
            hierarchies = inputs.encode('label', array(list(reversed(filter_datas['Label']))))
            process.evaluate_step(inputs=scores, model=ScoreVotingClassifier(hierarchies))

        # DECISION level predictions
        # Extract decision from predictions
        inputs.set_filters(slide_filters)
        process.checkpoint_step(inputs=inputs, model=ArgMaxTransform())

        inputs.set_filters(filter_datas)
        decisions = inputs.collapse({'Type': ['Full']}, 'Reference', {'Type': ['Multi']}, 'Source')

        # Evaluate using svm
        inputs.name = 'Multi_resolution_decision_svm'
        inputs.set_filters(filter_datas)
        process.evaluate_step(inputs=decisions, model=get_linear_svm())

        if homemade:
            inputs.name = 'Multi_resolution_decision_classifier'
            hierarchies = inputs.encode('label', array(list(reversed(filter_datas['Label']))))
            process.evaluate_step(inputs=inputs, model=DecisionVotingClassifier(hierarchies))

        process.end()


if __name__ == "__main__":
    # Parameters
    current_file = Path(__file__)
    output_folder = DermatologyDataset.get_results_location()/current_file.stem
    if not output_folder.is_dir():
        output_folder.mkdir()

    # Input patch
    multiresolution_input = DermatologyDataset.multiresolution(coefficients=[1, 0.75, 0.5, 0.25], modality='Microscopy')

    # Compute data
    multiscale_decision(multiresolution_input, output_folder)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

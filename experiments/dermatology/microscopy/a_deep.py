import itertools
import webbrowser
from pathlib import Path

from numpy import logspace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from experiments.processes import Process
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.models import KerasFineClassifier
from toolbox.core.parameters import BuiltInSettings, LocalParameters, DermatologyDataset
from toolbox.core.transforms import OrderedEncoder
from toolbox_jupyter.IO import image


def get_linear_svm():
    # Add scaling step
    steps = [('scale', StandardScaler()), ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]
    pipe = Pipeline(steps)
    pipe.name = 'LinearSVM'

    # Define parameters to validate through grid CV
    parameters = {'clf__C': logspace(-2, 3, 6).tolist()}
    return pipe, parameters


def get_cart():
    # Add scaling step
    steps = [('scale', StandardScaler()), ('clf', DecisionTreeClassifier(class_weight='balanced'))]
    pipe = Pipeline(steps)
    pipe.name = 'Cart'

    # Define parameters to validate through grid CV
    parameters = {
                'max_depth': [3, 4, 5, 6],
                'min_samples_leaf': [0.04, 0.06, 0.08],
                'max_features': [0.2, 0.4, 0.6, 0.8]
                }
    return pipe, parameters


def get_fine_tuning(output_classes, trainable_layers=0):
    model = KerasFineClassifier(build_fn=Classifiers.get_fine_tuning)
    parameters = {  # Build parameters
        'architecture': 'InceptionV3',
        # Parameters for fit
        'epochs': 50,
        'batch_size': 64,
    }
    parameters.update({'output_classes': output_classes,
                       'trainable_layers': trainable_layers})
    fit_parameters = {  # Transformations
        'preprocessing_function': Classifiers.get_preprocessing_application(architecture='ResNet'),
        'rotation_range': 180,
        'horizontal_flip': True,
        'vertical_flip': True,
    }

    return model, parameters, fit_parameters


def transfer_learning(original_inputs, folder):
    # Advanced parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_dermatology()
    scoring = LocalParameters.get_scorer()

    # Statistics expected
    statistics = LocalParameters.get_dermatology_statistics()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Methods
    methods = [('VGG16', Transforms.get_tl_extractor(architecture='VGG16')),
               ('InceptionV3', Transforms.get_tl_extractor(architecture='InceptionV3')),
               ('InceptionResNetV2', Transforms.get_tl_extractor(architecture='InceptionResNetV2')),
               ('ResNet50', Transforms.get_tl_extractor(architecture='ResNet'))]

    # Models
    models = [('CART', get_cart()), ('LinearSVM', get_linear_svm())]

    # Parameters combinations
    combinations = list(itertools.product(methods, models))

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu, scoring=scoring)

        for extractor, model in combinations:
            # Name experiment and filter data
            inputs = original_inputs.copy_and_change(filter_groups)
            inputs.name = f'{extractor[0]}_{model[0]}'

            # Filter datasets
            inputs.set_filters(filter_datas)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'group': LabelEncoder()})
            inputs.build_folds()

            # Extract features on datasets
            process.checkpoint_step(inputs=inputs, model=extractor[1])

            # Evaluate
            process.evaluate_step(inputs=inputs, model=model[1])

        process.end()


def fine_tune(original_inputs, folder):
    # Parameters
    nb_cpu = 1
    validation = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_dermatology()
    scoring = LocalParameters.get_scorer()

    # Statistics expected
    statistics = LocalParameters.get_dermatology_statistics()

    # Filters
    filters = LocalParameters.get_dermatology_filters()

    # Layers parameters
    layers = [('first', 280),
              ('second', 249),
              ('third', 229)]

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        process = Process(output_folder=folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu, scoring=scoring)

        for layer in layers:
            inputs = original_inputs.copy_and_change(filter_groups)
            # Specify the name of experiment
            inputs.name = f'{layer[0]}'

            # Filter datasets
            inputs.set_filters(filter_datas)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_encoder), 'group': LabelEncoder()})
            inputs.build_folds()

            process.evaluate_step(inputs=inputs,
                                  model=get_fine_tuning(output_classes=len(filter_datas['Label']) - 1,  # Remove Unknown
                                                        trainable_layers=layer[1]))
        process.end()


if __name__ == "__main__":
    test = image.Reader().scan_folder('C:\\Users\\Romain\Data\\Skin\\Thumbnails')
    # Parameters
    current_file = Path(__file__)
    LocalParameters.set_gpu()
    # Input patch
    image_inputs = DermatologyDataset.images(modality='Microscopy')
    image_types = ['Patch', 'Full']
    # Folder
    output_folder = DermatologyDataset.get_results_location() / 'Deep'

    for image_type in image_types:
        inputs = image_inputs.sub_inputs({'Type': image_type})
        # Transfer Learning
        output = output_folder / 'transfer' / image_type
        output.mkdir(parents=True, exist_ok=True)
        transfer_learning(inputs, output)
        # Fine Learning
        output = output_folder / 'fine' / image_type
        output.mkdir(parents=True, exist_ok=True)
        fine_tune(inputs, output)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

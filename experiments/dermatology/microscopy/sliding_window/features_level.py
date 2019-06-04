import itertools
from os import makedirs, startfile
from os.path import exists, splitext, basename, join
from numpy import geomspace
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
    steps = []
    parameters = {}

    steps.append(('flatten', FlattenTransform()))

    # Add reduction step
    steps.append(('reduction', None))
    features = [20, 50, 100]
    kbest_p = {'reduction': SelectAtMostKBest(f_classif),
               'reduction__k': features}

    # Add scaling step
    steps.append(('scale', StandardScaler()))

    # Add classifier full
    steps.append(('clf', SVC(kernel='linear', class_weight='balanced', probability=True)))
    parameters.update({'clf__C': geomspace(0.1, 1000, 5).tolist()})

    kbest_p.update(parameters)
    m_parameters = [kbest_p]

    pipe = Pipeline(steps)
    pipe.name = 'Reduce'
    # Define parameters to validate through grid CV
    return pipe, m_parameters


def get_norm_model(patch_level=True, norm=False):
    steps = []
    parameters = {}

    # Add dimensions reducer
    p_values = [2, 3, 5, 10]

    steps.append(('norm', PNormTransform()))
    parameters.update({'norm__p': p_values})

    # Add scaling step
    steps.append(('scale', StandardScaler()))

    # Add classifier full
    steps.append(('clf', SVC(kernel='linear', class_weight='balanced', probability=True)))
    parameters.update({'clf__C': geomspace(0.1, 1000, 5).tolist()})

    pipe = Pipeline(steps)
    pipe.name = 'Norm'
    # Define parameters to validate through grid CV
    return pipe, parameters


def features_level(slidings, folder):

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

    # Extracteur
    extractor = Transforms.get_keras_extractor(pooling='max')
    extractor.need_fit = False

    # Browse combinations
    for filter_name, filter_datas, filter_encoder, filter_groups in filters:

        # Launch process
        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=4)

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

            # Collapse information and make predictions
            inputs.set_filters(filter_datas)
            features = inputs.collapse({'Type': ['Full']}, 'Reference', {'Type': ['Window']}, 'Source')

            # Evaluate using svm
            inputs.name = '{name}_reduce'.format(name=name)
            process.evaluate_step(inputs=features, model=get_reduce_model())
            inputs.name = '{name}_norm'.format(name=name)
            process.evaluate_step(inputs=features, model=get_norm_model())

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

    # # Input patch
    # windows_inputs = [('NoOverlap', DermatologyDataset.sliding_images(size=250, overlap=0)),
    #                   ('Overlap50', DermatologyDataset.sliding_images(size=250, overlap=0.50))]

    windows_inputs = [('NoOverlap', DermatologyDataset.test_sliding_images(size=250, overlap=0))]

    # Compute data
    features_level(windows_inputs, output_folder)

    # Open result folder
    startfile(output_folder)

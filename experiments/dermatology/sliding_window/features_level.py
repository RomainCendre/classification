import itertools
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from numpy import geomspace
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.builtin_models import Transforms
from toolbox.core.models import SelectAtMostKBest
from toolbox.core.transforms import OrderedEncoder, PNormTransform, FlattenTransform
from toolbox.tools.limitations import Parameters


def get_model(patch_level=True, norm=False):
    steps = []
    parameters = {}

    # Add dimensions reducer
    p_values = [2, 3, 5, 10]

    if norm:
        if patch_level:
            steps.append(('norm', PNormTransform()))
            parameters.update({'norm__p': p_values})
        else:
            steps.append(('norm1', PNormTransform(axis=2)))
            parameters.update({'norm1__p': p_values})
            steps.append(('norm2', PNormTransform()))
            parameters.update({'norm2__p': p_values})
    else:
        steps.append(('flatten', FlattenTransform()))

    # Add reduction step
    steps.append(('reduction', None))
    features = [20, 50, 100]
    kbest_p = {'reduction': SelectAtMostKBest(f_classif),
               'reduction__k': features}

    # Add scaling step
    steps.append(('scale', StandardScaler()))

    # Add classifier simple
    steps.append(('clf', SVC(kernel='linear', class_weight='balanced', probability=True)))
    parameters.update({'clf__C': geomspace(0.1, 1000, 5).tolist()})

    kbest_p.update(parameters)
    m_parameters = [kbest_p]

    pipe = Pipeline(steps)
    pipe.name = 'NormAndSVM'
    # Define parameters to validate through grid CV
    return pipe, m_parameters


if __name__ == "__main__":

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5)
    test = validation  # GroupKFold(n_splits=5)
    settings = DefinedSettings.get_default_dermatology()

    # Output folders
    output_folder = normpath('{home}/Results/Dermatology/{filename}/'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    patch_folder = join(output_folder, 'Patch')
    if not exists(patch_folder):
        makedirs(patch_folder)

    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    # Statistics expected
    statistics = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant']}, {}),
               ('NvsM', {'Label': ['Normal', 'Malignant']}, {}),
               ('NvsB', {'Label': ['Normal', 'Benign']}, {}),
               ('BvsM', {'Label': ['Benign', 'Malignant']}, {}),
               ('NvsP', {'Label': ['Normal', 'Pathology']}, {'Label': (['Benign', 'Malignant'], 'Pathology')}),
               ('MvsR', {'Label': ['Malignant', 'Rest']}, {'Label': (['Normal', 'Benign'], 'Rest')})]

    # Inputs
    inputs = [('NoOverlap', Dataset.patches_images(folder=patch_folder, size=250, overlap=0)),
              ('Overlap25', Dataset.patches_images(folder=patch_folder, size=250, overlap=0.25)),
              ('Overlap50', Dataset.patches_images(folder=patch_folder, size=250, overlap=0.50))]

    # Methods
    methods = [('Haralick', Transforms.get_haralick(mean=False)),
               ('KerasAverage', Transforms.get_keras_extractor(pooling='avg')),
               ('KerasMaximum', Transforms.get_keras_extractor(pooling='max'))]

    # Parameters combinations
    combinations = list(itertools.product(inputs, methods))

    # Image classification
    for filter_name, filter_datas, filter_groups in filters:

        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)

        for input, method in combinations:
            copy_input = input[1].copy_and_change(filter_groups)

            # Image classification
            copy_input.name = 'Image_{input}_{method}'.format(input=input[0], method=method[0])
            print('Compute {name}'.format(name=copy_input.name))
            copy_input.set_filters(filter_datas)
            copy_input.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']),
                                     'groups': LabelEncoder()})
            process.checkpoint_step(inputs=copy_input, model=method, folder=features_folder)
            copy_input.collapse(reference_tag='Reference')
            process.evaluate_step(inputs=copy_input, model=get_model(norm=True, patch_level=True))
            copy_input.name = copy_input.name + '_WithoutReduction'
            process.evaluate_step(inputs=copy_input, model=get_model(norm=False, patch_level=True))
        process.end()

    # Patient classification
    filter_name, filter_datas, filter_groups = filters[0]
    process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
    process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)
    for input, method in combinations:
        copy_input = input[1].copy_and_change(filter_groups)
        copy_input.name = 'Patient_{input}_{method}'.format(input=input[0], method=method[0])
        print('Compute {name}'.format(name=copy_input.name))
        copy_input.collapse(reference_tag='ID')
        copy_input.tags.update({'label': 'Binary_Diagnosis'})
        copy_input.set_encoders({'label': OrderedEncoder().fit(['Benign', 'Malignant']),
                                 'groups': LabelEncoder()})
        process.evaluate_step(inputs=copy_input, model=get_model(norm=True, patch_level=False))
    process.end()

    # Open result folder
    startfile(output_folder)

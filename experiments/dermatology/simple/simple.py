import itertools
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.transforms import OrderedEncoder
from toolbox.tools.limitations import Parameters

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
    output_folder = normpath('{home}/Results/Dermatology/simple/{filename}/'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Statistics expected
    statistics = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']}, {}),
               ('NvsP', {'Label': ['Normal', 'Pathology'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']}, {'Label': (['Benign', 'Malignant'], 'Pathology')}),
               ('MvsR', {'Label': ['Malignant', 'Rest'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']}, {'Label': (['Normal', 'Benign'], 'Rest')})]

    # Image filters
    scales = [('Full', {'Type': 'Full'}), ('Thumbnails', {'Type': 'Patch'})]

    # Input patch
    inputs = Dataset.images()

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

        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)

        for method, model in combinations:

            for scale in scales:
                copy_input = inputs.copy_and_change(filter_groups)
                copy_input.name = '{input}_{method}_{model}'.format(input=scale[0], method=method[0], model=model[0])

                filter_datas.update(scale[1])
                copy_input.set_filters(filter_datas)
                copy_input.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']), 'groups': LabelEncoder()})
                process.checkpoint_step(inputs=copy_input, model=method[1], folder=features_folder)
                process.evaluate_step(inputs=copy_input, model=model[1])
        process.end()

    # Open result folder
    startfile(output_folder)

import itertools
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join

from numpy import geomspace
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.transforms import PredictorTransform, OrderedEncoder, FlattenTransform
from toolbox.tools.limitations import Parameters


def get_linear_svm():
    pipe = Pipeline([('flatten', FlattenTransform()),
                     ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
    pipe.name = 'LinearSVM'
    return pipe, {'clf__C': geomspace(0.01, 1000, 6).tolist()}


if __name__ == "__main__":
    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Parameters
    test_mode = True
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5, shuffle=True)
    test = validation  # GroupKFold(n_splits=5)
    settings = DefinedSettings.get_default_dermatology()

    # Output folders
    output_folder = normpath('{home}/Results/Dermatology/SVM/{filename}/'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # View
    view_folder = join(output_folder, 'View')
    if not exists(view_folder):
        makedirs(view_folder)

    # Inputs
    if not test_mode:
        train_inputs = Dataset.images()
        slidings_inputs = [('NoOverlap', Dataset.sliding_images(size=250, overlap=0)),
                           ('Overlap50', Dataset.sliding_images(size=250, overlap=0.50))]
    else:
        train_inputs = Dataset.test_images()
        slidings_inputs = [('NoOverlap', Dataset.test_sliding_images(size=250, overlap=0))]
    train_inputs = train_inputs.sub_inputs({'Type': 'Patch'})

    # Statistics expected
    statistics = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    # Models
    models = [('Svm', Classifiers.get_linear_svm())]

    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']}, {}),
               ('NvsP', {'Label': ['Normal', 'Pathology'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']}, {'Label': (['Benign', 'Malignant'], 'Pathology')}),
               ('MvsR', {'Label': ['Malignant', 'Rest'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']}, {'Label': (['Normal', 'Benign'], 'Rest')})]

    # Methods
    methods = Transforms.get_keras_extractor(pooling='avg')

    # Parameters combinations
    combinations = list(itertools.product(slidings_inputs, models))

    # Browse combinations
    for filter_name, filter_datas, filter_groups in filters:

        # Launch process
        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)

        for sliding, model in combinations:
            # Name experiment and filter data
            name = '{sliding}_{model}'.format(sliding=sliding[0], model=model[0])

            pre_inputs = train_inputs.copy_and_change(filter_groups)
            pre_inputs.set_filters(filter_datas)
            pre_inputs.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']), 'groups': LabelEncoder()})

            inputs = sliding[1].copy_and_change(filter_groups)
            inputs.set_filters(filter_datas)
            inputs.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']), 'groups': LabelEncoder()})
            inputs.name = name

            # Pretrain
            process.checkpoint_step(inputs=pre_inputs, model=methods)
            predictor, params = process.train_step(inputs=pre_inputs, model=model[1])

            # Now predict
            process.checkpoint_step(inputs=inputs, model=methods)
            process.checkpoint_step(inputs=inputs, model=PredictorTransform(predictor, probabilities=True))

            # Collapse informations and make predictions
            inputs.collapse(reference_tag='Reference')
            process.evaluate_step(inputs=inputs, model=get_linear_svm())

        process.end()

    # Open result folder
    startfile(output_folder)

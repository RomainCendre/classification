import itertools
from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold, GroupKFold
from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.transforms import PredictorTransform
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":
    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5, shuffle=True)
    test = validation  # GroupKFold(n_splits=5)
    settings = DefinedSettings.get_default_dermatology()

    # Output folders
    features_folder = join(home_path, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    output_folder = normpath('{home}/Results/Dermatology/SVM/{filename}/'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Statistics expected
    statistics = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']}, {}),
               ('NvsP', {'Label': ['Normal', 'Pathology'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']},
                {'Label': (['Benign', 'Malignant'], 'Pathology')}),
               ('MvsR', {'Label': ['Malignant', 'Rest'], 'Diagnosis': ['LM/LMM', 'SL', 'AL']},
                {'Label': (['Normal', 'Benign'], 'Rest')})]

    # Inputs
    train_inputs = Dataset.images()
    train_inputs = train_inputs.sub_inputs({'Type': 'Patch'})

    inputs = [('NoOverlap', Dataset.patches_images(size=250, overlap=0)),
              ('Overlap25', Dataset.patches_images(size=250, overlap=0.25)),
              ('Overlap50', Dataset.patches_images(size=250, overlap=0.50))]

    # Methods
    methods = ('KerasAverage', Transforms.get_keras_extractor(pooling='avg'))

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=test, n_jobs=2, settings=settings)

    # Parameters combinations
    combinations = list(itertools.product(filters, methods))

    for combination in combinations:
        working_inputs = deepcopy(input)
        name = '{method}'.format(method=method[0])
        # Pretrain
        process.checkpoint_step(inputs=pretrain_input, model=method[1], folder=features_folder)
        predictor, params = process.train_step(inputs=pretrain_input, model=Classifiers.get_linear_svm())
        # Now predict
        process.checkpoint_step(inputs=working_inputs, model=method[1], folder=features_folder,
                                projection_folder=projection_folder, projection_name=name)
        process.checkpoint_step(inputs=working_inputs, model=PredictorTransform(predictor, probabilities=True),
                                folder=features_folder)
        working_inputs.patch_method()
        process.end(inputs=working_inputs, model=Classifiers.get_linear_svm(), output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

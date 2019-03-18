import itertools
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold, GroupKFold
from experiments.processes import Process
from toolbox.IO.datasets import Dataset
from toolbox.core.models import Transforms, Classifiers
from toolbox.core.structures import Settings
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":
    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5, shuffle=True)
    test = validation  # GroupKFold(n_splits=5)

    # Output folders
    output_folder = normpath('{home}/Results/Dermatology/SVM/{filename}/'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Parameters
    settings = Settings({'patches': dict(Malignant=[255, 0, 0], Benign=[125, 125, 0], Normal=[0, 255, 0]),
                         'draw': dict(Malignant=(1, 0, 0), Benign=(0.5, 0.5, 0), Normal=(0, 1, 0))})
    # Inputs
    inputs = [('Thumbnails', Dataset.thumbnails()),
              ('FullImages', Dataset.full_images())]

    # Methods
    methods = [('Haralick', Transforms.get_haralick(mean=False)),
               ('HaralickMean', Transforms.get_haralick(mean=True)),
               ('Wavelet', Transforms.get_image_dwt()),
               ('KerasAverage', Transforms.get_keras_extractor(pooling='avg')),
               ('KerasMaximum', Transforms.get_keras_extractor(pooling='max'))]

    # Models
    models = [('Svm', Classifiers.get_linear_svm()),
              ('SvmPca', Classifiers.get_linear_svm(reduce=20))]

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=test, n_jobs=2)

    combinations = list(itertools.product(inputs, methods, models))
    for combination in combinations:
        input, method, model = combination
        name = '{input}_{method}_{model}'.format(input=input[0], method=method[0], model=model[0])
        process.checkpoint_step(inputs=input[1], model=method[1], folder=features_folder,
                                projection_folder=projection_folder, projection_name=name)
        process.end(inputs=inputs, model=model[1], output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

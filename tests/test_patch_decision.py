import itertools
from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, splitext, basename, join, dirname
from tempfile import gettempdir
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.IO.writers import PatchWriter
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.transforms import PredictorTransform, OrderedEncoder

if __name__ == "__main__":
    # Parameters
    filename = splitext(basename(__file__))[0]
    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = filename
    validation = StratifiedKFold(n_splits=2, shuffle=True)
    settings = DefinedSettings.get_default_dermatology()

    # Output dir
    output_folder = normpath('{temp}/dermatology/{filename}'.format(temp=temp_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Feature folder
    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # View
    view_folder = join(output_folder, 'View')
    if not exists(view_folder):
        makedirs(view_folder)


    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant']})]

    # Input data
    pretrain_input = Dataset.test_thumbnails()
    pretrain_input.set_encoders({'label': OrderedEncoder().fit(['Normal', 'Malignant']),
                         'groups': LabelEncoder()})
    inputs = Dataset.test_patches_images(size=250, overlap=0.25)
    inputs.set_encoders({'label': OrderedEncoder().fit(['Normal', 'Malignant']),
                         'groups': LabelEncoder()})

    # Methods
    methods = Transforms.get_keras_extractor(pooling='max')

    # Launch process
    keys = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']
    process = Process(output_folder=output_folder, name=name, settings=settings, stats_keys=keys)
    process.begin(inner_cv=validation, outer_cv=validation, n_jobs=2)

    # Pretrain
    process.checkpoint_step(inputs=pretrain_input, model=methods, folder=features_folder)
    predictor, params = process.train_step(inputs=pretrain_input, model=Classifiers.get_linear_svm())

    # Now predict
    inputs.name = 'PredictionTL'
    process.checkpoint_step(inputs=inputs, model=methods, folder=features_folder)
    transform = PredictorTransform(predictor, probabilities=False)
    process.checkpoint_step(inputs=inputs, model=transform,
                            folder=features_folder)
    PatchWriter(inputs, settings).write_patch(folder=view_folder)
    inputs.collapse(reference_tag='Reference')
    process.evaluate_step(inputs=inputs, model=Classifiers.get_linear_svm())

    process.end()

    # Open result folder
    startfile(output_folder)

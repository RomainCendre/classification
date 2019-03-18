from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.IO.datasets import Dataset
from toolbox.core.models import Classifiers, Transforms
from toolbox.core.structures import Settings

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = filename
    validation = StratifiedKFold(n_splits=2, shuffle=True)
    settings = Settings({'patches': dict(Malignant=[255, 0, 0], Benign=[125, 125, 0], Normal=[0, 255, 0]),
                         'draw': dict(Malignant=(1, 0, 0), Benign=(0.5, 0.5, 0), Normal=(0, 1, 0))})

    # Output dir
    output_folder = normpath('{temp}/dermatology/{filename}'.format(temp=temp_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Feature folder
    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Input data
    inputs = Dataset.test_full_images()

    # Initiate model and params
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation, settings=settings)
    process.checkpoint_step(inputs=inputs, model=Transforms.get_haralick(), folder=features_folder)
    process.end(inputs=inputs, model=Classifiers.get_dummy_simple(), output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

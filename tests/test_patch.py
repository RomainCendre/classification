from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.models import Classifiers, Transforms
from toolbox.core.transforms import OrderedEncoder

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
    patch_folder = join(output_folder, 'Patch')
    if not exists(patch_folder):
        makedirs(patch_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Input data
    inputs = Dataset.test_patches_images(patch_folder, 250)
    inputs.set_filters({'Label': ['Normal', 'Benign', 'Malignant']})
    inputs.set_encoders({'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                         'groups': LabelEncoder()})

    # Initiate model and params
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation, settings=settings)
    process.checkpoint_step(inputs=inputs, model=Transforms.get_haralick(), folder=features_folder)
    inputs.collapse(reference_tag='Reference', data_tag='ImageData', flatten=False)
    process.end(inputs=inputs, model=Classifiers.get_patch_model(), output_folder=output_folder, name='Images_{name}'.format(name=name))
    inputs.collapse(reference_tag='ID', data_tag='PatientData', flatten=False)
    inputs.tags.update({'label': 'Binary_Diagnosis'})
    inputs.set_encoders({'label': OrderedEncoder().fit(['Benign', 'Malignant']),
                         'groups': LabelEncoder()})
    process.end(inputs=inputs, model=Classifiers.get_patch_model(patch=False), output_folder=output_folder, name='Patient_{name}'.format(name=name))

    # Open result folder
    startfile(output_folder)

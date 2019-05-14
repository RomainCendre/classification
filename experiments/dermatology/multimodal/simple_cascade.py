from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.builtin_models import Transforms, Classifiers
from toolbox.core.transforms import OrderedEncoder
from toolbox.core.parameters import Parameters

if __name__ == "__main__":

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5)
    test = validation
    settings = DefinedSettings.get_default_dermatology()

    # Output folders
    output_folder = normpath('{home}/Results/MultiModal/{filename}/'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    # Filters
    filters = {'Label': ['Normal', 'Benign', 'Malignant']}

    # Input patch
    inputs = Dataset.full_images()

    # Methods
    methods = Transforms.get_keras_extractor(pooling='max')

    # Models
    pipe, parameters = Classifiers.get_linear_svm()


    inputs.name = 'Multimodal'
    inputs.set_filters(filters)
    inputs.set_encoders({'label': OrderedEncoder().fit(filters['Label']),
                         'groups': LabelEncoder()})

    # Open result folder
    startfile(output_folder)

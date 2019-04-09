import itertools
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join

from numpy import geomspace
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.IO.writers import PCAProjection
from toolbox.core.builtin_models import Transforms
from toolbox.core.transforms import OrderedEncoder, PNormTransform
from toolbox.tools.limitations import Parameters


def get_linear_svm():
    pipe = Pipeline([('scale', StandardScaler()),
                     ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
    pipe.name = 'LinearSVM'
    return pipe, {'clf__C': geomspace(0.01, 100, 5).tolist()}


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

    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    # Statistics expected
    statistics = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant']}, {}),
               ('NvsP', {'Label': ['Normal', 'Pathology']}, {'Label': (['Benign', 'Malignant'], 'Pathology')}),
               ('MvsR', {'Label': ['Malignant', 'Rest']}, {'Label': (['Normal', 'Benign'], 'Rest')})]

    # Inputs
    inputs = [('NoOverlap', Dataset.patches_images(size=250, overlap=0)),
              ('Overlap25', Dataset.patches_images(size=250, overlap=0.25)),
              ('Overlap50', Dataset.patches_images(size=250, overlap=0.50))]

    # Methods
    descriptors = [('Haralick', Transforms.get_haralick(mean=False)),
                   ('KerasAverage', Transforms.get_keras_extractor(pooling='avg'))]

    merges = [('Norm2', PNormTransform(p=2)),
              ('Norm5', PNormTransform(p=5))]

    # Parameters combinations
    combinations = list(itertools.product(inputs, descriptors, merges))

    # Image classification
    for filter_name, filter_datas, filter_groups in filters:

        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)
        pca_projector = PCAProjection(settings=settings, dir_name=output_folder, name=filter_name)
        for input, descriptor, merge in combinations:
            copy_input = input[1].copy_and_change(filter_groups)

            # Image classification
            copy_input.name = 'Image_{input}_{descriptor}_{merge}'.format(input=input[0], descriptor=descriptor[0], merge=merge[0])
            print(copy_input.name)
            copy_input.set_filters(filter_datas)
            copy_input.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']),
                                     'groups': LabelEncoder()})
            process.checkpoint_step(inputs=copy_input, model=descriptor[1], folder=features_folder)
            copy_input.collapse(reference_tag='Reference')
            process.checkpoint_step(inputs=copy_input, model=merge[1])
            pca_projector.write_projection(copy_input)
            process.evaluate_step(inputs=copy_input, model=get_linear_svm())
        process.end()
        pca_projector.end()

    startfile(output_folder)

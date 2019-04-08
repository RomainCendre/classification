import itertools
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join

from numpy import geomspace, array
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.IO.writers import PCAProjection, PatchWriter
from toolbox.core.builtin_models import Transforms
from toolbox.core.models import PatchClassifier
from toolbox.core.transforms import OrderedEncoder, PNormTransform, PredictorTransform, FlattenTransform
from toolbox.tools.limitations import Parameters


def get_linear_svm():
    pipe = Pipeline([('scale', StandardScaler()),
                     ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
    pipe.name = 'LinearSVM'
    return pipe, {'clf__C': geomspace(0.01, 100, 5).tolist()}


def get_patch_classifier(hierarchies):
    pipe = Pipeline([('clf', PatchClassifier(hierarchies=hierarchies))])
    pipe.name = 'PatchClassifier'
    return pipe, {}


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
               ('MvsR', {'Label': ['Rest', 'Malignant']}, {'Label': (['Normal', 'Benign'], 'Rest')})]

    # Inputs
    pretrain = Dataset.thumbnails()
    inputs = [('NoOverlap', Dataset.patches_images(size=250, overlap=0)),
              ('Overlap25', Dataset.patches_images(size=250, overlap=0.25))]

    # Methods
    descriptors = [('Haralick', Transforms.get_haralick(mean=False)),
                   ('KerasAverage', Transforms.get_keras_extractor(pooling='avg'))]

    # Parameters combinations
    combinations = list(itertools.product(inputs, descriptors))

    # Image classification
    for filter_name, filter_datas, filter_groups in filters:

        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)
        pca_projector = PCAProjection(settings=settings, dir_name=output_folder, name=filter_name)
        for input, descriptor in combinations:
            copy_pretrain = pretrain.copy_and_change(filter_groups)
            copy_pretrain.name = 'Image_{input}_{descriptor}'.format(input=input[0], descriptor=descriptor[0])
            copy_pretrain.set_filters(filter_datas)
            copy_pretrain.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']),
                                        'groups': LabelEncoder()})
            # Pretrain
            process.checkpoint_step(inputs=copy_pretrain, model=descriptor[1], folder=features_folder)
            predictor, params = process.train_step(inputs=copy_pretrain, model=get_linear_svm())

            # Image classification
            copy_input = input[1].copy_and_change(filter_groups)
            copy_input.name = 'Image_{input}_{descriptor}'.format(input=input[0], descriptor=descriptor[0])
            copy_input.set_filters(filter_datas)
            copy_input.set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']),
                                     'groups': LabelEncoder()})
            process.checkpoint_step(inputs=copy_input, model=descriptor[1], folder=features_folder)
            process.checkpoint_step(inputs=copy_input, model=PredictorTransform(predictor, probabilities=False))
            PatchWriter(copy_input, settings).write_patch(folder=join(output_folder, filter_name))

            copy_input.collapse(reference_tag='Reference')
            pca_projector.write_projection(copy_input)
            hierarchies = copy_input.encode('label', array(list(reversed(filter_datas['Label']))))
            process.evaluate_step(inputs=copy_input, model=get_patch_classifier(hierarchies))
            process.evaluate_step(inputs=copy_input, model=get_linear_svm())
        process.end()
        pca_projector.end()

    startfile(output_folder)

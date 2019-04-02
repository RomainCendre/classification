import itertools
from os import makedirs, startfile
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from os.path import exists, expanduser, normpath, basename, splitext, join

from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.IO.datasets import DefinedSettings, Dataset
from toolbox.core.builtin_models import BuiltInModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.core.transforms import OrderedEncoder
from toolbox.tools.limitations import Parameters


if __name__ == '__main__':
    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5)
    test = validation  # GroupKFold(n_splits=5)
    settings = DefinedSettings.get_default_dermatology()

    # Parameters
    layers_parameters = {'trainable_layer': [0, 1, 2],
                         'added_layer': [1, 2, 3, 4]}

    # Output folders
    output_folder = normpath('{home}/Results/Dermatology/{filename}/'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Filters
    filters = [('All', {'Label': ['Normal', 'Benign', 'Malignant']}, {}),
               ('NvsM', {'Label': ['Normal', 'Malignant']}, {}),
               ('NvsB', {'Label': ['Normal', 'Benign']}, {}),
               ('BvsM', {'Label': ['Benign', 'Malignant']}, {}),
               ('NvsP', {'Label': ['Normal', 'Pathology']}, {'Label': (['Benign', 'Malignant'], 'Pathology')}),
               ('MvsR', {'Label': ['Malignant', 'Rest']}, {'Label': (['Normal', 'Benign'], 'Rest')})]

    # Input patch
    inputs = [('Thumbnails', Dataset.thumbnails()),
              ('Full', Dataset.full_images())]

    # Launch process
    keys = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    # Parameters combinations
    combinations = list(itertools.product(inputs, ParameterGrid(layers_parameters)))

    for filter_name, filter_datas, filter_groups in filters:

        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=keys)
        process.begin(inner_cv=validation, outer_cv=test, n_jobs=4)

        for combination in combinations:
            try:
                input, params = combination
                # Patient classification
                input[1].name = 'Image_{input}_{params}'.format(input=input[0], params=params)
                input[1].set_encoders({'label': OrderedEncoder().fit(filter_datas['Label']),
                                       'groups': LabelEncoder()})
                process.evaluate_step(inputs=input,
                                      model=BuiltInModels.get_fine_tuning(output_classes=3,
                                                                       trainable_layers=params['trainable_layer'],
                                                                       added_layers=params['added_layer']))
            except:
                print('Error occured.')
        process.end()

    # Open result folder
    startfile(output_folder)
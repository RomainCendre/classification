from os import makedirs, startfile
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from os.path import exists, expanduser, normpath, basename, splitext, join
from experiments.processes import Process
from toolbox.core.models import BuiltInModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters


def launch_computation(inputs, inner_cv, outer_cv, folder, name, layers_parameters):
    # Browse combinations
    for parameters in ParameterGrid(layers_parameters):
        print('Current layers : trainable {trainable}, added {added}'.format(trainable=parameters['trainable_layer'],
                                                                             added=parameters['added_layer']))

        # Features folder
        temp_folder = join(output_folder, 'Features')
        if not exists(temp_folder):
            makedirs(temp_folder)

        # Projection folder
        projection_folder = join(output_folder, 'Projection')
        if not exists(projection_folder):
            makedirs(projection_folder)

        process = Process()
        process.begin(inner_cv=inner_cv, outer_cv=outer_cv)
        process.end(inputs=inputs, model=BuiltInModels.get_fine_tuning(output_classes=3,
                                                                       trainable_layers=parameters[
                                                                           'trainable_layer'],
                                                                       added_layers=parameters['added_layer']),
                    output_folder=folder, name=name)


if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    name_patch = 'Patch'
    validation = StratifiedKFold(n_splits=5)

    # Output dir
    output_folder = normpath(
        '{home}/Results/Dermatology/Deep/Transfer_learning/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    ################# PATCH
    # Input patch
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    filter_by = {'Label': ['Malignant', 'Benign', 'Normal']}
    inputs = Inputs(folders=[input_folder], instance=dermatology.Reader(), filter_by=filter_by,
                    loader=dermatology.Reader.scan_folder_for_images,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'})
    inputs.load()

    layers_parameters = {'trainable_layer': [0, 1, 2],
                         'added_layer': [1, 2, 3]}
    launch_computation(inputs=inputs, inner_cv=validation, outer_cv=validation, folder=output_folder,
                       name=name_patch, layers_parameters=layers_parameters)

    ################# FULL
    name_full = 'Full'
    # Input full
    # filter_by = {'Modality': 'Microscopy',
    #              'Label': ['Malignant', 'Benign', 'Normal']}
    # input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
    #                  normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    # inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
    #                 tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'}, filter_by=filter_by)
    # inputs.load()
    #
    # # Launch process
    # process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(), folder=temp_folder,
    #                         projection_folder=projection_folder, projection_name=name_full)
    # process.end(inputs=inputs, model=Classifiers.get_keras_classifier(output_classes=3),
    #             output_folder=output_folder, name=name_full)

    # Open result folder
    startfile(output_folder)

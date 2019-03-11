from os import makedirs, startfile
from sklearn.model_selection import StratifiedKFold, GroupKFold
from os.path import exists, expanduser, normpath, basename, splitext, join

from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.core.models import Transforms, Classifiers
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.core.transforms import OrderedEncoder
from toolbox.tools.limitations import Parameters

if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    validation = StratifiedKFold(n_splits=5, shuffle=True)
    test = validation #GroupKFold(n_splits=5)
    # Parameters
    colors = {'patches': dict(Malignant=[255, 0, 0], Benign=[125, 125, 0], Normal=[0, 255, 0]),
              'draw': dict(Malignant=(1, 0, 0), Benign=(0.5, 0.5, 0), Normal=(0, 1, 0))}

    # Output dir
    output_folder = normpath(
        '{home}/Results/Dermatology/SVM/Transfer_learning/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    ################# AVERAGE
    pooling = 'avg'
    ################# Normal
    name_patch = 'Patch_avg'
    name_full = 'Full_avg'
    # Features folder
    temp_folder = join(output_folder, 'Features_avg')
    if not exists(temp_folder):
        makedirs(temp_folder)

    # Projection folder
    projection_folder = join(output_folder, 'Projection_avg')
    if not exists(projection_folder):
        makedirs(projection_folder)

    ################# PATCH
    # Input patch
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs = Inputs(folders=[input_folder], instance=dermatology.Reader(), style=colors,
                    loader=dermatology.Reader.scan_folder_for_images,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'},
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=test, n_jobs=2)
    # process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(pooling=pooling), folder=temp_folder,
    #                         projection_folder=projection_folder, projection_name=name_patch)
    # process.end(inputs=inputs, model=Classifiers.get_linear_svm(), output_folder=output_folder, name=name_patch)

    ################# FULL
    # Input full
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder, style=colors,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference', 'groups': 'ID'}, filter_by=filter_by,
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # Launch process
    process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(pooling=pooling), folder=temp_folder,
                            projection_folder=projection_folder, projection_name=name_full)
    process.end(inputs=inputs, model=Classifiers.get_linear_svm(), output_folder=output_folder, name=name_full)

    ################# PCA
    name_patch = 'Patch_avg_pca'
    name_full = 'Full_avg_pca'
    # Features folder
    temp_folder = join(output_folder, 'Features_avg_PCA')
    if not exists(temp_folder):
        makedirs(temp_folder)

    # Projection folder
    projection_folder = join(output_folder, 'Projection_avg_PCA')
    if not exists(projection_folder):
        makedirs(projection_folder)
    ################# PATCH PCA
    # Input patch
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs = Inputs(folders=[input_folder], instance=dermatology.Reader(), style=colors,
                    loader=dermatology.Reader.scan_folder_for_images,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'},
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # # Launch process
    # process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(pooling=pooling), folder=temp_folder,
    #                         projection_folder=projection_folder, projection_name=name_patch)
    # process.end(inputs=inputs, model=Classifiers.get_linear_svm(reduce=20), output_folder=output_folder, name=name_patch)

    ################# FULL PCA
    # Input full
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                    style=colors,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference', 'groups': 'ID'},
                    filter_by=filter_by,
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # Launch process
    process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(pooling=pooling), folder=temp_folder,
                            projection_folder=projection_folder, projection_name=name_full)
    process.end(inputs=inputs, model=Classifiers.get_linear_svm(reduce=20), output_folder=output_folder, name=name_full)

    ################# MAXIMUM
    pooling = 'max'
    ################# Normal
    name_patch = 'Patch_max'
    name_full = 'Full_max'
    # Features folder
    temp_folder = join(output_folder, 'Features_max')
    if not exists(temp_folder):
        makedirs(temp_folder)

    # Projection folder
    projection_folder = join(output_folder, 'Projection_max')
    if not exists(projection_folder):
        makedirs(projection_folder)

    ################# PATCH
    # Input patch
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs = Inputs(folders=[input_folder], instance=dermatology.Reader(), style=colors,
                    loader=dermatology.Reader.scan_folder_for_images,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'},
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # Launch process
    # process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(pooling=pooling), folder=temp_folder,
    #                         projection_folder=projection_folder, projection_name=name_patch)
    # process.end(inputs=inputs, model=Classifiers.get_linear_svm(), output_folder=output_folder, name=name_patch)

    ################# FULL
    # Input full
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                    style=colors,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference', 'groups': 'ID'},
                    filter_by=filter_by,
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # Launch process
    process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(pooling=pooling), folder=temp_folder,
                            projection_folder=projection_folder, projection_name=name_full)
    process.end(inputs=inputs, model=Classifiers.get_linear_svm(), output_folder=output_folder, name=name_full)

    ################# PCA
    name_patch = 'Patch_max_pca'
    name_full = 'Full_max_pca'
    # Features folder
    temp_folder = join(output_folder, 'Features_max_PCA')
    if not exists(temp_folder):
        makedirs(temp_folder)

    # Projection folder
    projection_folder = join(output_folder, 'Projection_max_PCA')
    if not exists(projection_folder):
        makedirs(projection_folder)
    ################# PATCH PCA
    # Input patch
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs = Inputs(folders=[input_folder], instance=dermatology.Reader(), style=colors,
                    loader=dermatology.Reader.scan_folder_for_images,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'},
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # # Launch process
    # process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(pooling=pooling), folder=temp_folder,
    #                         projection_folder=projection_folder, projection_name=name_patch)
    # process.end(inputs=inputs, model=Classifiers.get_linear_svm(reduce=20), output_folder=output_folder, name=name_patch)

    ################# FULL PCA
    # Input full
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                    style=colors,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference', 'groups': 'ID'},
                    filter_by=filter_by,
                    encoders={'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant']),
                              'groups': LabelEncoder()})
    inputs.load()

    # Launch process
    process.checkpoint_step(inputs=inputs, model=Transforms.get_keras_extractor(pooling=pooling), folder=temp_folder,
                            projection_folder=projection_folder, projection_name=name_full)
    process.end(inputs=inputs, model=Classifiers.get_linear_svm(reduce=20), output_folder=output_folder, name=name_full)

    # Open result folder
    startfile(output_folder)




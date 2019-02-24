from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.core.models import Transforms, Classifiers
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    name_patch = 'Patch'
    name_full = 'Full'
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Dermatology/DWT/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Temporary folder
    temp_folder = join(output_folder, 'Features')
    if not exists(temp_folder):
        makedirs(temp_folder)

    # Projection folder
    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    ################# PATCH
    # Input patch
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs_patch = Inputs(folders=[input_folder], instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder_for_images,
                          tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'})
    inputs_patch.load()

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation)
    process.checkpoint_step(inputs=inputs_patch, model=Transforms.get_image_dwt(), folder=temp_folder,
                            projection_folder=projection_folder, projection_name=name_patch)
    process.end(inputs=inputs_patch, model=Classifiers.get_linear_svm(), output_folder=output_folder, name=name_patch)

    ################# FULL
    # Input full
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs_full = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                         tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'},
                         filter_by=filter_by)
    inputs_full.load()

    # Launch process
    process.checkpoint_step(inputs=inputs_full, model=Transforms.get_image_dwt(), folder=temp_folder,
                            projection_folder=projection_folder, projection_name=name_full)
    process.end(inputs=inputs_full, model=Classifiers.get_linear_svm(), output_folder=output_folder, name=name_full)

    # Open result folder
    startfile(output_folder)

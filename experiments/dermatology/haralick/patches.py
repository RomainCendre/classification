from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.IO.writers import DataProjectorWriter
from toolbox.core.models import SimpleModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.core.transforms import HaralickDescriptorTransform


if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    name = 'Results'
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Dermatology/Haralick/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Temporary folder
    temp_folder = join(output_folder, 'Temp')
    if not exists(temp_folder):
        makedirs(temp_folder)

    # Projection folder
    projection_folder = join(output_folder, 'Projection')
    if not exists(projection_folder):
        makedirs(projection_folder)

    # Input data
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs = Inputs(folders=[input_folder], loader=dermatology.Reader.scan_folder_for_images,
                    tags={'data_tag': 'Data', 'label_tag': 'Label'})
    inputs.load()

    # Write data to visualize it
    DataProjectorWriter.project_data(inputs, output_folder)

    # Initiate model and params
    model, params = SimpleModels.get_linear_svm_process()

    # Launch process
    process = Process()
    process.begin(inner_cv=validation, outer_cv=validation)
    process.checkpoint_step(inputs=inputs, model=HaralickDescriptorTransform(), folder=temp_folder,
                            projection_folder=projection_folder)
    process.end(inputs=inputs, model=model, params=params, output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)


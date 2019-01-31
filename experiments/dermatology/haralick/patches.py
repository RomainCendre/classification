import mahotas
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename
from PIL import Image
from numpy.ma import array
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Processes
from toolbox.core.models import SimpleModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology


def extract_haralick(inputs):

    for data in inputs.data.data_set:
        image = array(Image.open(data.data['Data']))
        data.update({'Haralick': mahotas.features.haralick(image[:, :, 0]).flatten()})


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

    # Input data
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs = Inputs(folders=[input_folder], loader=dermatology.Reader.scan_folder_for_images,
                    tags={'data_tag': 'Haralick', 'label_tag': 'Label'})
    inputs.load()

    # Format Data
    extract_haralick(inputs)

    # Write data to visualize it
    # DataProjector.project_data(datas=features, labels=labels, path=join(output_dir, 'Projector'))

    # Initiate model and params
    model, params = SimpleModels.get_linear_svm_process()
    params.update({'inner_cv': validation,
                   'outer_cv': validation})

    # Launch process
    Processes.dermatology(inputs, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)


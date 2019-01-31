import mahotas
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename
from PIL import Image
from numpy.ma import array, asarray
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.processes import Processes
from toolbox.core.models import SimpleModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology


def extract_haralick(inputs):
    files = inputs.get_datas()
    labels = inputs.get_labels()

    features = []
    for file, label in zip(files, labels):
        image = array(Image.open(file))
        features.append(mahotas.features.haralick(image[:, :, 0]).flatten())

    features = asarray(features)
    return features, labels


if __name__ == "__main__":

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
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'LB', 'Normal']}

    input_folders = [normpath('{home}/Data/Skin/Saint_Etienne/Elisa_DB/Patients'.format(home=home_path)),
                     normpath('{home}/Data/Skin/Saint_Etienne/Hors_DB/Patients'.format(home=home_path))]
    inputs = Inputs(folders=input_folders, loader=dermatology.Reader.scan_folder,
                    tags={'data_tag': 'Data', 'label_tag': 'Label'}, filter_by=filter_by)
    inputs.load()

    # Format Data
    features, labels = extract_haralick(inputs)
    features = StandardScaler().fit_transform(features)

    # Write data to visualize it
    # DataProjector.project_data(datas=features, labels=labels, path=join(output_dir, 'Projector'))

    # Initiate model and params
    model, params = SimpleModels.get_pca_process()
    params.update({'inner_cv': validation,
                   'outer_cv': validation})

    # Launch process
    Processes.dermatology(inputs, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)

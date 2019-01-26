from os import makedirs, startfile
from sklearn.model_selection import StratifiedKFold
from os.path import exists, expanduser, normpath, basename, splitext, join
from experiences.processes import Processes
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels, SimpleModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser("~")
    name = filename
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Otorhino/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Load data
    input_folder = normpath('{home}/Data/Neck/'.format(home=home_path))
    inputs = Inputs(folders=[join(input_folder, 'Patients.csv'), join(input_folder, 'Temoins.csv')],
                    loader=dermatology.Reader.scan_folder,
                    tags={'data_tag': 'Data', 'label_tag': 'Label'})

    # Filters
    data_filters = {
        'Results_All': {},
        'Results_SvsC': {'label': ['Sain', 'Cancer']},
        'Results_SvsP': {'label': ['Sain', 'Precancer']},
        'Results_PvsC': {'label': ['Precancer', 'Cancer']},
    }

    # Get experiences
    model, params = SimpleModels.get_pls_process()

    for item_name, item_filter in data_filters.items():
        inputs.se
        Processes.otorhinolaryngology(inputs=inputs, output_folder=output_folder, model=model,
                                      params=params, name=item_name)

    # Open result folder
    startfile(output_folder)

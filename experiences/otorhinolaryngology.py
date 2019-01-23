from os import makedirs, startfile
from os.path import expanduser, normpath, join, exists

from experiences.processes import Processes
from toolbox.core.models import SimpleModels

if __name__ == "__main__":

    home_path = expanduser("~")

    # Output dir
    output_folder = normpath('{home}/Results/Neck/'.format(home=home_path))
    if not exists(output_folder):
        makedirs(output_folder)

    # Load data
    input_folder = normpath('{home}/Data/Neck/'.format(home=home_path))
    input_folders = [join(input_folder, 'Patients.csv'), join(input_folder, 'Temoins.csv')]

    # Filters
    data_filters = {
        'Results_All': {},
        'Results_SvsC': {'label': ['Sain', 'Cancer']},
        'Results_SvsP': {'label': ['Sain', 'Precancer']},
        'Results_PvsC': {'label': ['Precancer', 'Cancer']},
    }

    # Get experiences
    pipe, param = SimpleModels.get_pls_process()
    learner = {'Model': pipe,
               'Parameters': param}

    for item_name, item_filter in data_filters.items():
        Processes.otorhinolaryngology(input_folders=input_folders, output_folder=output_folder,
                                      name=item_name, filter_by=item_filter, learner=learner)

    # Open result folder
    startfile(output_folder)

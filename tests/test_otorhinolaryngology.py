from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, join, dirname
from experiences.processes import Processes
from toolbox.core.models import SimpleModels

if __name__ == "__main__":

    here_path = dirname(__file__)
    temp_path = gettempdir()

    # Output dir
    output_folder = normpath('{temp}/spectroscopy/'.format(temp=temp_path))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    input_folder = normpath('{here}/data/spectroscopy'.format(here=here_path))
    input_folders = [join(input_folder, 'Patients.csv')]

    # Filters
    data_filters = {
        'Results_SvsC': {'label': ['Sain', 'Cancer']}
    }

    # Get experiences
    pipe, param = SimpleModels.get_dummy_process()
    learner = {'Model': pipe,
               'Parameters': param}

    for item_name, item_filter in data_filters.items():
        Processes.otorhinolaryngology(input_folders=input_folders, output_folder=output_folder,
                                      name=item_name, filter_by=item_filter, learner=learner)

    # Open result folder
    startfile(output_folder)


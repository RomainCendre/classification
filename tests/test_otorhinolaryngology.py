from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, join, dirname

from sklearn.model_selection import StratifiedKFold

from experiences.processes import Processes
from toolbox.core.models import SimpleModels

if __name__ == "__main__":

    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = 'OrlTest'
    epochs = 1
    batch_size = 10
    validation = StratifiedKFold(n_splits=2, shuffle=True)

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
    model, params = SimpleModels.get_dummy_process()

    for item_name, item_filter in data_filters.items():
        Processes.otorhinolaryngology(input_folders=input_folders, filter_by=item_filter, output_folder=output_folder,
                                      model=model, params=params, name=item_name)

    # Open result folder
    startfile(output_folder)

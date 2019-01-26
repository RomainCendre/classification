from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, join, dirname, splitext, basename
from sklearn.model_selection import StratifiedKFold
from experiences.processes import Processes
from toolbox.core.models import SimpleModels
from toolbox.core.structures import Inputs
from toolbox.IO import otorhinolaryngology

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = 'OrlTest'
    validation = StratifiedKFold(n_splits=2, shuffle=True)

    # Output dir
    output_folder = normpath('{temp}/spectroscopy/{filename}'.format(temp=temp_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    filters_by = {'Results_SvsC': {'label': ['Sain', 'Cancer']}}
    input_folder = normpath('{here}/data/spectroscopy'.format(here=here_path))
    inputs = Inputs(folders=[join(input_folder, 'Patients.csv')], loader=otorhinolaryngology.Reader.read_table,
                    tags={'data_tag': 'Data', 'label_tag': 'label',
                          'group_tag': 'patient_name', 'references_tags': ['patient_name', 'spectrum_id']})
    inputs.load()

    # Get experiences
    model, params = SimpleModels.get_dummy_process()

    for item_name, item_filter in filters_by.items():
        Processes.otorhinolaryngology(inputs=inputs, output_folder=output_folder, model=model,
                                      params=params, name=item_name)

    # Open result folder
    startfile(output_folder)

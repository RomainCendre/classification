from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, join, dirname, splitext, basename
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.core.models import Classifiers
from toolbox.core.structures import Inputs
from toolbox.IO import otorhinolaryngology

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = filename
    validation = StratifiedKFold(n_splits=2, shuffle=True)

    # Output dir
    output_folder = normpath('{temp}/spectroscopy/{filename}'.format(temp=temp_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    filters_by = {'Results_SvsC': {'label': ['Sain', 'Cancer']}}
    input_folder = normpath('{here}/data/spectroscopy'.format(here=here_path))
    inputs = Inputs(folders=[join(input_folder, 'Patients.csv')], instance=otorhinolaryngology.Reader(), loader=otorhinolaryngology.Reader.read_table,
                    tags={'data': 'Data', 'label': 'label', 'group': 'patient_name', 'references': 'Reference'})
    inputs.load()

    # Get experiments
    for item_name, item_filter in filters_by.items():
        process = Process()
        process.begin(validation, validation)
        process.end(inputs=inputs, model=Classifiers.get_dummy_simple(), output_folder=output_folder, name=item_name)

    # Open result folder
    startfile(output_folder)

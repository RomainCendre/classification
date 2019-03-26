from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.models import Classifiers
from toolbox.core.transforms import OrderedEncoder

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = filename
    validation = StratifiedKFold(n_splits=2, shuffle=True)
    settings = DefinedSettings.get_default_orl()

    # Output dir
    output_folder = normpath('{temp}/spectroscopy/{filename}'.format(temp=temp_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    inputs = Dataset.test_spectras()
    filters = {'Results_SvsC': {'label': ['Sain', 'Cancer']}}

    # Get experiments
    for item_name, item_filters in filters.items():
        # Change filters
        inputs.filters = item_filters
        inputs.encoders = {'label': OrderedEncoder().fit(item_filters['label']),
                           'groups': LabelEncoder()}
        process = Process()
        process.begin(inner_cv=validation, outer_cv=validation, settings=settings)
        process.end(inputs=inputs, model=Classifiers.get_dummy_simple(), output_folder=output_folder, name=item_name)

    # Open result folder
    startfile(output_folder)

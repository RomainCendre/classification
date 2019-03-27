from os import makedirs, startfile

from numpy.ma import arange
from sklearn.model_selection import StratifiedKFold
from os.path import exists, expanduser, normpath, basename, splitext

from sklearn.preprocessing import LabelEncoder

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.models import BuiltInModels
from toolbox.core.transforms import OrderedEncoder

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser("~")
    name = filename
    validation = StratifiedKFold(n_splits=5)
    settings = DefinedSettings.get_default_orl()

    # Output dir
    output_folder = normpath('{home}/Results/ORL/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Filters
    filters = [('All', {'label': ['Sain', 'Precancer', 'Cancer']}),
               ('SvsC', {'label': ['Sain', 'Cancer']}),
               ('SvsP', {'label': ['Sain', 'Precancer']}),
               ('PvsC', {'label': ['Precancer', 'Cancer']})]

    # Input data
    spectra = Dataset.spectras()
    spectra.apply_average_filter(size=5)
    spectra.apply_scaling()
    spectra.change_wavelength(wavelength=arange(start=445, stop=962, step=1))

    for item_name, item_filters in filters:
        # Change filters
        spectra.set_filters(item_filters)
        spectra.set_encoders({'label': OrderedEncoder().fit(item_filters['label']),
                              'groups': LabelEncoder()})
        process = Process()
        process.begin(inner_cv=validation, outer_cv=validation, settings=settings)
        process.end(inputs=spectra, model=BuiltInModels.get_pls_process(), output_folder=output_folder, name=item_name)

    # Open result folder
    startfile(output_folder)

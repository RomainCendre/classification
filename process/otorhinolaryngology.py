from numpy import arange
from os import makedirs
from os.path import expanduser, normpath, join, exists
from sklearn.model_selection import GroupKFold

from core.classification import Classifier
from core.models import Models
from IO.otorhinolaryngology import Reader
from IO.writer import ResultWriter, StatisticsWriter


def compute_data(data_set, out_dir, name, filter_by, meta):
    # Get process
    pipe, param = Models.get_pls_process()

    # Write statistics on current data
    StatisticsWriter(data_set).write_result(metas=meta, dir_name=out_dir, name=name, filter_by=filter_by)

    # Classify and write data results
    classifier = Classifier(pipeline=pipe, params=param,
                            inner_cv=GroupKFold(n_splits=5), outer_cv=GroupKFold(n_splits=5))
    result = classifier.evaluate(features=spectra.get_data(filter_by=filter_by),
                                 labels=spectra.get_meta(meta='label', filter_by=filter_by),
                                 groups=spectra.get_meta(meta='patient_name', filter_by=filter_by))
    ResultWriter(result).write_results(dir_name=out_dir, name=name)


if __name__ == "__main__":

    home_path = expanduser("~")

    # Output dir
    output_dir = normpath('{home}/Results/Neck/'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Load data
    data_dir = normpath('{home}/Data/Neck/'.format(home=home_path))
    spectra_patients = Reader(';').read_table(join(data_dir, 'Patients.csv'))
    spectra_temoins = Reader(';').read_table(join(data_dir, 'Temoins.csv'))
    spectra = spectra_patients + spectra_temoins

    # Filter data
    spectra.apply_method(name='apply_average_filter', parameters={'size': 5})
    spectra.apply_method(name='apply_scaling')
    spectra.apply_method(name='change_wavelength', parameters={'wavelength': arange(start=445, stop=962, step=1)})

    filters = {
        'Results_All': {},
        'Results_SvsC': {'label': ['Sain', 'Cancer']},
        'Results_SvsP': {'label': ['Sain', 'Precancer']},
        'Results_PvsC': {'label': ['Precancer', 'Cancer']},
    }
    metas = ['patient_label', 'device', 'label', 'location']

    for item_name, item_filter in filters.items():
        compute_data(data_set=spectra, out_dir=output_dir, name=item_name, filter_by=item_filter, meta=metas)

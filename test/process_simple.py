from tempfile import gettempdir
from os import makedirs
from os.path import expanduser, normpath, exists, join

from sklearn.model_selection import GroupKFold

from toolbox.IO.otorhinolaryngology import Reader
from toolbox.IO.writer import StatisticsWriter, ResultWriter
from toolbox.core.classification import Classifier
from toolbox.core.models import SimpleModels
from toolbox.core.structures import Inputs

if __name__ == "__main__":

    home_path = expanduser("~")
    temp_path = gettempdir()

    # Output dir
    output_dir = normpath('{temp}/Results/Neck/'.format(temp=temp_path))
    if not exists(output_dir):
        makedirs(output_dir)
    print('Output directory: {out}'.format(out=output_dir))

    # Load data
    data_dir = normpath('{home}/Data/Neck/'.format(home=home_path))
    spectra = Reader(';').read_table(join(data_dir, 'Patients.csv'))

    # Parameters
    name = 'Test'

    keys = ['patient_label', 'device', 'label', 'location']

    inputs = Inputs(spectra, data_tag='Data', label_tag='label', group_tag='patient_name',
                    references_tags=['patient_name', 'spectrum_id'])

    # Get process
    pipe, param = SimpleModels.get_dummy_process()

    # Write statistics on current data
    StatisticsWriter(spectra).write_result(keys=keys, dir_name=output_dir, name=name)

    # Classify and write data results
    classifier = Classifier(pipeline=pipe, params=param,
                            inner_cv=GroupKFold(n_splits=5), outer_cv=GroupKFold(n_splits=5))
    result = classifier.evaluate(inputs)
    ResultWriter(inputs, result).write_results(dir_name=output_dir, name=name)

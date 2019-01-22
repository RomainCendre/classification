from tempfile import gettempdir
from os import makedirs
from os.path import normpath, exists, dirname
from sklearn.model_selection import StratifiedKFold
from toolbox.IO.dermatology import Reader
from toolbox.IO.writer import StatisticsWriter, ResultWriter
from toolbox.core.classification import ClassifierDeep
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":

    here_path = dirname(__file__)
    temp_path = gettempdir()

    # Output dir
    output_dir = normpath('{temp}/dermatology/'.format(temp=temp_path))
    if not exists(output_dir):
        makedirs(output_dir)
    print('Output directory: {out}'.format(out=output_dir))

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Step 1 -- Load pretrain data
    thumbnail_folder = normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))
    data_set = Reader().scan_folder_for_images(thumbnail_folder)
    inputs = Inputs(data_set, data_tag='Data', label_tag='Label')

    # Get classification model for confocal
    model, preprocess, extractor = DeepModels.get_dummy_model(inputs)
    classifier = ClassifierDeep(model=model, outer_cv=StratifiedKFold(n_splits=2),
                                preprocess=preprocess, work_dir=output_dir)
    classifier.model = classifier.fit(inputs, epochs=10)

    # Step 2 -- Load data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}
    name = 'Test'
    keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
    patient_folder = normpath('{here}/data/dermatology/Patients'.format(here=here_path))
    data_set = Reader().scan_folder(patient_folder)
    inputs.change_data(data_set, filter_by=filter_by)
    inputs.change_group(group_tag='Patient')

    StatisticsWriter(data_set).write_result(keys=keys, dir_name=output_dir, filter_by=filter_by, name=name)
    result = classifier.evaluate(inputs, epochs=10)
    ResultWriter(inputs, result).write_results(dir_name=output_dir, name=name)


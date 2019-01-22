from tempfile import gettempdir
from os import makedirs
from os.path import expanduser, normpath, exists, join, dirname

import keras
from sklearn.model_selection import GroupKFold, StratifiedKFold

from toolbox.IO.dermatology import Reader
from toolbox.IO.writer import StatisticsWriter, ResultWriter, VisualizationWriter
from toolbox.core.classification import Classifier, ClassifierDeep
from toolbox.core.models import SimpleModels, DeepModels
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

    # Load data
    patient_folder = normpath('{here}/data/dermatology/Patients'.format(here=here_path))
    data_set = Reader().scan_folder(patient_folder)

    # Load data references
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}

    inputs = Inputs(data_set, data_tag='Data', label_tag='Label', group_tag='Patient', filter_by=filter_by)

    # Parameters
    name = 'Test'

    keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']

    StatisticsWriter(data_set).write_result(keys=keys, dir_name=output_dir, filter_by=filter_by, name=name)

    # Get classification model for confocal
    model, preprocess, extractor = DeepModels.get_dummy_model(inputs)

    classifier = ClassifierDeep(model=model, outer_cv=StratifiedKFold(n_splits=2, shuffle=True),
                                preprocess=preprocess, work_dir=output_dir)
    result = classifier.evaluate(inputs, epochs=10)
    ResultWriter(result).write_results(dir_name=output_dir, name=name)


    # Fit model and evaluate visualization
    model = classifier.fit(inputs)
    VisualizationWriter(model=model).write_activations_maps(dir=activation_dir)

    keras.Model()

from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename, join
from joblib import Memory
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from experiments.processes import Process
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.core.transforms import DWTDescriptorTransform

if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = filename
    validation = StratifiedKFold(n_splits=2, shuffle=True)

    # Output dir
    output_folder = normpath('{temp}/dermatology/{filename}'.format(temp=temp_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Feature folder
    features_folder = join(output_folder, 'Features')
    if not exists(features_folder):
        makedirs(features_folder)

    # Input data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                     normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
    inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                    tags={'data': 'Full_path', 'label': 'Label', 'reference': 'Reference'}, filter_by=filter_by)
    inputs.load()

    # Cache steps
    memory = Memory(cachedir=features_folder, verbose=10)

    pipe = Pipeline([('dwt', DWTDescriptorTransform(mode='db1')), ('clf', DummyClassifier())],
                    memory=memory)
    pipe.name = 'DWT'
    # Define parameters to validate through grid CV
    parameters = {'dwt__mode': ['db1', 'db2']}

    # Initiate model and params
    process = Process()
    process.begin(validation, validation)
    process.end(inputs=inputs, model=(pipe, parameters), output_folder=output_folder, name=name)



    # Open result folder
    startfile(output_folder)

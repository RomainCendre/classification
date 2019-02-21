from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname, splitext, basename, join
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Process
from toolbox.core.models import DeepModels, SimpleModels
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

    # Temporary folder
    temp_folder = join(output_folder, 'Features')
    if not exists(temp_folder):
        makedirs(temp_folder)

    # Input data
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['Malignant', 'Benign', 'Normal']}
    input_folders = [normpath('{here}/data/dermatology/DB_Test1/Patients'.format(here=here_path)),
                     normpath('{here}/data/dermatology/DB_Test2/Patients'.format(here=here_path))]
    inputs = Inputs(folders=input_folders, instance=dermatology.Reader(), loader=dermatology.Reader.scan_folder,
                    tags={'data_tag': 'Full_path', 'label_tag': 'Label', 'reference_tag': 'Reference'}, filter_by=filter_by)
    inputs.load()

    # Initiate model and params
    model, params = SimpleModels.get_dummy_process()

    process = Process()
    process.begin(validation, validation, DeepModels.get_callbacks(output_folder))
    process.checkpoint_step(inputs=inputs, model=DWTDescriptorTransform(), folder=temp_folder)
    process.end(inputs=inputs, model=model, params=params, output_folder=output_folder, name=name)

    # Open result folder
    startfile(output_folder)

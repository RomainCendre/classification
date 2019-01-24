from tempfile import gettempdir
from os import makedirs, startfile
from os.path import normpath, exists, dirname
from experiences.processes import Processes
from toolbox.core.models import DeepModels
from toolbox.tools.limitations import Parameters

if __name__ == "__main__":

    here_path = dirname(__file__)
    temp_path = gettempdir()
    name = 'Test'
    nb_class = 2
    epochs = 1

    # Output dir
    output_folder = normpath('{temp}/dermatology/'.format(temp=temp_path))
    if not exists(output_folder):
        makedirs(output_folder)

    # Pretrain data
    pretrain_folder = normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))

    # Input data
    input_folder = normpath('{here}/data/dermatology/Patients'.format(here=here_path))

    # Filters
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Get right model and process function
    model, preprocess = DeepModels.get_dummy_model(nb_class)
    learner = {'Model': model,
               'Preprocess': preprocess}

    Processes.dermatology_pretrain(pretrain_folder, input_folder, output_folder, name, filter_by, learner, epochs)

    # Open result folder
    startfile(output_folder)


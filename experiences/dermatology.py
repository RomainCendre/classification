from os import makedirs, startfile
from time import gmtime, strftime, time
from os.path import exists, expanduser, normpath
from experiences.processes import Processes
from toolbox.core.models import DeepModels
from toolbox.tools.limitations import Parameters
from toolbox.tools.tensorboard import TensorBoardTool

if __name__ == '__main__':

    home_path = expanduser("~")
    name = 'DeepLearning'
    nb_class = 2
    epochs = 100

    # Output dir
    output_folder = normpath('{home}/Results/Skin/Saint_Etienne/Deep/'.format(home=home_path))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    input_folder = normpath('{home}/Data/Skin/Saint_Etienne/Patients'.format(home=home_path))

    # Filters
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}

    # Adding experiences to watch our training experiences
    current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
    work_dir = normpath('{output_dir}/Graph/{time}'.format(output_dir=output_folder, time=current_time))
    makedirs(work_dir)

    # Tensorboard tool launch
    tb_tool = TensorBoardTool(work_dir)
    tb_tool.write_batch()
    tb_tool.run()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Get classification model for confocal
    model, preprocess = DeepModels.get_confocal_model(nb_class)
    learner = {'Model': model,
               'Preprocess': preprocess}

    Processes.dermatology(input_folder, output_folder, name, filter_by, learner, epochs)

    # Open result folder
    startfile(output_folder)

from os import makedirs, startfile
from time import gmtime, strftime, time
from os.path import exists, expanduser, normpath, join
from sklearn.model_selection import GroupKFold, StratifiedKFold

from process.processes import Processes
from toolbox.IO.dermatology import Reader, DataManager
from toolbox.IO.writers import ResultWriter, StatisticsWriter, VisualizationWriter
from toolbox.core.classification import ClassifierDeep
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.tools.limitations import Parameters
from toolbox.tools.tensorboard import TensorBoardTool

outer_cv = GroupKFold(n_splits=5)

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
    if not exists(input_folder):
        origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/Original'.format(home=home_path))
        DataManager(origin_folder).launch_converter(input_folder)

    # Filters
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}

    # Step 2 - Fit and Evaluate
    # Adding process to watch our training process
    current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
    work_dir = normpath('{output_dir}/Graph/{time}'.format(output_dir=output_folder, time=current_time))
    makedirs(work_dir)

    # Tensorboard tool launch
    tb_tool = TensorBoardTool(work_dir)
    tb_tool.write_batch()
    tb_tool.run()

    # Get classification model for confocal
    model, preprocess = DeepModels.get_confocal_model(nb_class)
    learner = {'Model': model,
               'Preprocess': preprocess}


    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)
    Processes.dermatology(input_folder, output_folder, name, filter_by, learner, epochs)

    # Open result folder
    startfile(output_folder)

from os import makedirs
from time import gmtime, strftime, time
from os.path import exists, expanduser, normpath, join
from sklearn.model_selection import GroupKFold, StratifiedKFold

from toolbox.IO.dermatology import Reader, DataManager
from toolbox.IO.writer import ResultWriter, StatisticsWriter, VisualizationWriter
from toolbox.core.classification import ClassifierDeep
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.tools.limitations import Parameters
from toolbox.tools.tensorboard import TensorBoardTool

outer_cv = GroupKFold(n_splits=5)

if __name__ == '__main__':

    home_path = expanduser("~")

    # Experience name
    name = 'DeepLearning'

    # Output dir
    output_dir = normpath('{home}/Results/Skin/Saint_Etienne/Deep/'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Activation dir
    activation_dir = join(output_dir, 'Activation/')
    if not exists(activation_dir):
        makedirs(activation_dir)

    # Prepare data
    patient_folder = normpath('{home}/Data/Skin/Saint_Etienne/Patients'.format(home=home_path))
    if not exists(patient_folder):
        origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/Original'.format(home=home_path))
        DataManager(origin_folder).launch_converter(patient_folder)

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Step 1 - Load data and statistics
    keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}
    data_set = Reader().scan_folder(patient_folder)
    inputs = Inputs(data_set, data_tag='Data', label_tag='Label', group_tag='Patient',
                    references_tags=['Data'], filter_by=filter_by)
    StatisticsWriter(data_set).write_result(keys=keys, dir_name=output_dir,
                                            name=name, filter_by=filter_by)

    # Step 2 - Fit and Evaluate
    # Adding process to watch our training process
    current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
    work_dir = normpath('{output_dir}/Graph/{time}'.format(output_dir=output_dir, time=current_time))
    makedirs(work_dir)

    # Tensorboard tool launch
    tb_tool = TensorBoardTool(work_dir)
    tb_tool.write_batch()
    tb_tool.run()

    # Get classification model for confocal
    model, preprocess, extractor = DeepModels.get_confocal_model(inputs)
    classifier = ClassifierDeep(model=model, outer_cv=StratifiedKFold(n_splits=5, shuffle=True),
                                preprocess=preprocess, work_dir=work_dir)
    result = classifier.evaluate(inputs)
    ResultWriter(result).write_results(dir_name=output_dir, name=name)

    # Fit model and evaluate visualization
    model = classifier.fit(inputs)
    VisualizationWriter(model=model).write_activations_maps(dir=activation_dir)

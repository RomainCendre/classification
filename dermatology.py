from os import makedirs
from time import gmtime, strftime, time
from os.path import exists, expanduser, normpath, join
from sklearn.model_selection import GroupKFold, StratifiedKFold

from IO.dermatology import Reader, DataManager
from IO.writer import ResultWriter, StatisticsWriter
from core.classification import ClassifierDeep
from core.models import DeepModels
from tools.limitations import Parameters
from tools.tensorboard import TensorBoardTool

outer_cv = GroupKFold(n_splits=5)

if __name__ == '__main__':

    home_path = expanduser("~")

    # Output dir
    output_dir = normpath('{home}/Results/Skin/Saint_Etienne/Deep/'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Activation dir
    activation_dir = join(output_dir, 'Activation/')
    if not exists(activation_dir):
        makedirs(activation_dir)

    # Prepare data
    origin_folder = normpath('{home}/Data/Skin/Saint_Etienne/Original'.format(home=home_path))
    patient_folder = normpath('{home}/Data/Skin/Saint_Etienne/Patients'.format(home=home_path))
    DataManager(origin_folder).launch_converter(patient_folder)

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Load data references
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}
    dataset = Reader().scan_folder(patient_folder)
    paths = dataset.get_data(filter_by=filter_by)
    labels = dataset.get_meta(meta='Label', filter_by=filter_by)
    groups = dataset.get_meta(meta='Patient', filter_by=filter_by)
    # Adding process to watch our training process
    current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
    work_dir = normpath('{output_dir}/Graph/{time}'.format(output_dir=output_dir, time=current_time))
    makedirs(work_dir)

    # Tensorboard tool launch
    tb_tool = TensorBoardTool(work_dir)
    tb_tool.write_batch()
    tb_tool.run()

    # Save statistics
    metas = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
    StatisticsWriter(dataset).write_result(metas=metas, dir_name=output_dir,
                                           name='DeepLearning', filter_by=filter_by)

    # Get classification model for confocal
    model, preprocess, extractor = DeepModels.get_confocal_model(labels=labels)

    classifier = ClassifierDeep(model=model, outer_cv=StratifiedKFold(n_splits=5, shuffle=True),
                                preprocess=preprocess, activation_dir=activation_dir,
                                work_dir=work_dir)
    result = classifier.evaluate(paths=paths, labels=labels, groups=groups)
    ResultWriter(result).write_results(dir_name=output_dir, name='DeepLearning')

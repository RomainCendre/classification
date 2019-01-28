from glob import glob
from os import makedirs
from os.path import expanduser, normpath, exists, join

from time import gmtime, strftime, time

from numpy import geomspace, concatenate, full
from sklearn.model_selection import StratifiedKFold

from toolbox.IO.dermatology import Reader
from toolbox.IO.writers import ResultWriter
from toolbox.core.classification import ClassifierDeep
from toolbox.core.models import DeepModels
from toolbox.tools.limitations import Parameters
from toolbox.tools.tensorboard import TensorBoardTool

if __name__ == "__main__":

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    home_path = expanduser("~")

    # Output dir
    output_dir = normpath('{home}/Results/Skin/Thumbnails/Deep/'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Adding experiences to watch our training experiences
    current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
    work_dir = normpath('{output_dir}/Graph/{time}'.format(output_dir=output_dir, time=current_time))
    makedirs(work_dir)

    # Load data
    benign_dir = normpath('{home}/Data/Skin/Thumbnails/Benin'.format(home=home_path))
    benign_files = glob(join(benign_dir, '*.bmp'))
    benign_label = full(len(benign_files), 'Normal')
    malignant_dir = normpath('{home}/Data/Skin/Thumbnails/Malin'.format(home=home_path))
    malignant_files = glob(join(malignant_dir, '*.bmp'))
    malignant_label = full(len(malignant_files), 'LM')
    paths = concatenate((benign_files, malignant_files), axis=0)
    labels = concatenate((benign_label, malignant_label), axis=0)


    # Pre trained model on patch
    model, preprocess, extractor = DeepModels.get_confocal_model(labels=labels)
    classifier = ClassifierDeep(model=model, outer_cv=StratifiedKFold(n_splits=5),
                                preprocess=preprocess, activation_dir=activation_dir, work_dir=work_dir)
    classifier.model = classifier.fit(paths, labels)

    # Load data on which we want to perform
    patient_folder = normpath('{home}/Data/Skin/Saint_Etienne/Patients'.format(home=home_path))
    filter_by = {'Modality': 'Microscopy',
                 'Label': ['LM', 'Normal']}
    dataset = Reader().scan_folder(patient_folder)
    paths = dataset.get_data(filter_by=filter_by)
    labels = dataset.get_meta(meta='Label', filter_by=filter_by)

    result = classifier.evaluate_patch(paths, labels)
    ResultWriter(result).write_results(dir_name=output_dir, name='DeepLearning')


    # Activation dir
    activation_dir = join(output_dir, 'Activation/')
    if not exists(activation_dir):
        makedirs(activation_dir)

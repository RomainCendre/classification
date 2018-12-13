from glob import glob
from os import makedirs
from os.path import expanduser, normpath, exists, join

from time import gmtime, strftime, time

from numpy import geomspace, concatenate, full
from sklearn.model_selection import StratifiedKFold

from IO.writer import ResultWriter
from core.classification import ClassifierDeep
from core.models import DeepModels
from tools.limitations import Parameters
from tools.tensorboard import TensorBoardTool

if __name__ == "__main__":

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    home_path = expanduser("~")

    # Output dir
    output_dir = normpath('{home}/Results/Skin/Thumbnails/Deep/'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Activation dir
    activation_dir = join(output_dir, 'Activation/')
    if not exists(activation_dir):
        makedirs(activation_dir)

    # Adding process to watch our training process
    current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
    work_dir = normpath('{output_dir}/Graph/{time}'.format(output_dir=output_dir, time=current_time))
    makedirs(work_dir)

    # Load data
    benign_dir = normpath('{home}/Data/Skin/Thumbnails/Benin'.format(home=home_path))
    benign_files = glob(join(benign_dir, '*.bmp'))
    benign_label = full(len(benign_files), 'benin')
    malignant_dir = normpath('{home}/Data/Skin/Thumbnails/Malin'.format(home=home_path))
    malignant_files = glob(join(malignant_dir, '*.bmp'))
    malignant_label = full(len(malignant_files), 'malin')
    paths = concatenate((benign_files, malignant_files), axis=0)
    labels = concatenate((benign_label, malignant_label), axis=0)

    # Tensorboard tool launch
    tb_tool = TensorBoardTool(work_dir)
    tb_tool.write_batch()
    tb_tool.run()

    # Get model for confocal microscopy
    model, preprocess, extractor = DeepModels.get_confocal_model()

    classifier = ClassifierDeep(model=model, outer_cv=StratifiedKFold(n_splits=5), preprocess=preprocess,
                                activation_dir=activation_dir, work_dir=work_dir)
    result = classifier.evaluate(paths=paths, labels=labels)
    ResultWriter(result).write_results(dir_name=output_dir, name='DeepLearning')
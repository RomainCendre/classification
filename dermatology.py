from os import makedirs
from time import gmtime, strftime, time
from os.path import expanduser, normpath
from sklearn.model_selection import GroupKFold

from IO.dermatology import Reader
from core.classification import ClassifierDeep
from tools.limitations import Parameters

outer_cv = GroupKFold(n_splits=5)

if __name__ == '__main__':

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Load data references
    home_path = expanduser("~")
    dataset = Reader(';').scan_folder(normpath('{home}/Data/Skin/Patients'.format(home=home_path)))
    datas = dataset.get(label='Malignant', filter={'modality': 'Microscopy'})

    # Adding process to watch our training process
    current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
    work_dir = normpath('{output_dir}/Graph/{time}'.format(output_dir=output_dir, time=current_time))
    makedirs(work_dir)

    # Tensorboard tool launch
    # tb_tool = TensorBoardTool(work_dir)
    # tb_tool.write_batch()
    # tb_tool.run()

    results = []
    classifier = ClassifierDeep(outer_cv=outer_cv, work_dir=work_dir)
    classifier.test_vis()
    classifier.save_features(paths=paths, labels=labels, output_path=work_dir)
    classifier.evaluate_top(output_path=work_dir)
    results.append(classifier.evaluate(paths=paths, labels=labels, groups=patients))

    # SpectrumResultsWriter(results).write_results('1', home_path, 'Results_Deep')

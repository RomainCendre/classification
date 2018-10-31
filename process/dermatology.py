from os import makedirs
from time import time
from os.path import expanduser
from sklearn.model_selection import GroupKFold

from IO.dermatology import Reader
from tools.limitations import Parameters

outer_cv = GroupKFold(n_splits=5)

if __name__ == '__main__':

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Load data references
    home_path = expanduser("~")
    dataset = Reader(';').scan_folder('{home}\\Data\\Skin\\Patients'.format(home=home_path))
    datas = dataset.get(label='Malignant', filter={'modality': 'Microscopy'})
    # labels[labels == ''] = '0'
    # # Adding process to watch our training process
    # work_dir = '{home}\\Graph\\{time}'.format(home=home_path, time=time())
    # makedirs(work_dir)

    # Tensorboard tool launch
    # tb_tool = TensorBoardTool(work_dir)
    # tb_tool.write_batch()
    # tb_tool.run()

    # results = []
    # classifier = SkinClassifier(outer_cv=outer_cv, work_dir=work_dir)
    # classifier.test_vis()
    # classifier.save_features(paths=paths, labels=labels, output_path=work_dir)
    # classifier.evaluate_top(output_path=work_dir)
    # results.append(classifier.evaluate(paths=paths, labels=labels, groups=patients))

    # SpectrumResultsWriter(results).write_results('1', home_path, 'Results_Deep')

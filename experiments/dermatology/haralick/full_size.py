from copy import deepcopy
from os import makedirs, startfile
from os.path import normpath, exists, expanduser, splitext, basename
from sklearn.model_selection import StratifiedKFold
from experiments.processes import Processes
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.IO import dermatology
from toolbox.tools.limitations import Parameters

def extract_haralick(input_dir, label):
    files = glob(join(input_dir, '*.bmp'))
    features = []
    for file in files:
        image = array(Image.open(file))
        features.append(mahotas.features.haralick(image[:, :, 0]).mean(axis=0))
    features = asarray(features)
    return features, full(features.shape[0], label)


if __name__ == "__main__":

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser("~")
    name = filename
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Dermatology/Haralick/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Load data
    benign_dir = normpath('{home}/Data/Skin/Thumbnails/Benin'.format(home=home_path))
    malignant_dir = normpath('{home}/Data/Skin/Thumbnails/Malin'.format(home=home_path))
    features, labels = extract_haralick(benign_dir, 'benin')
    features_m, labels_m = extract_haralick(malignant_dir, 'malin')

    # Format Data
    features = concatenate((features, features_m), axis=0)
    features = StandardScaler().fit_transform(features)
    labels = concatenate((labels, labels_m), axis=0)

    # Write data to visualize it
    DataProjector.project_data(datas=features, labels=labels, path=join(output_dir, 'Projector'))

    # Define parameters to validate through grid CV
    pipe = Pipeline([('clf', SVC(kernel='linear', probability=True))])
    parameters = {'clf__C': geomspace(0.01, 1000, 6)}

    # Classify and write data results
    classifier = Classifier(pipeline=pipe, params=parameters,
                            inner_cv=StratifiedKFold(n_splits=5), outer_cv=StratifiedKFold(n_splits=5))
    result = classifier.evaluate(features=features, labels=labels)
    ResultWriter(result).write_results(dir_name=output_dir, name='Results')

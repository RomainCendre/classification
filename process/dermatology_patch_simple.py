from glob import glob
from os import makedirs
from os.path import expanduser, normpath, exists, join

from numpy import array, full, asarray, concatenate, geomspace
from sklearn.model_selection import StratifiedKFold

import mahotas
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from IO.writer import ResultWriter
from core.classification import Classifier


def read_haralick(input_dir, label):
    files = glob(join(input_dir, '*.bmp'))
    features = []
    for file in files:
        image = array(Image.open(file))
        features.append(mahotas.features.haralick(image).flatten())
    features = asarray(features)
    return features, full(features.shape[0], label)


if __name__ == "__main__":

    home_path = expanduser("~")

    # Output dir
    output_dir = normpath('{home}/Results/Skin/Thumbnails'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Load data
    benign_dir = normpath('{home}/Data/Skin/Thumbnails/Benin'.format(home=home_path))
    malignant_dir = normpath('{home}/Data/Skin/Thumbnails/Malin'.format(home=home_path))
    features, labels = read_haralick(benign_dir, 'benin')
    features_m, labels_m = read_haralick(malignant_dir, 'malin')
    features = concatenate((features, features_m), axis=0)
    labels = concatenate((labels, labels_m), axis=0)

    # Classify data
    pipe = Pipeline([
    ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True))
    ])
    # Define parameters to validate through grid CV
    parameters = {
        'clf__C': geomspace(0.01, 1000, 6),
        'clf__gamma': geomspace(0.01, 1000, 6)
    }

    # Classify and write data results
    classifier = Classifier(pipeline=pipe, params=parameters,
                            inner_cv=StratifiedKFold(n_splits=5), outer_cv=StratifiedKFold(n_splits=5))
    result = classifier.evaluate(features=features, labels=labels)
    ResultWriter(result).write_results(dir_name=output_dir, name='Test')
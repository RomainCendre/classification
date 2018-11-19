from glob import glob
from os import makedirs
from os.path import expanduser, normpath, exists, join, isfile

from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from numpy import array, expand_dims, save, load
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from numpy import geomspace, concatenate, full, asarray
from sklearn.model_selection import StratifiedKFold

from IO.writer import ResultWriter
from core.classification import Classifier
from tools.limitations import Parameters
from tools.tensorboard import DataProjector


def extract_deepfeatures (input_dir, label):
    # Init model
    model = InceptionV3(weights='imagenet', include_top=False, pooling='max')

    files = glob(join(input_dir, '*.bmp'))
    features = []
    for file in files:
        image = expand_dims(array(Image.open(file)), axis=0)
        image = preprocess_input(image)
        features.append(model.predict(image)[0])
    features = asarray(features)
    return features, full(features.shape[0], label)


if __name__ == "__main__":
    # TODO DummyClassifier

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    home_path = expanduser("~")

    # Output dir
    output_dir = normpath('{home}/Results/Skin/Thumbnails/Deep_Features/'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Prepare input ressources
    features_file = join(output_dir, 'features.npy')
    labels_file = join(output_dir, 'labels.npy')

    if not isfile(features_file):
        # Load data
        input_dir = normpath('{home}/Data/Skin/Thumbnails/'.format(home=home_path))
        benign_dir = join(input_dir, 'Benin')
        malignant_dir = join(input_dir, 'Malin')
        features, labels = extract_deepfeatures(benign_dir, 'benin')
        features_m, labels_m = extract_deepfeatures(malignant_dir, 'malin')

        features = concatenate((features, features_m), axis=0)
        features = StandardScaler().fit_transform(features)
        labels = concatenate((labels, labels_m), axis=0)

        # Make a save of data
        save(features_file, features)
        save(labels_file, labels)
    else:
        # Reload data if exist
        features = load(features_file)
        labels = load(labels_file)

    # Save data as projector to visualize them
    DataProjector.project_data(datas=features, labels=labels, path=join(output_dir, 'Projector'))

    # Avoid this instructions in a first step to avoid kind of overfit
    # pipe = Pipeline([('clf', SVC(kernel='linear', probability=True))])
    # parameters = {'clf__C': geomspace(0.01, 1000, 6)}

    pipe = Pipeline([('pca', PCA()),
                     ('clf', SVC(kernel='linear', probability=True))])

    # Define parameters to validate through grid CV
    parameters = {'pca__n_components': [0.95],
                  'clf__C': geomspace(0.01, 1000, 6)}

    # Classify and write data results
    classifier = Classifier(pipeline=pipe, params=parameters,
                            inner_cv=StratifiedKFold(n_splits=5), outer_cv=StratifiedKFold(n_splits=5))
    result = classifier.evaluate(features=features, labels=labels)
    ResultWriter(result).write_results(dir_name=output_dir, name='Results')

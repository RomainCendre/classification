from glob import glob
from os import makedirs
from os.path import expanduser, normpath, exists, join

from PIL import Image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from numpy import array, expand_dims
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from time import gmtime, strftime, time

from numpy import geomspace, concatenate, full, asarray
from sklearn.model_selection import StratifiedKFold, train_test_split

from IO.writer import ResultWriter
from core.classification import Classifier
from tools.limitations import Parameters


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

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    home_path = expanduser("~")

    # Output dir
    output_dir = normpath('{home}/Results/Skin/Thumbnails/Deep/'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Adding process to watch our training process
    current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
    work_dir = normpath('{output_dir}/Graph/{time}'.format(output_dir=output_dir, time=current_time))
    makedirs(work_dir)

    # Load data
    benign_dir = normpath('{home}/Data/Skin/Thumbnails/Benin'.format(home=home_path))
    malignant_dir = normpath('{home}/Data/Skin/Thumbnails/Malin'.format(home=home_path))
    features, labels = extract_deepfeatures(benign_dir, 'benin')
    features_m, labels_m = extract_deepfeatures(malignant_dir, 'malin')

    features = concatenate((features, features_m), axis=0)
    labels = concatenate((labels, labels_m), axis=0)

    features = StandardScaler().fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.40)

    pipe = Pipeline([('clf', SVC(kernel='linear', probability=True))])
    parameters = {'clf__C': geomspace(0.01, 1000, 6)}

    # Classify and write data results
    classifier = Classifier(pipeline=pipe, params=parameters,
                            inner_cv=StratifiedKFold(n_splits=5), outer_cv=StratifiedKFold(n_splits=5))
    result = classifier.evaluate(features=features, labels=labels)
    ResultWriter(result).write_results(dir_name=output_dir, name='DeepFeatures')

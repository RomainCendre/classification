from os import makedirs, startfile
from sklearn.model_selection import StratifiedKFold
from os.path import exists, expanduser, normpath, splitext, basename
from experiments.processes import Processes
from toolbox.IO import dermatology
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs
from toolbox.tools.limitations import Parameters

if __name__ == '__main__':

    # Parameters
    filename = splitext(basename(__file__))[0]
    home_path = expanduser('~')
    name = 'Results'
    validation = StratifiedKFold(n_splits=5, shuffle=True)

    # Output dir
    output_folder = normpath('{home}/Results/Dermatology/Transfer_learning/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    input_folder = normpath('{home}/Data/Skin/Thumbnails'.format(home=home_path))
    inputs = Inputs(folders=[input_folder], loader=dermatology.Reader.scan_folder_for_images,
                    tags={'data_tag': 'Data', 'label_tag': 'Label'})
    inputs.load()

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    # Initiate model and params
    model = KerasBatchClassifier(DeepModels.get_confocal_model)
    params = {'epochs': [100],
              'batch_size': [10],
              'preprocessing_function': [DeepModels.get_confocal_preprocessing()],
              'inner_cv': validation,
              'outer_cv': validation}

    # Launch process
    Processes.dermatology(inputs, output_folder, model, params, name)

    # Open result folder
    startfile(output_folder)




#
# def extract_deepfeatures (input_dir, label):
#     # Init model
#     model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
#
#     files = glob(join(input_dir, '*.bmp'))
#     features = []
#     for file in files:
#         image = expand_dims(array(Image.open(file)), axis=0)
#         image = preprocess_input(image)
#         features.append(model.predict(image)[0])
#     features = asarray(features)
#     return features, full(features.shape[0], label)
#
#
# if __name__ == "__main__":
#     # TODO DummyClassifier
#
#     # Configure GPU consumption
#     Parameters.set_gpu(percent_gpu=0.5)
#
#     home_path = expanduser("~")
#
#     # Output dir
#     output_dir = normpath('{home}/Results/Skin/Thumbnails/Deep_Features/'.format(home=home_path))
#     if not exists(output_dir):
#         makedirs(output_dir)
#
#     # Prepare input ressources
#     features_file = join(output_dir, 'features.npy')
#     labels_file = join(output_dir, 'labels.npy')
#
#     if not isfile(features_file):
#         # Load data
#         input_dir = normpath('{home}/Data/Skin/Thumbnails/'.format(home=home_path))
#         benign_dir = join(input_dir, 'Benin')
#         malignant_dir = join(input_dir, 'Malin')
#         features, labels = extract_deepfeatures(benign_dir, 'benin')
#         features_m, labels_m = extract_deepfeatures(malignant_dir, 'malin')
#
#         features = concatenate((features, features_m), axis=0)
#         features = StandardScaler().fit_transform(features)
#         labels = concatenate((labels, labels_m), axis=0)
#
#         # Make a save of data
#         save(features_file, features)
#         save(labels_file, labels)
#     else:
#         # Reload data if exist
#         features = load(features_file)
#         labels = load(labels_file)
#
#     # Save data as projector to visualize them
#     DataProjector.project_data(datas=features, labels=labels, path=join(output_dir, 'Projector'))
#
#     ######
#     # Dummy learning pipe
#     pipe = Pipeline([('pca', PCA()),
#                      ('clf', DummyClassifier())])
#
#     # Define parameters to validate through grid CV
#     parameters = {'pca__n_components': [0.95]}
#
#     # Classify and write data results
#     classifier = Classifier(pipeline=pipe, params=parameters,
#                             inner_cv=StratifiedKFold(n_splits=5), outer_cv=StratifiedKFold(n_splits=5))
#     result = classifier.evaluate(features=features, labels=labels)
#     ResultWriter(result).write_results(dir_name=output_dir, name='Dummy')
#
#
#     ######
#     # Machine learning pipe
#     pipe = Pipeline([('pca', PCA()),
#                      ('clf', SVC(kernel='linear', probability=True))])
#
#     # Define parameters to validate through grid CV
#     parameters = {'pca__n_components': [0.95],
#                   'clf__C': geomspace(0.01, 1000, 6)}
#
#     # Classify and write data results
#     classifier = Classifier(pipeline=pipe, params=parameters,
#                             inner_cv=StratifiedKFold(n_splits=5), outer_cv=StratifiedKFold(n_splits=5))
#     result = classifier.evaluate(features=features, labels=labels)
#     ResultWriter(result).write_results(dir_name=output_dir, name='Machine Learning')
#
#     ######
#     # Deep learning pipe
#     keras_model = KerasClassifier(build_fn=DeepModels.get_confocal_final, epochs=100, batch_size=10, verbose=0)
#     pipe = Pipeline([('pca', PCA()),
#                      ('clf', keras_model)])
#
#     # Define parameters to validate through grid CV
#     parameters = {'pca__n_components': [0.95]}
#                   # 'clf__optimizer' : ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']}
#
#     # Classify and write data results
#     classifier = Classifier(pipeline=pipe, params=parameters,
#                             inner_cv=StratifiedKFold(n_splits=5), outer_cv=StratifiedKFold(n_splits=5))
#     result = classifier.evaluate(features=features, labels=labels)
#     ResultWriter(result).write_results(dir_name=output_dir, name='Deep Learning')

from itertools import product
import keras
from keras import Model, Sequential
from keras.engine import Layer
from keras.layers import Dense, K
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from numpy import arange, geomspace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from toolbox.core.transforms import DWTTransform, PLSTransform


class DeepModels:

    @staticmethod
    def get_dummy_model(output_classes):
        keras.layers.RandomLayer = RandomLayer
        # Extract labels
        model = Sequential()
        model.add(RandomLayer(output_classes, input_shape=(None, None, 3)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def get_confocal_preprocessing():
        return preprocess_input

    @staticmethod
    def get_confocal_model(output_classes):
        # We get the deep extractor part as include_top is false
        inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='max')

        # Now we customize the output consider our application field
        prediction_layers = Dense(output_classes, activation='softmax', name='predictions')(inception_model.output)

        # And defined model based on our input and next output
        model = Model(inputs=inception_model.input, outputs=prediction_layers)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    @staticmethod
    def get_confocal_final(optimizer='adam'):
        # Now we customize the output consider our application field
        model = Sequential()
        model.add(Dense(2, activation='softmax', name='predictions'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def __get_class_number(labels):
        return len(set(labels))


class SimpleModels:

    @staticmethod
    def get_dummy_process():
        pipe = Pipeline([('clf', DummyClassifier())])
        # Define parameters to validate through grid CV
        parameters = {}
        return pipe, parameters

    @staticmethod
    def get_testing_process():
        extractors = SimpleModels.get_extractors()
        estimators = SimpleModels.get_estimators()

        processes = []
        for prod in product(extractors, estimators):
            pipe = Pipeline(prod[0][0] + prod[1][0])
            params = prod[0][1].copy()
            params.update(prod[1][1])
            processes.append({'pipe': pipe, 'params': params})

        return processes

    @staticmethod
    def get_ahmed_process():
        pipe = Pipeline([('dwt', DWTTransform()),
                         ('cluster', KMeans()),
                         ('clf', SVC(probability=True)),
                         ])
        # Define parameters to validate through grid CV
        parameters = {
            'dwt__mode': ['db6'],
            'clf__C': geomspace(0.01, 1000, 6),
            'clf__gamma': [0.001, 0.0001],
            'clf__kernel': ['rbf']
        }
        return pipe, parameters

    @staticmethod
    def get_lda_process():
        pipe = Pipeline([('pca', PCA()),
                         ('lda', LinearDiscriminantAnalysis()),  # , PLSCanonical, CCA ?
                         ('clf', SVC(probability=True)),
                         ])
        # Define parameters to validate through grid CV
        parameters = {
            'pca__n_components': [0.99],
            'lda__n_components': range(2, 12, 2),
            'clf__C': [1, 10, 100, 1000],
            'clf__gamma': [0.001, 0.0001],
            'clf__kernel': ['rbf']
        }
        return pipe, parameters

    @staticmethod
    def get_pca_process():
        pipe = Pipeline([('pca', PCA()),
                         ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True)),
                         ])
        # Define parameters to validate through grid CV
        parameters = {
            'pca__n_components': [0.95, 0.975, 0.99],
            'clf__C': geomspace(0.01, 1000, 6),
            'clf__gamma': geomspace(0.01, 1000, 6)
        }
        return pipe, parameters

    @staticmethod
    def get_pls_process():
        pipe = Pipeline([('pls', PLSTransform()),
                         ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True)),
                         ])
        # Define parameters to validate through grid CV
        parameters = {
            'pls__n_components': range(2, 12, 2),
            'clf__C': geomspace(0.01, 1000, 6),
            'clf__gamma': geomspace(0.01, 1000, 6)
        }
        return pipe, parameters

    @staticmethod
    def get_mlp_process():
        pipe = Pipeline([('lda', LinearDiscriminantAnalysis()),  # , PLSCanonical, CCA ?
                         ('clf', MLPClassifier(verbose=0, random_state=0, max_iter=400))
                         ])
        # Define parameters to validate through grid CV
        parameters = {'clf__solver': 'sgd', 'clf__learning_rate': 'constant', 'clf__momentum': 0,
                      'clf__learning_rate_init': 0.2}
        return pipe, parameters

    @staticmethod
    def get_dwt_process():
        pipe = Pipeline([('dwt', DWTTransform()),
                         ('clf', SVC(probability=True)),
                         ])
        # Define parameters to validate through grid CV
        parameters = {
            'dwt__mode': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6'],
            'clf__C': geomspace(0.01, 1000, 6),
            'clf__gamma': [0.001, 0.0001],
            'clf__kernel': ['rbf']
        }
        return pipe, parameters

    @staticmethod
    def get_estimators():
        estimators = []
        estimators.append(([('SVC', SVC(probability=True))],
                           {
                               'SVC__C': geomspace(0.01, 1000, 6),
                               'SVC__gamma': geomspace(0.01, 1000, 6)
                           }))
        estimators.append(([('SVCl', SVC(kernel='linear', probability=True))],
                           {
                               'SVCl__C': geomspace(0.01, 1000, 6)
                           }))
        estimators.append(([('KNN', KNeighborsClassifier())],
                           {
                               'KNN__n_neighbors': arange(1, 10, 2)
                           }))
        return estimators

    @staticmethod
    def get_extractors():
        extractors = []
        extractors.append(([('PCA', PCA())],
                           {
                               'PCA__n_components': [0.95, 0.975, 0.99, 0.995, 0.999]
                           }))
        extractors.append(([('PLS', PLSTransform())],
                           {
                               'PLS__n_components': range(2, 12, 2)
                           }))

        extractors.append(([('DWT', DWTTransform())],
                           {
                               'DWT__mode': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6']
                           }))
        return extractors


class RandomLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.random_uniform_variable(shape=(1, self.output_dim), low=0, high=1)
        # val = random((1, self.output_dim))
        # return K.variable(value=val)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base_config = super().get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

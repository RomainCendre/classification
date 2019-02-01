from itertools import product
from os import makedirs
from os.path import normpath

from sklearn.preprocessing import StandardScaler
from time import strftime, gmtime, time
import keras
from keras import Sequential, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.engine import Layer
from keras.layers import Dense, K, Conv2D, GlobalMaxPooling2D
from keras import applications
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
from toolbox.tools.tensorboard import TensorBoardWriter, TensorBoardTool


class DeepModels:

    @staticmethod
    def get_callbacks(folder=None):
        callbacks = []
        if folder is not None:
            # Workdir creation
            current_time = strftime('%Y_%m_%d_%H_%M_%S', gmtime(time()))
            work_dir = normpath('{folder}/Graph/{time}'.format(folder=folder, time=current_time))
            makedirs(work_dir)

            # Tensorboard tool launch
            tb_tool = TensorBoardTool(work_dir)
            tb_tool.write_batch()
            tb_tool.run()

            callbacks.append(TensorBoardWriter(log_dir=work_dir))

        callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min'))
        callbacks.append(EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto'))
        return callbacks

    @staticmethod
    def get_dummy_model(output_classes):
        keras.layers.RandomLayer = RandomLayer
        # Extract labels
        model = Sequential()
        model.add(RandomLayer(output_classes, input_shape=(None, None, 3)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def get_scratch_patch_model(output_classes):
        model = Sequential()
        model.add(Conv2D(10, (1, 3), strides=(2, 2), input_shape=(250, 250, 3), activation='linear', name='Convolution_1'))
        model.add(Conv2D(10, (3, 1), strides=(2, 2), activation='relu', name='Convolution_2'))
        model.add(GlobalMaxPooling2D(name='Pooling_2D'))
        model.add(Dense(output_classes, activation='softmax', name='Predictions'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def get_transfer_model(optimizer='adam'):
        # Now we customize the output consider our application field
        model = Sequential()
        model.add(Dense(2, activation='softmax', name='predictions'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    @staticmethod
    def get_transfer_learning_model(output_classes, architecture='InceptionV3'):

        # We get the deep extractor part as include_top is false
        if architecture == 'VGG16':
            base_model = applications.VGG16(weights='imagenet', include_top=False, pooling='max')
        if architecture == 'VGG19':
            base_model = applications.VGG19(weights='imagenet', include_top=False, pooling='max')
        else:
            base_model = applications.InceptionV3(weights='imagenet', include_top=False, pooling='max')

        # Now we customize the output consider our application field
        prediction_layers = Dense(output_classes, activation='softmax', name='predictions')(base_model.output)

        # And defined model based on our input and next output
        model = Model(inputs=base_model.input, outputs=prediction_layers)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    @staticmethod
    def get_transfer_learning_preprocessing(architecture='InceptionV3'):
        if architecture == 'VGG16':
            return applications.vgg16.preprocess_input
        if architecture == 'VGG19':
            return applications.vgg19.preprocess_input
        else:
            return applications.inception_v3.preprocess_input

    @staticmethod
    def get_memory_usage(batch_size, model, unit=(1024.0 ** 3)):
        shapes_mem_count = 0
        for l in model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = sum([K.count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = sum([K.count_params(p) for p in set(model.non_trainable_weights)])

        if K.floatx() == 'float16':
            number_size = 2.0
        elif K.floatx() == 'float64':
            number_size = 8.0
        else:
            number_size = 4.0

        total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
        return round(total_memory / unit, 3)


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
    def get_linear_svm_process():
        pipe = Pipeline([('scale', StandardScaler()),
                         ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
        # Define parameters to validate through grid CV
        parameters = {
            'clf__C': geomspace(0.01, 1000, 6),
            'clf__gamma': geomspace(0.01, 1000, 6)
        }
        return pipe, parameters

    @staticmethod
    def get_pca_process():
        pipe = Pipeline([('scale', StandardScaler()),
                         ('pca', PCA()),
                         ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
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

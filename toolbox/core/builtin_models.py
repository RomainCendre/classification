from os import makedirs
from os.path import normpath
from sklearn.feature_selection import chi2
from time import strftime, gmtime, time
import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, K, Conv2D, GlobalMaxPooling2D, Dropout
from keras import applications
from keras import Sequential, Model
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import geomspace
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from toolbox.core.layers import RandomLayer
from toolbox.core.models import KerasBatchClassifier, SelectAtMostKBest
from toolbox.core.transforms import DWTTransform, PLSTransform, HaralickTransform, DWTDescriptorTransform, \
    PNormTransform
from toolbox.tools.tensorboard import TensorBoardWriter, TensorBoardTool


class Transforms:

    @staticmethod
    def get_application(architecture='InceptionV3', pooling='max'):
        # We get the deep extractor part as include_top is false
        if architecture == 'MobileNet':
            model = applications.MobileNet(weights='imagenet', include_top=False, pooling=pooling)
        elif architecture == 'VGG16':
            model = applications.VGG16(weights='imagenet', include_top=False, pooling=pooling)
        elif architecture == 'VGG19':
            model = applications.VGG19(weights='imagenet', include_top=False, pooling=pooling)
        else:
            model = applications.InceptionV3(weights='imagenet', include_top=False, pooling=pooling)
        model.name = architecture

        return model

    @staticmethod
    def get_image_dwt():
        pipe = Pipeline([('dwt', DWTDescriptorTransform(wavelets=['db2'], scale=4))])
        pipe.name = 'DWT'
        # Define parameters to validate through grid CV
        parameters = {}#{'dwt__mode': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6']}
        return pipe, parameters

    @staticmethod
    def get_keras_extractor(architecture='InceptionV3', pooling='max'):
        extractor_params = {'architecture': architecture,
                            'batch_size': 1,
                            'pooling': pooling,
                            'preprocessing_function':
                                Classifiers.get_preprocessing_application(architecture=architecture)}
        extractor = KerasBatchClassifier(Transforms.get_application, **extractor_params)
        extractor.name = 'Keras{pool}'.format(pool=pooling)
        return extractor

    @staticmethod
    def get_linear_dwt():
        pipe = Pipeline([('dwt', DWTTransform(mode='db1'))])
        pipe.name = 'DWT'
        # Define parameters to validate through grid CV
        parameters = {}#{'dwt__mode': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6']}
        return pipe, parameters

    @staticmethod
    def get_dwt():
        pipe = Pipeline([('dwt', DWTTransform(mode='db1'))])
        pipe.name = 'DWT'
        # Define parameters to validate through grid CV
        parameters = {}#{'dwt__mode': ['db1', 'db2', 'db3', 'db4', 'db5', 'db6']}
        return pipe, parameters

    @staticmethod
    def get_haralick(mean=True):
        pipe = Pipeline([('haralick', HaralickTransform(mean=mean))])
        if mean:
            pipe.name = 'HaralickMean'
        else:
            pipe.name = 'Haralick'
        # Define parameters to validate through grid CV
        parameters = {}
        return pipe, parameters

    @staticmethod
    def get_pca():
        pipe = Pipeline([('pca', PCA())])
        pipe.name = 'PCA'
        # Define parameters to validate through grid CV
        parameters = {'pca__n_components': [0.95, 0.975, 0.99]}
        return pipe, parameters

    @staticmethod
    def get_pls():
        pipe = Pipeline([('pls', PLSTransform())])
        pipe.name = 'PLS'
        # Define parameters to validate through grid CV
        parameters = {'pls__n_components': range(2, 12, 2)}
        return pipe, parameters


class Classifiers:

    @staticmethod
    def get_keras_callbacks(model_calls=[], folder=None):
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

        if 'Reduce' in model_calls:
            callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min'))

        if 'Early' in model_calls:
            callbacks.append(EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1, mode='auto'))

        return callbacks

    @staticmethod
    def get_keras_classifier(output_classes):
        predictor = KerasClassifier(Classifiers.get_dense)
        predictor.name = 'Deep_predictor'
        predictor_params = {'batch_size': 1,
                            'epochs': 100,
                            'optimizer': 'adam',
                            'output_classes': output_classes}
        return predictor, predictor_params

    @staticmethod
    def get_dense(output_classes, pretrain_model=None, nb_layers=1, activation='softmax', optimizer='adam', metrics=['accuracy']):
        # Now we customize the output consider our application field
        model = Sequential()
        if nb_layers > 1:
            model.add(Dense(1024, activation='relu', name='predictions_dense_1'))
            model.add(Dropout(0.5, name='predictions_dropout_1'))
        if nb_layers > 2:
            model.add(Dense(1024, activation='relu', name='predictions_dense_2'))
            model.add(Dropout(0.5, name='predictions_dropout_2'))
        if nb_layers > 3:
            model.add(Dense(512, activation='relu', name='predictions_dense_3'))
            model.add(Dropout(0.5, name='predictions_dropout_3'))

        # Now we customize the output consider our application field
        model.add(Dense(output_classes, activation=activation, name='predictions_final'))

        if output_classes > 2:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    @staticmethod
    def get_dummy_deep(output_classes):
        keras.layers.RandomLayer = RandomLayer
        # Extract labels
        model = Sequential()
        model.add(RandomLayer(output_classes))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.name = 'Dummy_Deep'
        return model

    @staticmethod
    def get_dummy_simple():
        pipe = Pipeline([('clf', DummyClassifier())])
        pipe.name = 'Dummy_Simple'
        # Define parameters to validate through grid CV
        parameters = {}
        return pipe, parameters

    @staticmethod
    def get_linear_svm(reduce=None, scaling=True):
        steps = []
        parameters = {}

        # Add dimensions reducer
        if reduce is not None:
            steps.append(('pca', PCA()))
            parameters.update({'pca__n_components': [reduce]})

        # Add scaling step
        if scaling:
            steps.append(('scale', StandardScaler()))

        steps.append(('clf', SVC(kernel='linear', class_weight='balanced', probability=True)))
        pipe = Pipeline(steps)
        pipe.name = 'LinearSVM'
        # Define parameters to validate through grid CV
        parameters.update({
            'clf__C': geomspace(0.01, 1000, 6).tolist()
        })
        return pipe, parameters

    @staticmethod
    def get_norm_model(patch=True):
        steps = []
        parameters = {}

        # Add dimensions reducer
        if patch:
            steps.append(('norm', PNormTransform()))
            parameters.update({'norm__p': [2, 3, 4]})
        else:
            steps.append(('norm1', PNormTransform(axis=2)))
            parameters.update({'norm1__p': [2, 3, 4]})
            steps.append(('norm2', PNormTransform()))
            parameters.update({'norm2__p': [2, 3, 4]})
        # Add scaling step
        steps.append(('select', SelectAtMostKBest(chi2, k=100)))
        steps.append(('scale', StandardScaler()))
        steps.append(('clf', SVC(kernel='linear', class_weight='balanced', probability=True)))
        pipe = Pipeline(steps)
        pipe.name = 'LinearSVM'
        # Define parameters to validate through grid CV
        parameters.update({
            'clf__C': geomspace(0.01, 1000, 6).tolist()
        })
        return pipe, parameters

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

    @staticmethod
    def get_preprocessing_application(architecture='InceptionV3'):
        if architecture == 'MobileNet':
            return applications.mobilenet.preprocess_input
        elif architecture == 'VGG16':
            return applications.vgg16.preprocess_input
        elif architecture == 'VGG19':
            return applications.vgg19.preprocess_input
        else:
            return applications.inception_v3.preprocess_input

    @staticmethod
    def get_scratch(output_classes, mode):
        model = Sequential()
        if mode == 'patch':
            model.add(Conv2D(10, (1, 3), strides=(2, 2), input_shape=(250, 250, 3), activation='linear', name='Convolution_1'))
        else:
            model.add(Conv2D(10, (1, 3), strides=(2, 2), input_shape=(1000, 1000, 3), activation='linear', name='Convolution_1'))
        model.add(Conv2D(10, (3, 1), strides=(2, 2), activation='relu', name='Convolution_2'))
        model.add(GlobalMaxPooling2D(name='Pooling_2D'))
        model.add(Dense(output_classes, activation='softmax', name='Predictions'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def get_fine_tuning(output_classes, trainable_layers=0, added_layers=0, architecture='InceptionV3', optimizer='adam', metrics=['accuracy']):

        # We get the deep extractor part as include_top is false
        base_model = Transforms.get_application(architecture)

        # We disable all layers trainable property
        for layer in base_model.layers:
            layer.trainable = False

        # Decide wich layers are trainables
        train_index = len(base_model.layers)-trainable_layers

        # Now switch it to trainable
        for layer in base_model.layers[train_index:]:
            layer.trainable = True

        x = base_model.output
        # Now we customize the output consider our application field
        if added_layers > 1:
            x = Dense(1024, activation='relu', name='predictions_dense_1')(x)
            x = Dropout(0.5, name='predictions_dropout_1')(x)
        if added_layers > 2:
            x = Dense(1024, activation='relu', name='predictions_dense_2')(x)
            x = Dropout(0.5, name='predictions_dropout_2')(x)
        if added_layers > 3:
            x = Dense(512, activation='relu', name='predictions_dense_3')(x)
            x = Dropout(0.5, name='predictions_dropout_3')(x)

        x = Dense(output_classes, activation='softmax', name='predictions_final')(x)

        if output_classes > 2:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'

        model = Model(inputs=base_model.inputs, outputs=x)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model


class BuiltInModels:

    @staticmethod
    def get_ahmed():
        pipe = Pipeline([('wavelet', DWTTransform()),
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
    def get_dummy():
        pipe = Pipeline([('clf', DummyClassifier())])
        # Define parameters to validate through grid CV
        parameters = {}
        return pipe, parameters

    @staticmethod
    def get_haralick():
        pipe = Pipeline([('haralick', HaralickTransform()),
                         ('scale', StandardScaler()),
                         ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
        # Define parameters to validate through grid CV
        parameters = {
            'clf__C': geomspace(0.01, 1000, 6).tolist(),
            'clf__gamma': geomspace(0.01, 1000, 6).tolist()
        }
        return pipe, parameters

    @staticmethod
    def get_fine_tuning(output_classes, trainable_layers=0, added_layers=0):
        model = KerasBatchClassifier(build_fn=Classifiers.get_fine_tuning)
        parameters = {# Build paramters
                      'architecture': 'VGG16',
                      'optimizer': 'adam',
                      'metrics': [['accuracy']],
                      # Parameters for fit
                      'epochs': 100,
                      'batch_size': 6,
                      }
        parameters.update({'output_classes': output_classes,
                           'trainable_layers': trainable_layers,
                           'added_layers': added_layers})
        fit_parameters = {# Transformations
                            'rotation_range': 180,
                            'horizontal_flip': True,
                            'vertical_flip': True,
                            }

        return model, parameters, fit_parameters

    @staticmethod
    def get_linear_svm_process():
        pipe = Pipeline([('scale', StandardScaler()),
                         ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
        # Define parameters to validate through grid CV
        parameters = {
            'clf__C': geomspace(0.01, 1000, 6).tolist(),
            'clf__gamma': geomspace(0.01, 1000, 6).tolist()
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
            'clf__C': geomspace(0.01, 1000, 6).tolist(),
            'clf__gamma': geomspace(0.01, 1000, 6).tolist()
        }
        return pipe, parameters

    @staticmethod
    def get_pls_process():
        pipe = Pipeline([('pls', PLSTransform()),
                         ('clf', SVC(kernel='linear', class_weight='balanced', probability=True)),
                         ])
        # Define parameters to validate through grid CV
        parameters = {
            'pls__n_components': list(range(2, 12, 2)),
            'clf__C': geomspace(0.01, 1000, 6).tolist()
        }
        return pipe, parameters

from copy import deepcopy, copy
from os import makedirs
from os.path import normpath

from sklearn.metrics import roc_curve, accuracy_score
from time import strftime, gmtime, time
import keras
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.engine import Layer
from keras.layers import Dense, K, Conv2D, GlobalMaxPooling2D
from keras import applications
from keras import Sequential, Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.generic_utils import has_arg, to_list
from numpy import arange, geomspace, array, searchsorted, unique, hstack, zeros, concatenate, ones, argmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import types
from toolbox.core.generators import ResourcesGenerator
from toolbox.core.transforms import DWTTransform, PLSTransform, HaralickDescriptorTransform
from toolbox.tools.tensorboard import TensorBoardWriter, TensorBoardTool


class DeepModels:

    @staticmethod
    def get_application_model(architecture='InceptionV3'):
        # We get the deep extractor part as include_top is false
        if architecture == 'MobileNet':
            model = applications.MobileNet(weights='imagenet', include_top=False, pooling='max')
        elif architecture == 'VGG16':
            model = applications.VGG16(weights='imagenet', include_top=False, pooling='max')
        elif architecture == 'VGG19':
            model = applications.VGG19(weights='imagenet', include_top=False, pooling='max')
        else:
            model = applications.InceptionV3(weights='imagenet', include_top=False, pooling='max')

        return model

    @staticmethod
    def get_application_preprocessing(architecture='InceptionV3'):
        if architecture == 'VGG16':
            return applications.vgg16.preprocess_input
        if architecture == 'VGG19':
            return applications.vgg19.preprocess_input
        else:
            return applications.inception_v3.preprocess_input

    @staticmethod
    def get_callbacks(model_calls=[], folder=None):
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
    def get_dummy_model(output_classes):
        keras.layers.RandomLayer = RandomLayer
        # Extract labels
        model = Sequential()
        model.add(RandomLayer(output_classes))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def get_scratch_model(output_classes, mode):
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
    def get_prediction_model(output_classes, optimizer='adam', metrics=['accuracy']):
        # Now we customize the output consider our application field
        model = Sequential()
        # Now we customize the output consider our application field
        model.add(Dense(output_classes, activation='softmax', name='predictions'))

        if output_classes > 2:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    @staticmethod
    def get_transfer_learning_model(output_classes, architecture='InceptionV3', optimizer='adam', metrics=['accuracy']):

        # We get the deep extractor part as include_top is false
        base_model = DeepModels.get_application_model(architecture)

        # Now we customize the output consider our application field
        prediction_layers = Dense(output_classes, activation='softmax', name='predictions')(base_model.output)

        if output_classes > 2:
            loss = 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'

        # And defined model based on our input and next output
        model = Model(inputs=base_model.input, outputs=prediction_layers)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model

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
    def get_ahmed_process():
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
    def get_dummy_process():
        pipe = Pipeline([('clf', DummyClassifier())])
        # Define parameters to validate through grid CV
        parameters = {}
        return pipe, parameters

    @staticmethod
    def get_haralick_process():
        pipe = Pipeline([('haralick', HaralickDescriptorTransform()),
                         ('scale', StandardScaler()),
                         ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
        # Define parameters to validate through grid CV
        parameters = {
            'clf__C': geomspace(0.01, 1000, 6).tolist(),
            'clf__gamma': geomspace(0.01, 1000, 6).tolist()
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
    def get_dwt_process():
        pipe = Pipeline([('wavelet', DWTTransform()),
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


class KerasBatchClassifier(KerasClassifier):

    def check_params(self, params):
        """Checks for user typos in `params`.

        # Arguments
            params: dictionary; the parameters to be checked

        # Raises
            ValueError: if any member of `params` is not a valid argument.
        """
        local_param = deepcopy(params)
        legal_params_fns = [ResourcesGenerator.__init__, ResourcesGenerator.flow_from_paths,
                            Sequential.fit_generator, Sequential.predict_generator,
                            Sequential.evaluate_generator]
        found_params = set()
        for param_name in params:
            for fn in legal_params_fns:
                if has_arg(fn, param_name):
                    found_params.add(param_name)
        if not len(found_params) == 0:
            [local_param.pop(key) for key in list(found_params)]
        super().check_params(local_param)

    def init_model(self, y=[]):
        self.sk_params.update({'output_classes': len(unique_labels(y))})
        # Get the deep model
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        # Store labels
        y = array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = unique(y)
            y = searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))

    def fit(self, X, y, callbacks=[], X_validation=None, y_validation=None, **kwargs):
        self.init_model(y)

        # Get arguments for fit
        fit_args = deepcopy(self.filter_sk_params(Sequential.fit_generator))
        fit_args.update(kwargs)

        # Create generator
        generator = ResourcesGenerator(**self.filter_sk_params(ResourcesGenerator.__init__))
        train = generator.flow_from_paths(X, y, **self.filter_sk_params(ResourcesGenerator.flow_from_paths))

        validation = None
        if X_validation is not None:
            validation = generator.flow_from_paths(X_validation, y_validation,
                                                   **self.filter_sk_params(ResourcesGenerator.flow_from_paths))

        if not self.model._is_compiled:
            tr_x, tr_y = train[0]
            self.model.fit(tr_x, tr_y)

        self.history = self.model.fit_generator(generator=train, validation_data=validation, callbacks=callbacks,
                                                **fit_args)

        return self.history

    def predict(self, X, **kwargs):
        probs = self.predict_proba(X, **kwargs)
        if probs.shape[-1] > 1:
            classes = probs.argmax(axis=-1)
        else:
            classes = (probs > 0.5).astype('int32')
        return self.classes_[classes]

    def predict_proba(self, X, **kwargs):
        self.init_model()
        # Get arguments for generator
        fit_args = deepcopy(self.filter_sk_params(ResourcesGenerator.flow_from_paths))
        fit_args.update(self.__filter_params(kwargs, ResourcesGenerator.flow_from_paths))
        fit_args.update({'shuffle': False})
        fit_args.update({'batch_size': 1})

        # Create generator
        generator = ResourcesGenerator(preprocessing_function=kwargs.get('Preprocess', None))
        valid = generator.flow_from_paths(X, **fit_args)

        # Get arguments for predict
        fit_args = deepcopy(self.filter_sk_params(Sequential.predict_generator))
        fit_args.update(self.__filter_params(kwargs, Sequential.predict_generator))

        probs = self.model.predict_generator(generator=valid, **fit_args)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = hstack([1 - probs, probs])
        return probs

    def score(self, X, y, **kwargs):
        kwargs = self.filter_sk_params(Sequential.evaluate_generator, kwargs)

        # Create generator
        generator = ResourcesGenerator(preprocessing_function=kwargs.get('Preprocess', None))
        valid = generator.flow_from_paths(X, y, batch_size=1, shuffle=False)

        # Get arguments for fit
        fit_args = deepcopy(self.filter_sk_params(Sequential.evaluate_generator))
        fit_args.update(kwargs)

        # sparse to numpy array
        outputs = self.model.evaluate_generator(generator=valid, **fit_args)
        outputs = to_list(outputs)
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')

    def __filter_params(self, params, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `sk_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res


class ClassifierPatch(BaseEstimator, ClassifierMixin):

    def __init__(self, extractor, model, patch_size):
        self.extractor = extractor
        self.model = model
        self.patch_size = patch_size

    def fit(self, X, y, **kwargs):
        features = self.__extract_features_patch(X, **kwargs)
        self.model.fit(features, y)
        return self

    def predict(self, X, **kwargs):
        features = self.__extract_features_patch(X, **kwargs)
        return self.model.predict(features)

    def predict_proba(self, X, **kwargs):
        features = self.__extract_features_patch(X, **kwargs)
        return self.model.predict_proba(features)

    def __extract_features_patch(self, X, **kwargs):
        # Browse images
        predictions = []
        probabilities = []
        for patches in X:
            sub_predictions = None
            sub_probabilities = []
            for patch in patches:
                current = self.extractor.predict_proba(patch.reshape(1, -1))
                if sub_predictions is None:
                    sub_predictions = zeros(current.shape)
                sub_predictions[:, current.argmax(axis=1)] += 1
                # sub_probabilities.append(current)
            predictions.append(sub_predictions/len(patches))

        return concatenate(array(predictions), axis=0)

    @staticmethod
    def __filter_params(params, fn, override=None):
        """Filters `sk_params` and returns those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override `sk_params`

        # Returns
            res : dictionary containing variables
                in both `sk_params` and `fn`'s arguments.
        """
        override = override or {}
        res = {}
        for name, value in params.items():
            if has_arg(fn, name):
                res.update({name: value})
        res.update(override)
        return res


class RandomLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.random_uniform_variable(shape=(1, self.output_dim), low=0, high=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base_config = super().get_config()
        base_config['output_dim'] = self.output_dim
        return base_config


class PatchClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, hierarchies, metric=None):
        if metric:
            self.metric = metric
        else:
            self.metric = accuracy_score
        self.hierarchies = hierarchies
        self.thresholds = None

    def fit(self, x, y=None):
        """
        This should fit this transformer, but DWT doesn't need to fit to train data

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
        """
        # if not x:
        #     raise Exception('At least one X has to be found.')
        global_score = 0
        self.thresholds = zeros(len(self.hierarchies))
        for index, hierarchy in enumerate(self.hierarchies):
            fpr, tpr, roc_thresholds = roc_curve(y, x[:, hierarchy], pos_label=hierarchy)
            for thresh in roc_thresholds:
                thresholds = copy(self.thresholds)
                thresholds[index] = thresh
                score = self.metric(self.get_predictions(x, thresholds), y)
                if global_score <= score:
                    global_score = score
                    self.thresholds[index] = thresh
        return self

    def predict(self, x, y=None, copy=True):
        """
        This method is the main part of this transformer.
        Return a wavelet transform, as specified mode.

        Args:
             x (:obj): Not used.
             y (:obj): Not used.
             copy (:obj): Not used.
        """
        return self.get_predictions(x, self.thresholds)

    def get_predictions(self, x, thresholds):
        thresh_values = ((ones(x.shape) * thresholds) < x).astype(int)
        priorities = arange(len(thresholds), 0, -1)[self.hierarchies]
        return argmax(thresh_values*priorities, axis=1)

    def predict_proba(self, x, y=None, copy=True):
        return None# return array(self.predictor.predict_proba(x))
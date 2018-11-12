from os.path import join

from keras import Model
from keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.optimizers import SGD
from keras_preprocessing.image import ImageDataGenerator
from numpy import load, save, concatenate, uint8


from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.cm as cm
from matplotlib import pyplot as plt

from numpy import unique, zeros
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from core.generators import ResourcesGenerator
from core.outputs import Results
from tools.tensorboard import TensorBoardWriter


class Classifier:
    """Class that manage a spectrum representation.

     In this class we afford an object that represent a spectrum and all
     needed method to manage it.

     Attributes:
         __pipeline (:obj:):
         __params (:obj:):
         __inner_cv (:obj:):
         __outer_cv (:obj:):

     """

    def __init__(self, pipeline, params, inner_cv, outer_cv, scoring = None):
        """Make an initialisation of SpectraClassifier object.

        Take a pipeline object from scikit learn to process data and params for parameters
        to cross validate.

        Args:
             pipeline (:obj:):
             params (:obj:):
             inner_cv (:obj:):
             outer_cv (:obj:):
        """
        self.__pipeline = pipeline
        self.__params = params
        self.__inner_cv = inner_cv
        self.__outer_cv = outer_cv
        self.scoring = scoring

    def evaluate(self, features, labels, groups=None):
        """

        Args:
             features (:obj:):
             labels (:obj:):
             groups (:obj:):

         Returns:

        """
        # Encode labels to go from string to int
        labels_encode = preprocessing.LabelEncoder()
        labels_encode.fit(labels)
        labels = labels_encode.transform(labels)

        if groups:
            groups_encode = preprocessing.LabelEncoder()
            groups_encode.fit(groups)
            groups = groups_encode.transform(groups)

        # Encode labels to go from string to int
        folds = []
        predictions = zeros(len(labels), dtype='int')
        probabilities = zeros((len(labels), len(list(set(labels)))))
        for fold, (train, test) in enumerate(self.__outer_cv.split(X=features, y=labels, groups=groups)):
            grid_search = GridSearchCV(estimator=self.__pipeline, param_grid=self.__params, cv=self.__inner_cv,
                                       scoring=self.scoring, verbose=1, iid=False)

            if groups:
                grid_search.fit(X=features[train], y=labels[train], groups=groups[train])
            else:
                grid_search.fit(X=features[train], y=labels[train])

            # Folds storage
            folds.append(test)

            # Compute ROC curve data
            probabilities[test] = grid_search.predict_proba(features[test])

            # Kept predictions
            predictions[test] = grid_search.predict(features[test])

            print(grid_search.best_params_ )

        map_index = unique(labels)
        map_index.sort()
        map_index = list(labels_encode.inverse_transform(map_index))
        labels = labels_encode.inverse_transform(labels)
        predictions = labels_encode.inverse_transform(predictions)
        name = "_".join(self.__pipeline.named_steps)
        return Results(labels, folds, predictions, map_index, probabilities, name)


class ClassifierDeep:

    def __init__(self, outer_cv, work_dir, scoring=None):
        self.__outer_cv = outer_cv
        self.work_dir = work_dir
        self.scoring = scoring
        # We get the deep extractor part as include_top is false
        self.model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='max')
        self.generator = ResourcesGenerator(rescale=1. / 255)

    def extract_features(self, paths, labels):
        # Generate an iterator
        iterator = self.generator.flow_from_paths(paths, labels, batch_size=32, class_mode='sparse')
        # Now try to get all bottleneck (features extracted by network)
        bottleneck = []
        labels = []
        for i in range(0, 2):#len(iterator)):
            x, y = next(iterator)
            bottleneck.append(self.model.predict(x))
            labels.append(y)

        bottleneck = concatenate(bottleneck)
        labels = concatenate(labels)
        return bottleneck, labels

    def save_features(self, paths, labels, output_path):
        bottleneck, labels = self.extract_features(paths, labels)
        save(join(output_path, 'bottle.npy'), bottleneck)
        save(join(output_path, 'labels.npy'), labels)

    def evaluate_top(self, output_path):
        bottleneck = load(join(output_path, 'bottle.npy'))
        labels = load(join(output_path, 'labels.npy'))
        # SkinParameters.test(bottleneck, labels, output_path)

    def test_vis(self):
        img = utils.load_img('C:\\Users\\Romain\\Data\\Skin\\Patients\\1\\Microscopy\\v0000000.bmp')
        model = self.get_confocal_model()
        lay = model.layers
        grads = visualize_cam(model, len(lay)-5, filter_indices=None, seed_input=img)#, penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)
        jet_heatmap = uint8(cm.jet(grads)[..., :3] * 255)
        plt.figure()
        f, ax = plt.subplots(1, 2)
        ax.imshow(overlay(jet_heatmap, img))


    def get_confocal_model(self):
        # Set layers to non trainable
        for layer in self.model.layers:
            layer.trainable = False

        # Now we customize the output consider our application field
        # prediction_layers = Dense(1024, activation='relu')(self.model.output)
        # prediction_layers = Dropout(0.5)(prediction_layers)
        # prediction_layers = Dense(1024, activation="relu")(prediction_layers)
        prediction_layers = Dense(2, activation='softmax', name='predictions')(self.model.output)

        # And defined model based on our input and nex output
        model = Model(inputs=self.model.input, outputs=prediction_layers)
        model.compile(metrics=['accuracy'], optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True), loss='categorical_crossentropy')
        return model

    @staticmethod
    def get_callbacks(directory):
        callbacks = []
        callbacks.append(TensorBoardWriter(log_dir=directory))
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto'))
        return callbacks

    def evaluate(self, paths, labels, groups=None):

        # Encode labels to go from string to int
        labels_encode = preprocessing.LabelEncoder()
        labels_encode.fit(labels)
        labels = labels_encode.transform(labels)

        if groups:
            groups_encode = preprocessing.LabelEncoder()
            groups_encode.fit(groups)
            groups = groups_encode.transform(groups)

        # Encode labels to go from string to int
        folds = []
        predictions = zeros(len(labels), dtype='int')
        probabilities = zeros((len(labels), 2))

        for fold, (train, valid) in enumerate(self.__outer_cv.split(X=paths, y=labels, groups=groups)):
            # Announce fold and work dir
            print('Fold number {}'.format(fold+1))
            current_dir = join(self.work_dir, 'Fold {fold}'.format(fold=(fold + 1)))

            # Prepare data
            generator = ResourcesGenerator(rescale=1. / 255)
            train_generator = generator.flow_from_paths(paths[train], labels[train], batch_size=32)
            valid_generator = generator.flow_from_paths(paths[valid], labels[valid], batch_size=32)

            # Create model and fit
            model = self.get_confocal_model()
            callbacks = ClassifierDeep.get_callbacks(current_dir)
            model.fit_generator(generator=train_generator, epochs=100, validation_data=valid_generator, callbacks=callbacks)

            # Folds storage
            folds.append(valid)

            # Compute ROC curve data
            probas = model.predict_generator(valid_generator)
            probabilities[valid] = probas
            # Kept predictions
            predictions[valid] = ClassifierDeep.predict_classes(probas)

        map_index = unique(labels)
        map_index.sort()
        map_index = list(labels_encode.inverse_transform(map_index))
        labels = labels_encode.inverse_transform(labels)
        predictions = labels_encode.inverse_transform(predictions)
        return Results(labels, folds, predictions, map_index, probabilities, "Deep")


    @staticmethod
    def predict_classes(probabilities):
        if probabilities.shape[-1] > 1:
            return probabilities.argmax(axis=-1)
        else:
            return (probabilities > 0.5).astype('int32')


import matplotlib.cm as cm
from keras import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import clone_model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from numpy import concatenate, uint8, unique, zeros, arange, mean, repeat, newaxis, array
from os.path import join

from scipy.misc import imsave
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from vis.utils.utils import load_img
from vis.visualization import visualize_cam, visualize_activation, overlay
from vis.utils import utils

from core.generators import ResourcesGenerator
from core.structures import Results, Result
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

    def __init__(self, pipeline, params, inner_cv, outer_cv, scoring=None):
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
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
        encoded_labels = encoder.transform(labels)

        if groups is not None:
            groups_encode = preprocessing.LabelEncoder()
            groups_encode.fit(groups)
            groups = groups_encode.transform(groups)

        # Encode labels to go from string to int
        results = []
        for fold, (train, test) in enumerate(self.__outer_cv.split(X=features, y=labels, groups=groups)):
            grid_search = GridSearchCV(estimator=self.__pipeline, param_grid=self.__params, cv=self.__inner_cv,
                                       scoring=self.scoring, verbose=1, iid=False)

            if groups is not None:
                grid_search.fit(X=features[train], y=encoded_labels[train], groups=groups[train])
            else:
                grid_search.fit(X=features[train], y=encoded_labels[train])

            for index in test:
                result = Result()
                result.update({"Fold": fold})
                result.update({"Label": labels[index]})
                result.update({"LabelIndex": encoded_labels[index]})
                # result.update({"Reference": valid_generator.filenames[index]})
                # Compute probabilities for ROC curve data
                result.update({"Probability": grid_search.predict_proba(features[newaxis, index])[0]})
                # Kept predictions
                result.update({"Prediction": encoder.inverse_transform(grid_search.predict(features[newaxis, index]))[0]})
                results.append(result)

            print(grid_search.best_params_)

        map_index = unique(encoded_labels)
        map_index.sort()
        map_index = list(encoder.inverse_transform(map_index))
        name = "_".join(self.__pipeline.named_steps)
        return Results(results, map_index, name)


class ClassifierDeep:

    def __init__(self, model, outer_cv, work_dir, preprocess, activation_dir='', scoring=None):
        self.model = model
        self.outer_cv = outer_cv
        self.work_dir = work_dir
        self.scoring = scoring
        self.generator = ResourcesGenerator(rescale=1. / 255)
        self.activation_dir = activation_dir
        self.preprocess = preprocess

    def extract_features(self, paths, labels):
        # Generate an iterator
        iterator = self.generator.flow_from_paths(paths, labels, batch_size=1, class_mode='sparse')

        ft = self.__get_features_extractor()

        # Now try to get all bottleneck (features extracted by network)
        bottleneck = []
        labels = []
        for i in range(0, 2):
            x, y = next(iterator)
            bottleneck.append(self.model.predict(x))
            labels.append(y)

        bottleneck = concatenate(bottleneck)
        labels = concatenate(labels)
        return bottleneck, labels

    @staticmethod
    def __get_activation_map(model, seed_input, predict, image):
        if image.ndim == 2:
            image = repeat(image[:, :, newaxis], 3, axis=2)
        grads = visualize_cam(model, len(model.layers)-1, filter_indices=predict, seed_input=seed_input, backprop_modifier='guided')#, penultimate_layer_idx=None, backprop_modifier=None, grad_modifier=None)
        jet_heatmap = uint8(cm.jet(grads)[..., :3] * 255)
        return overlay(jet_heatmap, image)

    @staticmethod
    def __get_callbacks(directory):
        callbacks = []

        callbacks.append(TensorBoardWriter(log_dir=directory))
        # callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min'))
        # callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto'))
        return callbacks

    def __get_classifier(self):
        model = Model(outputs=self.model.output)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @staticmethod
    def __get_class_number(labels):
        return len(set(labels))

    def __get_features_extractor(self):
        model = Model(inputs=self.model.input)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def evaluate(self, paths, labels, groups=None):

        # Encode labels to go from string to int
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
        encoded_labels = encoder.transform(labels)

        if groups is not None:
            groups_encode = preprocessing.LabelEncoder()
            groups_encode.fit(groups)
            groups = groups_encode.transform(groups)

        results = []

        for fold, (train, valid) in enumerate(self.outer_cv.split(X=paths, y=encoded_labels, groups=groups)):
            # Announce fold and work dir
            print('Fold number {}'.format(fold+1))
            current_dir = join(self.work_dir, 'Fold {fold}'.format(fold=(fold + 1)))

            # Fit model
            model = self.fit(paths[train], encoded_labels[train])

            # Prepare data
            generator = ResourcesGenerator(preprocessing_function=self.preprocess)
            valid_generator = generator.flow_from_paths(paths[valid], labels[valid],
                                                        batch_size=1, shuffle=False)

            # Folds storage
            for index in arange(len(valid_generator)):
                x, y = valid_generator[index]
                result = Result()
                result.update({"Fold": fold})
                result.update({"Label": labels[index]})
                result.update({"LabelIndex": encoded_labels[index]})
                result.update({"Reference": valid_generator.filenames[index]})
                probability = model.predict(x)
                result.update({"Probability": probability[0]})
                # Kept predictions
                pred_class = ClassifierDeep.__predict_classes(probabilities=probability)
                result.update({"Prediction": encoder.inverse_transform(array([pred_class]))[0]})
                results.append(result)

                # if self.activation_dir:
                #     activation = ClassifierDeep.__get_activation_map(model=model, seed_input=x, predict=pred_class,
                #                                                      image=load_img(paths[valid[index]]))
                #     imsave('{dir}{number}_{label}.png'.format(dir=self.activation_dir, number=valid[index],
                #                                               label=labels_encode.inverse_transform(pred_class)), activation)
                    # test_activate = visualize_activation(model, len(model.layers)-1,pred_class)
                    # imsave('{dir}Test_{number}_{label}.png'.format(dir=self.activation_dir, number=valid[index],
                    #                                           label=labels_encode.inverse_transform(pred_class)), test_activate)

        map_index = unique(encoded_labels)
        map_index.sort()
        map_index = list(encoder.inverse_transform(map_index))
        return Results(results, map_index, "Deep")

    def fit(self, paths, labels, batch_size=32, epochs=100):

        # Encode labels to go from string to int
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
        encoded_labels = encoder.transform(labels)

        # Clone locally model
        model = clone_model(self.model)
        model.set_weights(self.model.get_weights())
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Prepare data
        generator = ResourcesGenerator(preprocessing_function=self.preprocess)
        train_generator = generator.flow_from_paths(paths, encoded_labels, batch_size=batch_size)

        # Create model and fit
        model.fit_generator(generator=train_generator, epochs=epochs,
                            callbacks=ClassifierDeep.__get_callbacks(self.work_dir))

        return model

    def evaluate_patch(self, paths, labels):

        # Encode labels to go from string to int
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
        encoded_labels = encoder.transform(labels)
        malignant = encoder.transform(['LM'])[0]
        normal = encoder.transform(['Normal'])[0]

        # Prepare data
        patch_size = 250
        generator = ResourcesGenerator(preprocessing_function=self.preprocess)
        test_generator = generator.flow_from_paths(paths, encoded_labels, batch_size=1, shuffle=False)

        # Encode labels to go from string to int
        results = []
        for index in arange(len(test_generator)):
            x, y = test_generator[index]

            result = Result()
            result.update({"Reference": test_generator.filenames[index]})
            result.update({"Label": labels[index]})
            result.update({"LabelIndex": encoded_labels[index]})

            prediction = []
            for i in range(0, 4):
                for j in range(0, 4):
                    x_patch = x[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
                    prediction.append(ClassifierDeep.__predict_classes(probabilities=self.model.predict(x_patch)))

            # Kept predictions
            result.update({"Prediction": malignant if prediction.count(malignant) > 0 else normal})
            results.append(result)

        map_index = unique(encoded_labels)
        map_index.sort()
        map_index = list(encoder.inverse_transform(map_index))
        return Results(results, map_index, "DeepPatch")

    @staticmethod
    def __predict_classes(probabilities):
        if probabilities.shape[-1] > 1:
            return probabilities.argmax(axis=-1)
        else:
            return (probabilities > 0.5).astype('int32')


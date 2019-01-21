from keras import Model
from keras.models import clone_model
from numpy import concatenate, uint8, unique, zeros, arange, mean, repeat, newaxis, array
from os.path import join

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from toolbox.core.generators import ResourcesGenerator
from toolbox.core.structures import Results, Result, Data
from toolbox.tools.tensorboard import TensorBoardWriter


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

    def evaluate(self, inputs):
        """

        Args:
             inputs (:obj:):

         Returns:

        """
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()
        reference = inputs.get_reference()

        # Encode labels to go from string to int
        results = []
        for fold, (train, test) in enumerate(self.__outer_cv.split(X=datas, y=labels, groups=groups)):
            grid_search = GridSearchCV(estimator=self.__pipeline, param_grid=self.__params, cv=self.__inner_cv,
                                       scoring=self.scoring, verbose=1, iid=False)

            if groups is not None:
                grid_search.fit(X=datas[train], y=labels[train], groups=groups[train])
            else:
                grid_search.fit(X=datas[train], y=labels[train])

            for index in test:
                result = Result()
                result.update({"Fold": fold})
                result.update({"Label": labels[index]})
                result.update({"Reference": reference[index]})
                # Compute probabilities for ROC curve data
                result.update({"Probability": grid_search.predict_proba(datas[newaxis, index])[0]})
                # Kept predictions
                result.update({"Prediction": grid_search.predict(datas[newaxis, index])[0]})
                results.append(result)

        name = "_".join(self.__pipeline.named_steps)
        return Results(results, name)


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

    def evaluate(self, inputs):
        datas = inputs.get_datas()
        labels = inputs.get_labels()
        groups = inputs.get_groups()

        results = []
        work_dir = self.work_dir
        for fold, (train, valid) in enumerate(self.outer_cv.split(X=datas, y=labels, groups=groups)):
            # Announce fold and work dir
            print('Fold number {}'.format(fold+1))
            self.work_dir = join(self.work_dir, 'Fold {fold}'.format(fold=(fold + 1)))
            # Fit model
            model = self.__fit(datas[train], labels[train])

            # Prepare data
            generator = ResourcesGenerator(preprocessing_function=self.preprocess)
            valid_generator = generator.flow_from_paths(datas[valid], labels[valid],
                                                        batch_size=1, shuffle=False)

            # Folds storage
            for index in arange(len(valid_generator)):
                x, y = valid_generator[index]
                result = Result()
                result.update({"Fold": fold})
                result.update({"Label": labels[index]})
                result.update({"Reference": valid_generator.filenames[index]})
                probability = model.predict(x)
                result.update({"Probability": probability[0]})
                # Kept predictions
                result.update({"Prediction": ClassifierDeep.__predict_classes(probabilities=probability)})
                results.append(result)

        self.work_dir = work_dir
        return Results(results, "Deep")

    def fit(self, inputs, batch_size=32, epochs=100):
        # Extract data for fit
        datas = inputs.get_datas()
        labels = inputs.get_labels()

    def __fit(self, datas, labels, batch_size=32, epochs=100):

        # Clone locally model
        model = clone_model(self.model)
        model.set_weights(self.model.get_weights())
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Prepare data
        generator = ResourcesGenerator(preprocessing_function=self.preprocess)
        train_generator = generator.flow_from_paths(datas, labels, batch_size=batch_size)

        # Create model and fit
        model.fit_generator(generator=train_generator, epochs=epochs,
                            callbacks=ClassifierDeep.__get_callbacks(self.work_dir))

        return model

    def evaluate_patch(self, inputs):
        # Extract data for fit
        paths = inputs.get_datas()
        labels = inputs.get_labels()

        # Prepare data
        patch_size = 250
        generator = ResourcesGenerator(preprocessing_function=self.preprocess)
        test_generator = generator.flow_from_paths(paths, labels, batch_size=1, shuffle=False)

        # Encode labels to go from string to int
        results = []
        for index in arange(len(test_generator)):
            x, y = test_generator[index]

            result = Result()
            result.update({"Reference": test_generator.filenames[index]})
            result.update({"Label": labels[index]})
            result.update({"LabelIndex": labels[index]})

            prediction = []
            for i in range(0, 4):
                for j in range(0, 4):
                    x_patch = x[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size, :]
                    prediction.append(ClassifierDeep.__predict_classes(probabilities=self.model.predict(x_patch)))

            # Kept predictions
            result.update({"Prediction": malignant if prediction.count(malignant) > 0 else normal})
            results.append(result)

        return Results(results, "DeepPatch")

    @staticmethod
    def __predict_classes(probabilities):
        if probabilities.shape[-1] > 1:
            return probabilities.argmax(axis=-1)
        else:
            return (probabilities > 0.5).astype('int32')


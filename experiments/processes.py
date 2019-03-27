from os import makedirs
from os.path import join, exists
from tempfile import gettempdir
from keras.wrappers.scikit_learn import BaseWrapper
from toolbox.IO.writers import StatisticsWriter, VisualizationWriter, ResultWriter, DataProjectorWriter
from toolbox.core.classification import Classifier
from keras import backend as K


class Process:

    def begin(self, inner_cv, outer_cv, n_jobs=-1, callbacks=[], settings=None, scoring=None):
        self.settings = settings
        self.classifier = Classifier(callbacks=callbacks, inner_cv=inner_cv, outer_cv=outer_cv,
                                     n_jobs=n_jobs, scoring=scoring)
        self.inputs = []
        self.results = []

    def checkpoint_step(self, inputs, model, folder, projection_folder=None,):
        self.classifier.set_model(model)
        self.classifier.features_checkpoint(inputs, folder)
        # Write data to visualize it
        if projection_folder is not None:
            projection_folder = join(projection_folder, inputs.name)
            if not exists(projection_folder):
                makedirs(projection_folder)
            DataProjectorWriter.project_data(inputs, projection_folder)

    def train_step(self, inputs, model):
        self.classifier.set_model(model)
        return self.classifier.fit(inputs)

    def evaluate_step(self, inputs, model):
        # Evaluate model
        self.classifier.set_model(model)
        self.inputs.append(inputs)
        self.results.append(self.classifier.evaluate(inputs))

    def end(self, output_folder=gettempdir(), name='Default'):
        # Step 1 - Write statistics on current data
        keys = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']
        StatisticsWriter(self.inputs).write_result(keys=keys, dir_name=output_folder, name=name)
        ResultWriter(self.results, self.settings).write_results(dir_name=output_folder, name=name)

        # Step 3 - Visualization of CAM
        # if isinstance(model, BaseWrapper):
        #     model, best_params = self.classifier.fit(inputs)
        #     trainable_count = int(
        #         sum([K.count_params(p) for p in set(model.trainable_weights)]))
        #     non_trainable_count = int(
        #         sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
        #
        #     print('Trainable params: {:,}'.format(trainable_count))
        #     VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder, inputs=inputs)

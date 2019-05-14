from os import makedirs
from os.path import join, exists
from toolbox.IO.writers import StatisticsWriter, ResultWriter, DataProjectorWriter
from toolbox.core.classification import Classifier


class Process:

    def __init__(self, output_folder, name, settings, stats_keys):
        self.folder = output_folder
        self.name = name
        self.results = []
        self.settings = settings
        self.stat_writer = StatisticsWriter(stats_keys, output_folder, name)
        self.classifier = None

    def begin(self, inner_cv, outer_cv, n_jobs=-1, callbacks=[], scoring=None):
        self.classifier = Classifier(callbacks=callbacks, inner_cv=inner_cv, outer_cv=outer_cv,
                                     n_jobs=n_jobs, scoring=scoring)

    def checkpoint_step(self, inputs, model, projection_folder=None):
        self.classifier.set_model(model)
        self.classifier.features_checkpoint(inputs)
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
        try:
            self.__add_input_stat(inputs)
            # Evaluate model
            self.classifier.set_model(model)
            self.results.append(self.classifier.evaluate(inputs))
        except Exception as ex:
            print(ex)

    def end(self):
        ResultWriter(self.results, self.settings).write_results(dir_name=self.folder, name=self.name)
        self.stat_writer.end()

    def __add_input_stat(self, inputs):
        if self.stat_writer is not None:
            self.stat_writer.write(inputs)

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

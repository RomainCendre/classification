from os import makedirs
from os.path import join, exists
from tempfile import gettempdir

from keras.wrappers.scikit_learn import BaseWrapper
from numpy.ma import arange
from sklearn.model_selection import GroupKFold
from toolbox.IO.writers import StatisticsWriter, VisualizationWriter, ResultWriter, DataProjectorWriter
from toolbox.core.classification import Classifier
from toolbox.core.models import ClassifierPatch


class Process:

    def begin(self, inner_cv, outer_cv, callbacks=[], scoring=None):
        self.classifier = Classifier(callbacks=callbacks, inner_cv=inner_cv, outer_cv=outer_cv, scoring=scoring)

    def checkpoint_step(self, inputs, model, folder, params={}, projection_folder=None, projection_name=None):
        self.classifier.set_model(model, params)
        self.classifier.features_checkpoint(inputs, folder)
        # Write data to visualize it
        if projection_folder is not None and projection_name is not None:
            projection_folder = join(projection_folder, projection_name)
            if not exists(projection_folder):
                makedirs(projection_folder)
            DataProjectorWriter.project_data(inputs, projection_folder)

    def train_step(self, inputs, model, params={}):
        self.classifier.set_model(model, params)
        return self.classifier.fit(inputs)

    def end(self, inputs, model, params={}, output_folder=gettempdir(), name='Default'):
        # Step 1 - Write statistics on current data
        keys = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)

        # Step 2 - Evaluate model
        self.classifier.set_model(model, params)
        results = self.classifier.evaluate(inputs, name)
        ResultWriter(inputs, results).write_results(dir_name=output_folder, name=name)

        # Step 3 - Visualization of CAM
        if isinstance(model, BaseWrapper):
            model, best_params = self.classifier.fit(inputs)
            VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder, inputs=inputs)
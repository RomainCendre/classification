from tempfile import gettempdir
from keras.wrappers.scikit_learn import BaseWrapper
from toolbox.IO.writers import StatisticsWriter, VisualizationWriter, ResultWriter
from toolbox.core.classification import Classifier


class Process:

    def begin(self, inner_cv, outer_cv, n_jobs=-1, callbacks=[], scoring=None):
        self.classifier = Classifier(callbacks=callbacks, inner_cv=inner_cv, outer_cv=outer_cv,
                                     n_jobs=n_jobs, scoring=scoring)

    def train_step(self, inputs, model):
        self.classifier.set_model(model)
        return self.classifier.fit(inputs)

    def end(self, inputs, model, output_folder=gettempdir(), name='Default'):
        # Step 1 - Write statistics on current data
        keys = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)

        # Step 2 - Evaluate model
        self.classifier.set_model(model)
        results = self.classifier.evaluate(inputs, name)
        ResultWriter(inputs, results).write_results(dir_name=output_folder, name=name)

        # Step 3 - Visualization of CAM
        if isinstance(model, BaseWrapper):
            model, best_params = self.classifier.fit(inputs)
            VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder, inputs=inputs)

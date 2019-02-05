from keras.wrappers.scikit_learn import BaseWrapper
from numpy.ma import arange
from sklearn.model_selection import GroupKFold
from toolbox.IO.writers import StatisticsWriter, VisualizationWriter, ResultWriter
from toolbox.core.classification import Classifier


class Processes:

    @staticmethod
    def dermatology(inputs, output_folder, model, params, name):
        # Step 1 - Write statistics on current data
        keys = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)

        # Step 2 - Evaluate model
        classifier = Classifier(model, params, callbacks=params.pop('callbacks', []),
                                inner_cv=params.pop('inner_cv', 5), outer_cv=params.pop('outer_cv', 5), scoring=None)
        results = classifier.evaluate(inputs, name)
        ResultWriter(inputs, results).write_results(dir_name=output_folder, name=name)

        # Step 3 - Visualization of CAM
        if isinstance(model, BaseWrapper):
            model, best_params = classifier.fit(inputs)
            VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder, inputs=inputs)

    @staticmethod
    def dermatology_bottleneck(inputs, temp_folder, output_folder, model, m_params, predictor, p_params, name):
        # Step 1 - Write statistics on current data
        keys = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)

        # Step 2-1 - Evaluate model
        classifier = Classifier(model, m_params, callbacks=p_params.pop('callbacks', []),
                                inner_cv=p_params.pop('inner_cv', 5), outer_cv=p_params.pop('outer_cv', 5),
                                scoring=None)
        classifier.write_bottleneck(inputs, temp_folder)

        # Step 2-2 - Evaluate model
        classifier.change_model(predictor, p_params)
        classifier.load_bottleneck(inputs, temp_folder)
        results = classifier.evaluate(inputs, name)
        ResultWriter(inputs, results).write_results(dir_name=output_folder, name=name)

        # Step 3 - Visualization of CAM
        if isinstance(model, BaseWrapper):
            model, best_params = classifier.fit(inputs)
            VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder,
                                                                          inputs=inputs)

    @staticmethod
    def dermatology_pretrain(pretrain_inputs, inputs, output_folder, model, params, name):
        # Step 1 - Fit pre input
        classifier = Classifier(model, params, callbacks=params.pop('callbacks', []),
                                inner_cv=params.pop('inner_cv'), outer_cv=params.pop('outer_cv'), scoring=None)
        classifier.model, best_params = classifier.fit(pretrain_inputs)

        # Step 2 - Write statistics, and Evaluate on final data
        keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)
        result = classifier.evaluate(inputs, name)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

        # Step 3 - Visualization of CAM
        model, best_params = classifier.fit(inputs)
        VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder, inputs=inputs)

    @staticmethod
    def dermatology_pretrain_patch(pretrain_inputs, inputs, benign, malignant, output_folder, model, params, name):
        # Step 1 - Fit pre input
        classifier = Classifier(model, params, callbacks=params.pop('callbacks', []),
                                inner_cv=params.pop('inner_cv'), outer_cv=params.pop('outer_cv'), scoring=None)
        fited_model, fited_params = classifier.fit(pretrain_inputs)
        mod = fited_model
        par = fited_params
        classifier.change_model(mod, par)

        # Step 2 - Write statistics, and Evaluate on final data
        keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)
        result = classifier.evaluate_patch(inputs, malignant, benign, name)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

    @staticmethod
    def otorhinolaryngology(inputs, output_folder, model, params, name):
        # Step 1 - Filter data
        inputs.data.apply_method(name='apply_average_filter', parameters={'size': 5})
        inputs.data.apply_method(name='apply_scaling')
        inputs.data.apply_method(name='change_wavelength', parameters={'wavelength': arange(start=445, stop=962, step=1)})

        # Step 2 - Write statistics on current data
        keys = ['patient_label', 'operator', 'label', 'location']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)

        # Step 3 - Classify and write data results
        classifier = Classifier(model=model, params=params,
                                inner_cv=GroupKFold(n_splits=5), outer_cv=GroupKFold(n_splits=5))
        result = classifier.evaluate(inputs)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

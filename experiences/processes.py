from numpy.ma import arange
from sklearn.model_selection import GroupKFold
from toolbox.IO import otorhinolaryngology
from toolbox.IO.writers import StatisticsWriter, VisualizationWriter, ResultWriter
from toolbox.core.classification import Classifier
from toolbox.core.structures import Inputs, DataSet


class Processes:

    @staticmethod
    def dermatology(inputs, output_folder, model, params, name):
        # Step 1 - Write statistics on current data
        keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)

        # Step 2 - Evaluate model
        classifier = Classifier(model, params, params.pop('inner_cv'), params.pop('outer_cv'), scoring=None)
        results = classifier.evaluate(inputs, name)
        ResultWriter(inputs, results).write_results(dir_name=output_folder, name=name)

        # Step 3 - Visualization of CAM
        model = classifier.fit(inputs)
        VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder, inputs=inputs)

    @staticmethod
    def dermatology_pretrain(pretrain_inputs, inputs, output_folder, model, params, name):
        # Step 1 - Fit pre input
        classifier = Classifier(model, params, params.pop('inner_cv'), params.pop('outer_cv'), scoring=None)
        classifier.model = classifier.fit(pretrain_inputs)

        # Step 2 - Write statistics, and Evaluate on final data
        keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
        StatisticsWriter(inputs).write_result(keys=keys, dir_name=output_folder, name=name)
        result = classifier.evaluate(inputs)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

        # Step 3 - Visualization of CAM
        model = classifier.fit(inputs)
        VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder, inputs=inputs)

    @staticmethod
    def otorhinolaryngology(input_folders, filter_by, output_folder, model, params, name):
        # Import data
        data_set = DataSet()
        for input_folder in input_folders:
            data_set += otorhinolaryngology.Reader().read_table(input_folder)

        # Filter data
        data_set.apply_method(name='apply_average_filter', parameters={'size': 5})
        data_set.apply_method(name='apply_scaling')
        data_set.apply_method(name='change_wavelength', parameters={'wavelength': arange(start=445, stop=962, step=1)})

        inputs = Inputs(data_set, data_tag='Data', label_tag='label', group_tag='patient_name',
                        references_tags=['patient_name', 'spectrum_id'], filter_by=filter_by)

        # Write statistics on current data
        keys = ['patient_label', 'operator', 'label', 'location']
        StatisticsWriter(data_set).write_result(keys=keys, dir_name=output_folder, name=name, filter_by=filter_by)

        # Classify and write data results
        classifier = Classifier(model=model, params=params,
                                inner_cv=GroupKFold(n_splits=5), outer_cv=GroupKFold(n_splits=5))
        result = classifier.evaluate(inputs)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

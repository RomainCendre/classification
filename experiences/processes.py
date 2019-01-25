from numpy.ma import arange
from sklearn.model_selection import GroupKFold, StratifiedKFold
from toolbox.IO import dermatology, otorhinolaryngology
from toolbox.IO.writers import StatisticsWriter, VisualizationWriter, ResultWriter
from toolbox.core.classification import Classifier
from toolbox.core.structures import Inputs, DataSet


class Processes:

    @staticmethod
    def dermatology(inputs, output_folder, model, params, name):
        # Step 1 - Write statistics on current data
        inputs.load()

        # Step 2 - Evaluate
        classifier = Classifier(model, params, params.pop('inner_cv'), params.pop('outer_cv'), scoring=None)
        results = classifier.evaluate(inputs, name)
        ResultWriter(inputs, results).write_results(dir_name=output_folder, name=name)

        # Step 3 - Fit model and evaluate visualization
        model = classifier.fit(inputs)
        VisualizationWriter(model=model.model).write_activations_maps(output_folder=output_folder, inputs=inputs)

    @staticmethod
    def dermatology_pretrain(pretrain_inputs, inputs, filter_by, output_folder, model, params, name):
        # Step 0 - Load pretrain data
        data_set = dermatology.Reader().scan_folder_for_images(pretrain_folder)
        inputs = Inputs(data_set, data_tag='Data', label_tag='Label')

        # Step 1 - Fit
        classifier = Classifier(model, params, params.pop('inner_cv'), params.pop('outer_cv'), scoring=None)
        classifier.model = classifier.fit(inputs)

        # Step 2 - Load data
        data_set = DataSet()
        for input_folder in input_folders:
            data_set += dermatology.Reader().scan_folder(input_folder)

        # Step 3 - Write statistics on current data
        keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
        StatisticsWriter(data_set).write_result(keys=keys, dir_name=output_folder, filter_by=filter_by,
                                                name=name)
        inputs.change_data(data_set, filter_by=filter_by)
        inputs.change_group(group_tag='Patient')

        # Step 2 - Fit and Evaluate
        result = classifier.evaluate(inputs)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

        # Step 3 - Fit model and evaluate visualization
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

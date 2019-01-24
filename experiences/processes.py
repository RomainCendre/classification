from numpy.ma import arange
from sklearn.model_selection import GroupKFold, StratifiedKFold

from toolbox.IO import dermatology,otorhinolaryngology
from toolbox.IO.writers import StatisticsWriter, VisualizationWriter, ResultWriter
from toolbox.core.classification import ClassifierDeep, Classifier
from toolbox.core.structures import Inputs, DataSet


class Processes:

    @staticmethod
    def dermatology(input_folder, output_folder, name, filter_by, learner, epochs):

        # Step 0 - Load data
        data_set = dermatology.Reader().scan_folder(input_folder)

        # Step 1 - Write statistics on current data
        keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
        StatisticsWriter(data_set).write_result(keys=keys, dir_name=output_folder, filter_by=filter_by, name=name)
        inputs = Inputs(data_set, data_tag='Data', label_tag='Label', group_tag='Patient', filter_by=filter_by)

        # Step 2 - Fit and Evaluate
        classifier = ClassifierDeep(model=learner['Model'], outer_cv=StratifiedKFold(n_splits=5, shuffle=True),
                                    preprocess=learner['Preprocess'], work_dir=output_folder)
        result = classifier.evaluate(inputs, epochs=epochs)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

        # Step 3 - Fit model and evaluate visualization
        model = classifier.fit(inputs, epochs=epochs)
        VisualizationWriter(model=model).write_activations_maps(output_folder=output_folder, inputs=inputs)

    @staticmethod
    def dermatology_pretrain(pretrain_folder, input_folder, output_folder, name, filter_by, learner, epochs):
        # Step 0 - Load pretrain data
        data_set = dermatology.Reader().scan_folder_for_images(pretrain_folder)
        inputs = Inputs(data_set, data_tag='Data', label_tag='Label')

        # Step 1 - Fit
        classifier = ClassifierDeep(model=learner['Model'], outer_cv=StratifiedKFold(n_splits=5, shuffle=True),
                                    preprocess=learner['Preprocess'], work_dir=output_folder)
        classifier.model = classifier.fit(inputs, epochs=epochs)

        # Step 2 - Load data
        data_set = dermatology.Reader().scan_folder(input_folder)

        # Step 3 - Write statistics on current data
        keys = ['Sex', 'PatientDiagnosis', 'PatientLabel', 'Label']
        StatisticsWriter(data_set).write_result(keys=keys, dir_name=output_folder, filter_by=filter_by,
                                                name=name)
        inputs.change_data(data_set, filter_by=filter_by)
        inputs.change_group(group_tag='Patient')

        # Step 2 - Fit and Evaluate
        classifier = ClassifierDeep(model=learner['Model'], outer_cv=StratifiedKFold(n_splits=5, shuffle=True),
                                    preprocess=learner['Preprocess'], work_dir=output_folder)
        result = classifier.evaluate(inputs, epochs=epochs)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

        # Step 3 - Fit model and evaluate visualization
        model = classifier.fit(inputs, epochs=epochs)
        VisualizationWriter(model=model).write_activations_maps(output_folder=output_folder, inputs=inputs)

    @staticmethod
    def otorhinolaryngology(input_folders, output_folder, name, filter_by, learner):
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
        classifier = Classifier(pipeline=learner['Model'], params=learner['Parameters'],
                                inner_cv=GroupKFold(n_splits=5), outer_cv=GroupKFold(n_splits=5))
        result = classifier.evaluate(inputs)
        ResultWriter(inputs, result).write_results(dir_name=output_folder, name=name)

from numpy import arange
from os import makedirs
from os.path import expanduser, normpath, join, exists
from sklearn.model_selection import GroupKFold

from IO.otorhinolaryngology import Reader
from IO.writer import ResultsWriter, StatisticsWriter
from core.classification import Classifier
from core.models import Models


def compute_data(data_set, out_dir, name, filter_by, meta):
    # Get process
    pipe, param = Models.get_pls_process()

    # Write statistics on current data
    StatisticsWriter(data_set).write_result(metas=meta, dir_name=out_dir, name=name, filter_by=filter_by)

    # Classify and write data results
    classifier = Classifier(pipeline=pipe, params=param,
                            inner_cv=GroupKFold(n_splits=5), outer_cv=GroupKFold(n_splits=5))
    results = [classifier.evaluate(features=spectra.get_data(filter_by=filter_by),
                                   labels=spectra.get_meta(meta='label', filter_by=filter_by),
                                   groups=spectra.get_meta(meta='patient_name', filter_by=filter_by))]
    ResultsWriter(results).write_results('Cancer', dir_name=out_dir, name=name)


if __name__ == "__main__":

    home_path = expanduser("~")

    # Output dir
    output_dir = normpath('{home}/Results/Neck/'.format(home=home_path))
    if not exists(output_dir):
        makedirs(output_dir)

    # Load data
    data_dir = normpath('{home}/Data/Neck/'.format(home=home_path))
    spectra_patients = Reader(';').read_table(join(data_dir, 'Patients.csv'))
    spectra_temoins = Reader(';').read_table(join(data_dir, 'Temoins.csv'))
    spectra = spectra_patients + spectra_temoins

    # Filter data
    spectra.apply_method(name='apply_average_filter', parameters={'size': 5})
    spectra.apply_method(name='apply_scaling')
    spectra.apply_method(name='change_wavelength', parameters={'wavelength': arange(start=445, stop=962, step=1)})

    # All data
    name_exp = 'Results_All'
    filters = {'label': ['Sain', 'Cancer']}
    metas = ['patient_label', 'device', 'label', 'location']
    compute_data(data_set=spectra, out_dir=output_dir, name=name_exp, filter_by=filters, meta=metas)



#
# # Get testing cases
# processes = ClassificationProcess.get_testing_process()
# pipe_pca, param_pca = ClassificationProcess.get_pca_process()
#
# inner_cv = GroupKFold(n_splits=5)
# outer_cv = GroupKFold(n_splits=5)
#
# # multiprocessing requires the fork to happen in a __main__ protected
# # block
# if __name__ == "__main__":
#
#     # All data
#     spectra = deepcopy(spectra_full)
#     results = []
#     classifier = SpectraClassifier(pipeline=pipe_pca, params=param_pca,
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
#     SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_All')
#
#     # Operators
#     # V1 data
#     spectra = deepcopy(spectra_full)
#     spectra.filter_device('V1')
#     results = []
#     classifier = SpectraClassifier(pipeline=pipe_pca, params=param_pca,
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
#     SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_V1')
#
#     # V2 data
#     spectra = deepcopy(spectra_full)
#     spectra.filter_device('V2')
#     results = []
#     classifier = SpectraClassifier(pipeline=pipe_pca, params=param_pca,
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
#     SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_V2')
#
#     # Locations
#     # Mouth data
#     spectra = deepcopy(spectra_full)
#     spectra.filter_location('Bouche')
#     results = []
#     classifier = SpectraClassifier(pipeline=pipe_pca, params=param_pca,
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
#     SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Mouth')
#
#     # Throat data
#     spectra = deepcopy(spectra_full)
#     spectra.filter_location('Gorge')
#     results = []
#     classifier = SpectraClassifier(pipeline=pipe_pca, params=param_pca,
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
#     SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Throat')
#
#     # Pathology
#     # Patient healthy
#     spectra = deepcopy(spectra_full)
#     spectra.filter_patient_label('Sain')
#     results = []
#     classifier = SpectraClassifier(pipeline=pipe_pca, params=param_pca,
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
#     SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Healthy')
#
#     # Patient precancer
#     spectra = deepcopy(spectra_full)
#     spectra.filter_patient_label('Precancer')
#     results = []
#     classifier = SpectraClassifier(pipeline=pipe_pca, params=param_pca,
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
#     SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Precancer')
#
#     # Patient cancer
#     spectra = deepcopy(spectra_full)
#     spectra.filter_patient_label('Cancer')
#     results = []
#     classifier = SpectraClassifier(pipeline=pipe_pca, params=param_pca,
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
#     SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Cancer')

# MLP Test
# clf = Pipeline([('mlp', MLPClassifier(solver='adam', hidden_layer_sizes=(100,)))])
# # All data
# spectra = deepcopy(spectra_full)
# results = []
# classifier = SpectraClassifier(pipeline=clf, params={},
#                                inner_cv=inner_cv, outer_cv=outer_cv)
# results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                    groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Global')
#
# # All data
# spectra = deepcopy(spectra_full)
# results = []
# for process in processes:
#     classifier = SpectraClassifier(pipeline=process['pipe'], params=process['params'],
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Global')
#
# # Operators
# # V1 data
# spectra = deepcopy(spectra_full)
# spectra_used.filter_device('V1')
# results = []
# for process in processes:
#     classifier = SpectraClassifier(pipeline=process['pipe'], params=process['params'],
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_V1')
#
# # V2 data
# spectra = deepcopy(spectra_full)
# spectra_used.filter_device('V2')
# results = []
# for process in processes:
#     classifier = SpectraClassifier(pipeline=process['pipe'], params=process['params'],
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_V2')
#
# # Locations
# # Mouth data
# spectra = deepcopy(spectra_full)
# spectra_used.filter_location('Bouche')
# results = []
# for process in processes:
#     classifier = SpectraClassifier(pipeline=process['pipe'], params=process['params'],
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Mouth')
#
# # Throat data
# spectra = deepcopy(spectra_full)
# spectra_used.filter_location('Gorge')
# results = []
# for process in processes:
#     classifier = SpectraClassifier(pipeline=process['pipe'], params=process['params'],
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Throat')
#
# # Pathology
# # Patient healthy
# spectra = deepcopy(spectra_full)
# spectra_used.filter_patient_label('Sain')
# results = []
# for process in processes:
#     classifier = SpectraClassifier(pipeline=process['pipe'], params=process['params'],
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Healthy')
#
# # Patient precancer
# spectra = deepcopy(spectra_full)
# spectra_used.filter_patient_label('Precancer')
# results = []
# for process in processes:
#     classifier = SpectraClassifier(pipeline=process['pipe'], params=process['params'],
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Precancer')
#
# # Patient cancer
# spectra = deepcopy(spectra_full)
# spectra_used.filter_patient_label('Cancer')
# results = []
# for process in processes:
#     classifier = SpectraClassifier(pipeline=process['pipe'], params=process['params'],
#                                    inner_cv=inner_cv, outer_cv=outer_cv)
#     results.append(classifier.evaluate(features=spectra.get_features(), labels=spectra.get_labels(),
#                                        groups=spectra.get_patients_names()))
# SpectrumWriter(results).write_results('Cancer', 'C:\\Users\\Romain\\Desktop\\', 'Results_Cancer')

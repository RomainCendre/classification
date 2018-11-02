from numpy import arange
from sklearn.model_selection import GroupKFold

from os.path import expanduser
from IO.otorhinolaryngology import Reader

if __name__ == "__main__":

    # Load data
    home_path = expanduser("~")
    spectra_patients = Reader(';').read_table('{home}\\Data\\Neck\\Patients.csv'.format(home=home_path))
    spectra_temoins = Reader(';').read_table('{home}\\Data\\Neck\\Patients.csv'.format(home=home_path))
    spectra = spectra_patients + spectra_temoins

    print(spectra.methods())
#
# # Preprocessed spectra
# spectra_full.apply_average_filter(5)
# spectra_full.apply_scaling()
# spectra_full.change_wavelength_all(arange(start=445, stop=962, step=1))
# spectra_full.filter_label(['Sain', 'Cancer'])
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
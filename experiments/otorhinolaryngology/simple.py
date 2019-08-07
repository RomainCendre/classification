from os import startfile
from pathlib import Path

from numpy import geomspace
from numpy.ma import arange
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from experiments.processes import Process
from toolbox.core.parameters import LocalParameters, BuiltInSettings, ORLDataset
from toolbox.core.transforms import OrderedEncoder


def get_spectrum_classifier():
    pipe = Pipeline([('pca', PCA(n_components=0.99)),
                     # ('lda', LinearDiscriminantAnalysis(n_components=20)),
                     ('scale', StandardScaler()),
                     ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
    pipe.name = 'PatchClassifier'
    return pipe, {'clf__C': geomspace(0.01, 100, 5).tolist()}


def simple(spectra, output_folder):

    # Parameters
    nb_cpu = LocalParameters.get_cpu_number()
    validation = LocalParameters.get_validation()
    settings = BuiltInSettings.get_default_orl()

    # Statistics expected
    statistics = LocalParameters.get_orl_statistics()

    # Filters
    filters = LocalParameters.get_orl_filters()

    for filter_name, filter_datas, filter_groups in filters:

        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, n_jobs=nb_cpu)

        # Change filters
        inputs = spectra.copy_and_change(filter_groups)

        inputs.set_filters(filter_datas)
        inputs.set_encoders({'label': OrderedEncoder().fit(filter_datas['label']),
                             'group': LabelEncoder()})
        inputs.build_folds()

        process.evaluate_step(inputs=inputs, model=get_spectrum_classifier())
        process.end()


if __name__ == "__main__":

    # Output dir
    current_file = Path(__file__)
    output_folder = ORLDataset.get_results_location()/current_file.stem
    output_folder.mkdir(exist_ok=True)

    # Input data
    spectra = ORLDataset.spectras()
    spectra.change_wavelength(wavelength=arange(start=445, stop=962, step=1))
    spectra.apply_average_filter(size=5)
    spectra.norm_patient()
    # spectra.apply_scaling()

    # Compute data
    simple(spectra, output_folder)

    # Open result folder
    startfile(output_folder)

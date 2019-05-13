from os import makedirs, startfile

from numpy import geomspace
from numpy.ma import arange
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from os.path import exists, expanduser, normpath, basename, splitext

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from experiments.processes import Process
from toolbox.IO.datasets import Dataset, DefinedSettings
from toolbox.core.transforms import OrderedEncoder


def get_spectrum_classifier():
    pipe = Pipeline([('pca', PCA(n_components=0.99)),
                     ('lda', LinearDiscriminantAnalysis(n_components=20)),
                     ('scale', StandardScaler()),
                     ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
    pipe.name = 'PatchClassifier'
    return pipe, {'clf__C': geomspace(0.01, 100, 5).tolist()}


def simple(spectra, output_folder):

    # Parameters
    validation = StratifiedKFold(n_splits=5)
    settings = DefinedSettings.get_default_orl()

    # Filters
    filters = [('All', {'label': ['Sain', 'Precancer', 'Cancer']}, {}),
               ('NvsP', {'label': ['Sain', 'Pathology']}, {'label': (['Precancer', 'Cancer'], 'Pathology')}),
               ('MvsR', {'label': ['Rest', 'Cancer']}, {'label': (['Sain', 'Precancer'], 'Rest')})]

    # Statistics expected
    statistics = ['Sex', 'Diagnosis', 'Binary_Diagnosis', 'Area', 'Label']

    for filter_name, filter_datas, filter_groups in filters:
        inputs = spectra.copy_and_change(filter_groups)

        process = Process(output_folder=output_folder, name=filter_name, settings=settings, stats_keys=statistics)
        process.begin(inner_cv=validation, outer_cv=validation, n_jobs=2)

        # Change filters
        inputs.set_filters(filter_datas)
        inputs.set_encoders({'label': OrderedEncoder().fit(filter_datas['label']),
                             'groups': LabelEncoder()})
        process.evaluate_step(inputs=inputs, model=get_spectrum_classifier())
        process.end()

    # Open result folder
    startfile(output_folder)


if __name__ == "__main__":
    # Output dir
    filename = splitext(basename(__file__))[0]
    home_path = expanduser("~")
    name = filename
    output_folder = normpath('{home}/Results/ORL/{filename}'.format(home=home_path, filename=filename))
    if not exists(output_folder):
        makedirs(output_folder)

    # Input data
    spectra = Dataset.spectras()
    spectra.change_wavelength(wavelength=arange(start=445, stop=962, step=1))
    spectra.apply_average_filter(size=5)
    # spectra.norm_patient()
    # spectra.apply_scaling()

    simple(spectra, output_folder)

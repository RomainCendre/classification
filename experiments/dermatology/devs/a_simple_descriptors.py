import itertools
import webbrowser
from pathlib import Path
from numpy import logspace
from scipy.stats import randint as randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from experiments.processes import Process
from toolbox.core.builtin_models import Transforms
from toolbox.core.parameters import BuiltInSettings, LocalParameters, DermatologyDataset
from toolbox.core.transforms import OrderedEncoder


def get_linear_svm():
    # Add scaling step
    steps = [('scale', StandardScaler()), ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))]
    pipe = Pipeline(steps)
    pipe.name = 'LinearSVM'

    # Define parameters to validate through grid CV
    parameters = {'clf__C': logspace(-2, 3, 6).tolist()}
    return pipe, parameters


def get_cart():
    # Add scaling step
    steps = [('scale', StandardScaler()), ('clf', DecisionTreeClassifier(class_weight='balanced'))]
    pipe = Pipeline(steps)
    pipe.name = 'Cart'

    # Define parameters to validate through grid CV
    parameters = {'max_depth': [3, None],
                  'max_features': randint(1, 9),
                  'min_samples_leaf': randint(1, 9),
                  'criterion': ['gini', 'entropy']}
    return pipe, parameters


if __name__ == '__main__':
    # Parameters
    current_file = Path(__file__)
    # Input patch
    image_inputs = DermatologyDataset.images(modality='Microscopy')
    image_types = ['Patch', 'Full']
    # Folder
    output_folder = DermatologyDataset.get_results_location() / 'Manual'

    for image_type in image_types:
        inputs = image_inputs.sub_inputs({'Type': image_type})
        # Compute data
        output = output_folder / image_type
        output.mkdir(parents=True, exist_ok=True)
        manual(inputs, output)

    # Open result folder
    webbrowser.open(output_folder.as_uri())

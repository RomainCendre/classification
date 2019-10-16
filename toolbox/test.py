import sys
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from toolbox.transforms.labels import OrderedEncoder
from toolbox.transforms.signals import FilterTransform, ScaleTransform, FittingTransform, FittingTransform2

sys.path.append('/home/rcendre/classification')
from toolbox.classification.common import Tools, Folds
from toolbox.classification.parameters import ORL, Settings
from toolbox.views.common import Views, ViewsTools
from toolbox.views.signals import SignalsViews
wavelength = np.arange(start=440, stop=960, step=1)
inputs = ORL.get_spectra(wavelength)
inputs = Tools.transform(inputs, {'datum': 'Datum'}, FilterTransform(5), 'Mean')
inputs = Tools.transform(inputs, {'datum': 'Mean'}, ScaleTransform(), 'Scale')
ViewsTools.dataframe_renderer(inputs, title='Testing')
t = time.time()
fit = FittingTransform2().fit(np.array(inputs['Scale'].tolist()))
inputs = Tools.transform(inputs, {'datum': 'Scale'}, fit, 'Fit')
# do stuff
print(time.time() - t)

inputs = Folds.build_group_folds(inputs, {'datum': 'Datum', 'label_encode': 'LabelEncode', 'group': 'GroupEncode'}, 4)
simple_pca = Pipeline([('pca', PCA(n_components=0.95)),
                       ('clf', SVC(kernel='linear', class_weight='balanced', probability=True))])
grid_pca = {'clf__C': np.geomspace(0.01, 100, 5).tolist()}
inputs = Tools.evaluate(inputs, {'datum': 'Scale', 'label_encode': 'LabelEncode'}, simple_pca, 'PCA_SVM', grid=grid_pca)
Views.details(inputs, {'result': 'PCA_SVM'})
Views.report(ViewsTools.data_as(inputs, 'PCA_SVM'), {'label_encode': 'LabelEncode', 'prediction': 'PCA_SVM'}, label_encoder)

import time
from toolbox.transforms.labels import OrderedEncoder
from toolbox.classification.common import Folds, Tools
from toolbox.classification.parameters import Dermatology, Settings
from toolbox.models.builtin import Applications
from toolbox.transforms.common import PredictorTransform
from toolbox.transforms.images import DistributionImageTransform

# Inputs
statistics = Dermatology.get_statistics_keys()
inputs = Dermatology.images(modality='Microscopy')
inputs = inputs[inputs['Type']== 'Patch'].reset_index(drop=True)
# Advanced parameters
settings = Settings.get_default_dermatology()
label_encoder = OrderedEncoder().fit(['Normal', 'Benign', 'Malignant'])

# Encoder
label_encoder = OrderedEncoder().fit(['Normal', 'Benign', 'Malignant'])
inputs = Tools.transform(inputs, {'datum': 'Label'}, label_encoder, 'LabelEncode')
inputs = Folds.build_folds(inputs, {'datum': 'Datum', 'label_encode': 'LabelEncode'}, 5)
settings = Settings.get_default_dermatology()
extraction = DistributionImageTransform()
inputs = Tools.transform(inputs, {'datum': 'Datum'}, extraction, 'Distribution')

transfer = PredictorTransform(Applications.get_transfer_tuning())



# CART
cart = Pipeline([('scale', StandardScaler()), ('clf', DecisionTreeClassifier(class_weight='balanced'))])
cart.name = 'Cart'
distribution = {'clf__max_depth': [3, None],
                'clf__max_features': randint(1, 9),
                'clf__min_samples_leaf': randint(1, 9),
                'clf__criterion': ['gini', 'entropy']}
ViewTools.write(Views.histogram(inputs, {'datum': 'Data', 'label': 'Label'}, settings, mode='std'), 'C:\\Users\\Romain\\Desktop\\test.pdf')
inputs = Tools.transform(inputs, {'datum': 'Data'}, wavelet, 'DWT')
inputs = Tools.evaluate(inputs, {'datum': 'DWT', 'label_encode': 'LabelEncode'}, cart, 'CART',
                        distribution=distribution)
ViewTools.write(Views.projection(inputs, {'datum': 'DWT', 'label': 'Label'}, settings),
                'C:\\Users\\Romain\\Desktop\\test.pdf')
Views.report(inputs, {'label_encode': 'LabelEncode', 'prediction': 'CART'}, label_encoder)
ViewTools.write(Views.receiver_operator_curves(inputs, label_encoder, {'label_encode': 'LabelE', 'result': 'CART'}, settings), 'C:\\Users\\Romain\\Desktop\\test.pdf')
print(inputs)

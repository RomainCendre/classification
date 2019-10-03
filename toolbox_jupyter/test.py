import sys
import numpy as np

from toolbox.core.transforms import OrderedEncoder
from toolbox_jupyter.transforms.signals import FilterTransform

sys.path.append('/home/rcendre/classification')
from toolbox_jupyter.core.classification import Tools
from toolbox_jupyter.core.parameters import ORL, Settings
from toolbox_jupyter.views.common import ViewTools
from toolbox_jupyter.views.signals import ORLViews

inputs = ORL.get_spectra(np.arange(start=445, stop=962, step=1))
label_encoder = OrderedEncoder().fit(['Sain', 'Precancer', 'Cancer'])
inputs = Tools.transform(inputs, {'datum': 'Label'}, label_encoder, 'LabelEncode')
settings = Settings.get_default_orl()
Tools.transform(inputs, {'datum': 'Datum'}, FilterTransform(5), 'Mean')
ViewTools.write(ORLViews.variables(inputs, {'datum':'Datum', 'label_encode':'Label'}, settings), 'C:\\Users\\Romain\\Desktop\\test.pdf')

#
# from scipy.stats import randint as randint
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from toolbox.core.transforms import OrderedEncoder
# from toolbox_jupyter.core.classification import Folds, IO, Tools
# from toolbox_jupyter.core.parameters import Dermatology, Settings
# from toolbox_jupyter.views.common import ViewTools, Views
# from toolbox_jupyter.models.builtin import Applications
# from toolbox_jupyter.IO import image, dermatology
# from toolbox_jupyter.transforms.transforms import PredictorTransform
# from toolbox_jupyter.transforms.images import DWTImageTransform, FourierImageTransform, SpatialImageTransform
#
# # Inputs
# statistics = Dermatology.get_statistics_keys()
# inputs = image.Reader().scan_folder('C:\\Users\\Romain\\Data\\Skin\\Thumbnails')
#
# # Encoder
# label_encoder = OrderedEncoder().fit(['Normal', 'Benign', 'Malignant'])
# inputs = Tools.transform(inputs, {'datum': 'Label'}, label_encoder, 'LabelEncode')
# inputs = Folds.build_folds(inputs, {'datum': 'Data', 'label_encode': 'LabelEncode'}, 5)
# settings = Settings.get_default_dermatology()
# wavelet = DWTImageTransform()
# transfer = PredictorTransform(Applications.get_transfer_tuning())
#
# # CART
# cart = Pipeline([('scale', StandardScaler()), ('clf', DecisionTreeClassifier(class_weight='balanced'))])
# cart.name = 'Cart'
# distribution = {'clf__max_depth': [3, None],
#                 'clf__max_features': randint(1, 9),
#                 'clf__min_samples_leaf': randint(1, 9),
#                 'clf__criterion': ['gini', 'entropy']}
# ViewTools.write(Views.histogram(inputs, {'datum': 'Data', 'label': 'Label'}, settings, mode='std'), 'C:\\Users\\Romain\\Desktop\\test.pdf')
# inputs = Tools.transform(inputs, {'datum': 'Data'}, wavelet, 'DWT')
# inputs = Tools.evaluate(inputs, {'datum': 'DWT', 'label_encode': 'LabelEncode'}, cart, 'CART',
#                         distribution=distribution)
# ViewTools.write(Views.projection(inputs, {'datum': 'DWT', 'label': 'Label'}, settings),
#                 'C:\\Users\\Romain\\Desktop\\test.pdf')
# Views.report(inputs, {'label_encode': 'LabelEncode', 'prediction': 'CART'}, label_encoder)
# ViewTools.write(Views.receiver_operator_curves(inputs, label_encoder, {'label_encode': 'LabelE', 'result': 'CART'}, settings), 'C:\\Users\\Romain\\Desktop\\test.pdf')
print(inputs)

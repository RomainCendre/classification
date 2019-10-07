import sys
import numpy as np

from toolbox_jupyter.transforms.labels import OrderedEncoder
from toolbox_jupyter.transforms.signals import DWTTransform, FilterTransform, ScaleTransform

sys.path.append('/home/rcendre/classification')
from toolbox_jupyter.core.classification import Tools
from toolbox_jupyter.core.parameters import ORL, Settings
from toolbox_jupyter.views.common import ViewsTools
from toolbox_jupyter.views.signals import SignalsViews

inputs = ORL.get_spectra(np.arange(start=445, stop=962, step=1))
label_encoder = OrderedEncoder().fit(['Sain', 'Precancer', 'Cancer'])
inputs = Tools.transform(inputs, {'datum': 'Label'}, label_encoder, 'LabelEncode')
settings = Settings.get_default_orl()
inputs = Tools.transform(inputs, {'datum': 'Datum'}, FilterTransform(5), 'Mean')
inputs = Tools.transform(inputs, {'datum': 'Mean'}, ScaleTransform(), 'Normalize')
inputs = Tools.transform(inputs, {'datum': 'Normalize'}, DWTTransform(mode='db6', segment=80), 'DWT')
# inputs = Tools.transform(inputs, {'datum': 'Normalize'}, DWTTransform(mode='db6', segment_length=80), 'DWT')

ViewsTools.write(SignalsViews.variables(inputs, {'datum':'Datum', 'label_encode':'Label'}, settings), 'C:\\Users\\Romain\\Desktop\\test.pdf')

# from toolbox.core.transforms import OrderedEncoder
# from toolbox_jupyter.core.classification import Folds, Tools
# from toolbox_jupyter.core.parameters import Dermatology, Settings
# from toolbox_jupyter.models.builtin import Applications
# from toolbox_jupyter.IO import images
# from toolbox_jupyter.transforms.common import PredictorTransform
# from toolbox_jupyter.transforms.images import DWTImageTransform, DistributionImageTransform
#
# # Inputs
# statistics = Dermatology.get_statistics_keys()
# inputs = images.Reader().scan_folder('C:\\Users\\Romain\\Data\\Skin\\Thumbnails')
#
# # Encoder
# label_encoder = OrderedEncoder().fit(['Normal', 'Benign', 'Malignant'])
# inputs = Tools.transform(inputs, {'datum': 'Label'}, label_encoder, 'LabelEncode')
# inputs = Folds.build_folds(inputs, {'datum': 'Datum', 'label_encode': 'LabelEncode'}, 5)
# settings = Settings.get_default_dermatology()
# extraction = DistributionImageTransform()
# inputs = Tools.transform(inputs, {'datum': 'Datum'}, extraction, 'Distribution')
#
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

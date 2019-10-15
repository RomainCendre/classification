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

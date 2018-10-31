from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
#from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
from vis.utils import utils
import numpy as np
from keras import activations
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from vis.visualization import visualize_cam,overlay

img = utils.load_img('C:\\Users\\Romain\\Documents\\Notebook\\resnet\\images\\ouzel1.jpg')
preprocess_img = preprocess_input(img.astype('float64'))
preprocess_img = np.expand_dims(preprocess_img, axis=0)
input_shape = img.shape
#build the inceptionv3 model with imagenet weights
model = InceptionV3(weights='imagenet',include_top=True)#,input_shape = input_shape)

predictions = model.predict(preprocess_img)
prediction = np.argmax(predictions)
print(prediction)

# Utility to search for layer index by name
layer_idx = -1 #utils.find_layer_idx(model,'predictions') consider that last layer is prediction layer

#swap with softmax with linear classifier for the reasons mentioned above
# model.layers[layer_idx].activation = activations.linear
# model = utils.apply_modifications(model)


heatmap = visualize_cam(model, layer_idx, filter_indices= prediction,#20, #20 for ouzel and 292 for tiger
                                seed_input=preprocess_img, backprop_modifier=None )

jet_heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)
plt.imshow(overlay(img, jet_heatmap))
plt.show()
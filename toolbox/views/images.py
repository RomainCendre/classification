import skimage
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from matplotlib import pyplot
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import skimage as sk
from sklearn.preprocessing import normalize
from vis.utils import utils
# from vis.visualization import overlay, visualize_cam, visualize_saliency


class ImagesViews:

    @staticmethod
    def activation_map_deep(inputs, tags, network, layer, modifier=None, index=0):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in tags.keys() for elem in mandatory):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Layer access
        layer_idx = -1  # Use last instead utils.find_layer_idx(network, 'fc1000')
        penultimate_layer = utils.find_layer_idx(network, layer)

        # Read image
        data = inputs.loc[index, tags['datum']]
        label = inputs.loc[index, tags['label_encode']]
        image = utils.load_img(data)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        # Activation map
        plt.figure()
        grads = visualize_cam(network, layer_idx, filter_indices=label,
                              seed_input=image, penultimate_layer_idx=penultimate_layer,
                              backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        plt.imshow(overlay(jet_heatmap, image))
        plt.axis('off')
        plt.show()

    @staticmethod
    def activation_map_svm(inputs, tags, extractor, model, index=0):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in tags.keys() for elem in mandatory):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Coeffcients
        coefficients = model.coef_

        # model suppose to be svm
        data = inputs.loc[index, tags['datum']]
        image = utils.load_img(data)
        features = extractor.transform([data])

        labels = inputs.loc[index, tags['label_encode']]
        heatmap = np.squeeze((coefficients[labels] * features), axis=0)
        heatmap = np.resize(np.squeeze(normalize(np.expand_dims(heatmap.ravel(), axis=0), norm='max'), axis=0),
                               heatmap.shape)
        heatmap = np.sum(heatmap, axis=2)
        return ImagesViews.__add(np.array(image), heatmap)

    @staticmethod
    def saliency_map_deep(inputs, tags, network, modifier=None, index=0):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in tags.keys() for elem in mandatory):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Layer access
        layer_idx = -1  # Use last instead utils.find_layer_idx(network, 'fc1000')

        # Read image
        data = inputs.loc[index, tags['datum']]
        label = inputs.loc[index, tags['label_encode']]
        image = utils.load_img(data)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        # Saliency map
        plt.figure()
        grads = visualize_saliency(network, layer_idx, filter_indices=label,
                                   seed_input=image, backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        plt.imshow(grads, cmap='jet')
        plt.axis('off')
        plt.show()

    @staticmethod
    def histogram(inputs, tags, settings, mode='default'):
        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in tags.keys() for elem in mandatory):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Inputs
        histograms = np.array(inputs[tags['datum']].apply(ImagesViews.__get_histogram).tolist())
        labels = np.array(inputs[tags['label']].tolist())
        ulabels = np.unique(labels)
        bins = np.arange(histograms.shape[1])

        # Now browse right histograms
        figure, axe = pyplot.subplots()
        for label in ulabels:
            if mode == 'default':
                pyplot.bar(bins, np.mean(histograms[labels == label, :], axis=0).astype('int'),
                           color=np.expand_dims(np.array(settings.get_color(label)), axis=0),
                           alpha=0.5, label=label)
            else:
                pyplot.plot(bins, np.mean(histograms[labels == label, :], axis=0).astype('int'),
                            color=settings.get_color(label), label=label)
                pyplot.fill_between(bins, histograms[labels == label].mean(axis=0) - histograms.std(axis=0),
                                    histograms.mean(axis=0) + histograms.std(axis=0), color=settings.get_color(label), alpha=0.1)
        axe.set(xlabel='Intensities', ylabel='Occurrences', title='Histogram')
        axe.legend(loc='lower right')
        return figure

    @staticmethod
    def __add(image, heat_map, alpha=0.6, cmap='viridis'):

        height = image.shape[0]
        width = image.shape[1]

        # resize heat map
        heat_map_resized = sk.transform.resize(heat_map, (height, width))

        # normalize heat map
        max_value = np.max(heat_map_resized)
        min_value = np.min(heat_map_resized)
        normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

        # display
        plt.imshow(image)
        plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
        plt.axis('off')
        plt.show()

    @staticmethod
    def __get_histogram(x):
        return skimage.exposure.histogram(skimage.io.imread(x).flatten(), source_range='dtype')[0]


class PatchViews:

    @staticmethod
    def display(inputs, prediction_tag, settings, encode, index=0):
        # Full
        fulls = inputs[inputs.Type == 'Full']
        data = fulls.iloc[index]

        # Instances
        instances = inputs[inputs.Type == 'Instance']
        instances = instances[instances.Source == data.loc['Reference']]

        # Colorize image
        # image = utils.load_img(data.loc['Datum'])
        original = Image.open(data.loc['Datum']).convert('RGBA')
        patch = Image.open(data.loc['Datum']).convert('RGBA')
        draw = ImageDraw.Draw(patch)
        for index, row in instances.iterrows():
            center_x = row['Center_X']
            center_y = row['Center_Y']
            height = row['Height']
            width = row['Width']
            start = (center_x-width/2+1, center_y-height/2+1)
            end = (center_x+width/2+1, center_y+height/2+1)

            pred = row['Prediction']
            prediction = row[prediction_tag]
            if isinstance(prediction, np.ndarray) and len(prediction) != 1:
                temp = prediction.argmax()
                intensity = prediction[temp]
                prediction = temp
            else:
                prediction = int(prediction)
                intensity = 0.5
            color = settings.get_color(encode.inverse_transform(np.array([prediction]))[0]) + (intensity,)  # Add alpha
            color = tuple(np.multiply(color, 255).astype(int))
            draw.rectangle((start, end), fill=color)

        # display
        figure = plt.figure()
        plt.imshow(Image.alpha_composite(original, patch).convert("RGB"))
        plt.axis('off')
        # plt.show()
        return figure

import cv2
import numpy as np
import tensorflow as tf
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
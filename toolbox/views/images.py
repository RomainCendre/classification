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
from vis.visualization import overlay, visualize_cam, visualize_saliency


class ImagesViews:

    @staticmethod
    def svm_activation_map(inputs, tags, extractor, model, index=0):
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
    def deep_activation_map(inputs, tags, network, layer, modifier=None, index=0):
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
        if len(image) != 3:
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
    def deep_saliency_map(inputs, tags, network, layer, modifier=None, index=0):
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
        if len(image) != 3:
            image = np.stack((image,) * 3, axis=-1)

        # Saliency map
        plt.figure()
        # 20 is the imagenet index corresponding to `ouzel`
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


class Patchs:

    def __init__(self, inputs, settings):
        self.inputs = inputs
        self.settings = settings

    def write_patch(self, output_folder):
        # Check output folder
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        output_folder = output_folder / self.inputs.name
        output_folder.mkdir(exist_ok=True)

        references = list(set(self.inputs.get_from_key('Reference')))

        for index, reference in enumerate(references):
            work_input = self.inputs.sub_inputs({'Reference': [reference]})
            path = list(set(work_input.get_from_key('Full_path')))
            label = list(set(work_input.get_from_key('Label')))
            image = Image.open(path[0]).convert('RGBA')
            for sub_index, entity in work_input.data.iterrows():
                start = entity['Patch_Start']
                end = entity['Patch_End']
                center = ((end[0] + start[0]) / 2, (end[1] + start[1]) / 2)
                center = tuple(np.subtract(center, 10)), tuple(np.add(center, 10))
                predict = entity['PredictorTransform']
                color = self.settings.get_color(self.inputs.decode('label', predict)) + (0.5,)  # Add alpha
                color = tuple(np.multiply(color, 255).astype(int))
                draw = ImageDraw.Draw(image)
                draw.rectangle(center, fill=color)
                # draw.rectangle((start, end), outline="white")
            image.save(output_folder / '{ref}_{lab}.png'.format(ref=reference, lab=label[0]))

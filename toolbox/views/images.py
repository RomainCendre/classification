import skimage
import numpy as np
from pathlib import Path
from matplotlib import pyplot
from PIL import Image, ImageDraw


class ImagesViews:

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
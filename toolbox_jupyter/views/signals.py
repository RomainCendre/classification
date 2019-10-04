import skimage
import numpy as np
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest, chi2


class SignalsViews:

    @staticmethod
    def histogram(inputs, tags, settings, mode='default'):
        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in tags.keys() for elem in mandatory):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Inputs
        histograms = np.array(inputs[tags['datum']].apply(SignalsViews.__get_histogram).tolist())
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
    def mean_and_deviation(inputs, tags, settings, title='Mean and Deviation'):
        # Check mandatory fields
        mandatory = ['datum', 'wavelength', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        data = np.array(inputs[tags['datum']].tolist())
        wavelength = np.array(inputs[tags['wavelength']].tolist())[0]
        labels = inputs[tags['label']]
        unique_labels = np.unique(labels)

        figure, axe = pyplot.subplots()
        for label in unique_labels:
            axe.plot(wavelength, data[labels == label].mean(axis=0), alpha=1, color=settings.get_color(label), label=label, linewidth=1.0)
            axe.fill_between(wavelength, data[labels == label].mean(axis=0) - data.std(axis=0),
                            data.mean(axis=0) + data.std(axis=0), color=settings.get_color(label), alpha=0.1)
        # Now set title and legend
        axe.set(xlabel=tags['wavelength'],
                ylabel=tags['datum'],
                title=title)
        axe.legend(loc='lower right')
        return figure

    @staticmethod
    def variables(inputs, tags, title='Mean and Deviation'):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Inputs
        data = np.array(inputs[tags['datum']].tolist())
        labels = inputs[tags['label_encode']]

        # Compute p_values
        kbest = SelectKBest(chi2, k='all').fit(data, labels)

        figure, axe = pyplot.subplots()
        # Now set title and legend
        axe.set(xlabel=tags['wavelength'],
                ylabel=tags['datum'],
                title=title)
        axe.legend(loc='lower right')
        return figure

    @staticmethod
    def __get_histogram(x):
        return skimage.exposure.histogram(x, source_range='dtype')[0]

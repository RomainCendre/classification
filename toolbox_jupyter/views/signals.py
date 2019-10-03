from matplotlib import pyplot
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


class ORLViews:

    @staticmethod
    def variables(inputs, tags, title='Mean and Deviation'):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        data = np.array(inputs[tags['datum']].tolist())
        labels = inputs[tags['label_encode']]

        pvalues = SelectKBest(chi2, k='all').fit_transform(data, labels)
        figure, axe = pyplot.subplots(figsize=(21, 7))

        # Now set title and legend
        axe.set(xlabel=tags['wavelength'],
                ylabel=tags['datum'],
                title=title)
        axe.legend(loc='lower right')

        colormap = pyplot.cm.viridis
        pyplot.title('The Correlation', y=1.05, size=15)
        sns.heatmap(Correlation.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap="YlGnBu",
                    linecolor='white', annot=True)
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

        figure, axe = pyplot.subplots(figsize=(21, 7))
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

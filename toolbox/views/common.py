from pathlib import Path

import pandas
import numpy as np
from collections import Counter
from jinja2 import Environment, ChoiceLoader, FileSystemLoader
from matplotlib.colors import ListedColormap
from pandas.io.formats.style import Styler
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve, classification_report
from toolbox.classification.common import Tools


class Views:

    @staticmethod
    def details(inputs, tags):
        # Fold needed for evaluation
        if 'Fold' not in inputs:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['eval']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Tags
        features = f'{tags["eval"]}_{Tools.FEATURES}'
        parameters = f'{tags["eval"]}_{Tools.PARAMETERS}'

        unique_folds = np.unique(inputs['Fold'])
        data = {'Fold': [], 'Features': [], 'Parameters': []}
        for fold in unique_folds:
            data['Fold'].append(fold)
            data['Features'].append(int(inputs.at[0, f'{features}_{fold}']))
            data['Parameters'].append(inputs.at[0, f'{parameters}_{fold}'])
        return pandas.DataFrame(data)

    @staticmethod
    def fold_visualization(inputs, tags, encoder, settings, lw=20, sub_encoder=None):
        # Fold needed for evaluation
        if 'Fold' not in inputs:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['label_encode', 'group_encode']
        # if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
        #     raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        folds = np.array(inputs['Fold'].tolist())
        if 'sub_encode' in tags:
            sub = np.array(inputs[tags['sub_encode']].tolist())
        labels = np.array(inputs[tags['label_encode']].tolist())
        groups = np.array(inputs[tags['group_encode']].tolist())

        figure, axe = pyplot.subplots()
        unique_folds = np.unique(folds)
        # Generate the training/testing visualizations for each CV split
        for index, fold in enumerate(unique_folds):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(folds))
            indices[folds == fold] = 1
            indices[folds != fold] = 0

            # Visualize the results
            axe.scatter(range(len(indices)), [index + .5] * len(indices),
                        c=indices, marker='_', lw=lw, cmap=pyplot.cm.coolwarm,
                        vmin=-.2, vmax=1.2)

        nb_folds = len(unique_folds)

        elements = []
        # Plot the data classes and groups at the end
        if 'sub_encode' in tags:
            axe.scatter(range(len(sub)), [nb_folds + .5 + len(elements)] * len(sub),
                        c=sub, marker='_', lw=lw, cmap=Views.__get_color_map(sub, sub_encoder, settings))
            elements.append('sub')


        # settings.get_color(positive_class)
        axe.scatter(range(len(labels)), [nb_folds + .5 + len(elements)] * len(labels),
                    c=labels, marker='_', lw=lw, cmap=Views.__get_color_map(labels, encoder, settings))
        elements.append('class')

        axe.scatter(range(len(groups)), [nb_folds + .5 + len(elements)] * len(groups),
                    c=groups%9, marker='_', lw=lw, cmap=pyplot.cm.Paired)
        elements.append('group')

        # Formatting
        yticklabels = list(range(1, nb_folds+1)) + elements

        axe.set(yticks=np.arange(nb_folds + len(elements)) + .5,
                yticklabels=yticklabels,
                xlabel='Sample index',
                ylabel="CV iteration",
                ylim=[nb_folds + len(elements) + .2, -.2])
        axe.set_title('Folds', fontsize=15)
        figure.show()
        return figure

    @staticmethod
    def misclassified(inputs, tags):
        # Check mandatory fields
        mandatory = ['datum', 'label_encode', 'result']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Prediction tag
        tag_pred = f'{tags["result"]}_{Tools.PREDICTION}'

        # Mask
        mask = (inputs[tags['label_encode']] == inputs[tag_pred])
        inputs = inputs[mask]
        data = {'datum': inputs[tags['datum']],
                'labels': inputs[tags['label_encode']],
                'predictions': inputs[tag_pred]}
        return pandas.DataFrame(data)

    @staticmethod
    def projection(inputs, tags, settings, mode='PCA', name=None):
        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in tags.keys() for elem in mandatory):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Inputs
        labels = inputs[tags['label']]
        ulabels = np.unique(labels)

        # Compute
        if mode == 'PCA':
            method = PCA(n_components=2, whiten=True)  # project to 2 dimensions
        else:
            method = TSNE(n_components=2)

        projected = method.fit_transform(np.array(inputs[tags['datum']].tolist()))
        figure = pyplot.figure()
        for label in ulabels:
            pyplot.scatter(projected[labels == label, 0], projected[labels == label, 1],
                           c=np.expand_dims(np.array(settings.get_color(label)), axis=0),
                           alpha=0.5, label=label, edgecolor='none')
        pyplot.axis('off')
        pyplot.legend(loc='lower right')
        if name:
            figure.suptitle(name)
        else:
            figure.suptitle(tags['datum'])
        return figure

    @staticmethod
    def receiver_operator_curves(inputs, encoder, tags, settings, name=None):
        # Check mandatory fields
        mandatory = ['label_encode', 'eval']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Data
        labels = np.array(inputs[tags['label_encode']].to_list())
        unique = np.unique(inputs[tags['label_encode']])
        probabilities = np.array(inputs[f'{tags["eval"]}_{Tools.PROBABILITY}'].to_list())

        # Check if Nan values
        if np.isnan(probabilities).any():
            raise Exception(f'Unexpected values (NaN) found in probabilities.')

        figure, axe = pyplot.subplots()
        # Plot luck
        axe.plot([0, 1], [0, 1], linestyle='--', lw=2, color=settings.get_color('Luck'), label='Luck', alpha=.8)
        title = ''

        # Browse each label
        for positive_index in unique:
            positive_class = encoder.inverse_transform(positive_index)[0]
            fpr, tpr, threshold = roc_curve(labels, probabilities[:, positive_index], pos_label=positive_index)
            axe.plot(fpr, tpr, lw=2, alpha=.8, color=settings.get_color(positive_class),
                     label='ROC {label} (AUC = {auc:.2f})'.format(label=positive_class, auc=auc(fpr, tpr),
                                                                  **settings.get_line(positive_class)))
        # Switch depend on the mode of display
        title = f'{tags["eval"]} - Receiver operating characteristic'
        # Now set title and legend
        axe.set(adjustable='box',
                aspect='equal',
                xlabel='False Positive Rate (1-Specificity)',
                ylabel='True Positive Rate (Sensitivity)',
                title=title)
        axe.legend(loc='lower right')  # If better than random, no curve is display on bottom right part
        if name:
            figure.suptitle(name)
        return figure

    @staticmethod
    def report(inputs, tags, encode):
        # Fold needed for evaluation
        if 'Fold' not in inputs:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['label_encode', 'eval']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Inputs
        labels = np.array(inputs[tags['label_encode']].tolist())
        predictions = np.array(inputs[f'{tags["eval"]}_{Tools.PREDICTION}'].tolist())
        folds = np.array(inputs['Fold'].tolist())

        # Mean score
        report = pandas.DataFrame(
            classification_report(labels, predictions, output_dict=True, target_names=encode.map_list)).transpose()
        # Std score
        scores = []
        for fold in np.unique(folds):
            # Create mask
            mask = folds == fold
            scores.append(
                classification_report(labels[mask], predictions[mask], output_dict=True, target_names=encode.map_list))
        report = report.apply(lambda x: pandas.DataFrame(x).apply(lambda y: Views.__format_std(x, y, scores), axis=1))
        return report

    @staticmethod
    def statistics(inputs, keys, name=None):
        figure, axes = pyplot.subplots(ncols=len(keys))
        # Browse each kind of parameter
        for index, key in enumerate(keys):
            axes[index].set_title(key)
            axes[index].axis('off')
            if key not in inputs.columns:
                print('Key {key} is missing from data.'.format(key=key))
                continue

            counter = Counter(list(inputs[key]))
            axes[index].pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
            axes[index].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        figure.suptitle(f'Samples {len(inputs)}')
        if name:
            figure.suptitle(name)
        return figure

    @staticmethod
    def __get_color_map(values, encoder, settings):
        unique_values = np.unique(values)
        colors = []
        for value in unique_values:
            colors.append(settings.get_color(encoder.inverse_transform(value)[0]))
        return ListedColormap(colors)

    @staticmethod
    def __format_std(x, y, scores):
        std = np.std([score[y.name][x.name] for score in scores])
        return f'{y[x.name]:0.2f}±{std:0.2f}'


class ViewsTools:
    class DataframeStyler(Styler):
        env = Environment(loader=ChoiceLoader([
            FileSystemLoader(searchpath=str(Path(__file__).parent / 'templates')),
            Styler.loader,  # the default
        ]))
        template = env.get_template('dataframe.tpl')

    @staticmethod
    def data_as(inputs, out_tag, as_train=False):
        # Fold needed for evaluation
        if 'Fold' not in inputs:
            raise Exception('Need to build fold.')

        # Out fields
        out_preds = f'{out_tag}_{Tools.PREDICTION}'
        out_probas = f'{out_tag}_{Tools.PROBABILITY}'
        out_features = f'{out_tag}_{Tools.FEATURES}'
        out_params = f'{out_tag}_{Tools.PARAMETERS}'

        # Folds array
        data = []
        folds = inputs['Fold']
        for fold in np.unique(folds):
            # Create mask
            mask = folds == fold
            if as_train:
                mask = ~mask

            # Manage data
            inputs.loc[mask, out_preds] = inputs.loc[mask, f'{out_preds}_{fold}']
            inputs.loc[mask, out_probas] = inputs.loc[mask, f'{out_probas}_{fold}']
            inputs.loc[mask, out_features] = inputs.loc[mask, f'{out_features}_{fold}']
            inputs.loc[mask, out_params] = inputs.loc[mask, f'{out_params}_{fold}']

            # Inputs
            data.append(inputs[mask])
        # Build dataframe and purge useless
        dataframe = pandas.concat(data)
        # dataframe.drop()
        return dataframe

    @staticmethod
    def dataframe_renderer(dataframe, title):
        if not isinstance(dataframe, list) and not isinstance(title, list):
            return ViewsTools.DataframeStyler(dataframe).render(table_title=title)

        # Check both are lists
        if not type(dataframe) == type(title) or not len(dataframe) == len(title):
            raise Exception('Types are inconsistents.')

        html = ''
        for df, tit in zip(dataframe, title):
            html += ViewsTools.DataframeStyler(df).render(table_title=tit) + '<br/>'
        return html

    @staticmethod
    def plot_size(size):
        pyplot.rcParams["figure.figsize"] = size

    @staticmethod
    def write(data, out_file):
        if isinstance(data, pandas.DataFrame):
            data.to_csv(out_file, index=True)
        else:
            save_to = PdfPages(out_file)
            save_to.savefig(data)
            save_to.close()
            pyplot.close()

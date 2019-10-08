import pandas
import numpy as np
from collections import Counter
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve, classification_report


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
        features = f'{tags["eval"]}_Features'
        parameters = f'{tags["eval"]}_Parameters'

        unique_folds = np.unique(inputs['Fold'])
        data = {'Fold': [], 'Features': [], 'Parameters': []}
        for fold in unique_folds:
            data['Fold'].append(fold)
            data['Features'].append(int(inputs.at[0, f'{features}_{fold}'][0]))
            data['Parameters'].append(inputs.at[0, f'{parameters}_{fold}'][0])
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
        probabilities = np.array(inputs[f'{tags["eval"]}_Probabilities'].to_list())

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
            title = 'Receiver operating characteristic'

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
        predictions = np.array(inputs[f'{tags["eval"]}_Predictions'].tolist())
        folds = np.array(inputs['Fold'].tolist())

        # Mean score
        report = pandas.DataFrame(classification_report(labels, predictions, output_dict=True, target_names=encode.map_list)).transpose()
        # Std score
        scores = []
        for fold in np.unique(folds):
            # Create mask
            mask = folds == fold
            scores.append(classification_report(labels[mask], predictions[mask], output_dict=True, target_names=encode.map_list))
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
    def __format_std(x, y, scores):
        std = np.std([score[y.name][x.name] for score in scores])
        return f'{y[x.name]:0.2f}±{std:0.2f}'


class ViewsTools:

    @staticmethod
    def data_as(inputs, out_tag, as_train=False):
        # Fold needed for evaluation
        if 'Fold' not in inputs:
            raise Exception('Need to build fold.')

        # Out fields
        out_preds = f'{out_tag}_Predictions'
        out_probas = f'{out_tag}_Probabilities'
        out_features = f'{out_tag}_Features'
        out_params = f'{out_tag}_Parameters'

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

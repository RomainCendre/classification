from pathlib import Path

import pandas
import numpy as np
from collections import Counter
from jinja2 import Environment, ChoiceLoader, FileSystemLoader
from matplotlib.colors import ListedColormap
from pandas.io.formats.style import Styler
from matplotlib import pyplot, transforms
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve, classification_report
from toolbox.classification.common import Tools
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class Views:
    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of `x` and `y`

        Parameters
        ----------
        x, y : array_like, shape (n, )
            Input data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        Returns
        -------
        matplotlib.patches.Ellipse

        Other parameters
        ----------------
        kwargs : `~matplotlib.patches.Patch` properties
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

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
    def group_statistics(inputs, tags):
        mandatory = ['group']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        images = []
        for group in inputs.groupby(tags['group']):
            images.append(len(group[1]))

        images = np.array(images)
        statistics = f'[{np.min(images)}, {np.max(images)}] {np.mean(images)}±{np.std(images)}'
        return statistics

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
    def projection(inputs, tags, settings, mode='PCA', name=None, legend=False):
        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in tags.keys() for elem in mandatory):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Inputs
        labels = inputs[tags['label']]
        ulabels = np.unique(labels)

        # Compute
        if mode == 'PCA':
            method = PCA(whiten=True)  # project to 2 dimensions
        else:
            method = TSNE(perplexity=5)

        projected = method.fit_transform(np.array(inputs[tags['datum']].tolist()))

        figure = pyplot.figure()
        axis = figure.add_subplot(111)
        for label in ulabels:
            color = np.expand_dims(np.array(settings.get_color(label)), axis=0)
            pyplot.scatter(projected[labels == label, 0], projected[labels == label, 1],
                           c=color, alpha=0.5, label=label, edgecolor='none')
        # Centroids & Ellipsis
        for label in ulabels:
            # Centroids
            color = np.array(settings.get_color(label))
            mean_x = np.mean(projected[labels == label, 0])
            mean_y = np.mean(projected[labels == label, 1])
            pyplot.scatter(mean_x, mean_y, marker='X', c=color,
                           s=150, linewidth=3, edgecolor='gray', label=f'Centroid {label}')
            # Ellipsis
            Views.confidence_ellipse(projected[labels == label, 0], projected[labels == label, 1], axis,
                                      edgecolor=color, linewidth=3, zorder=0)

        pyplot.axis('off')
        if legend:
            pyplot.legend(loc='lower right')

        if name:
            title = name
        else:
            title = tags['datum']

        if mode == 'PCA':
            variance_explained = method.explained_variance_ratio_.cumsum()[1]
            title = f'{title} (Variance : {variance_explained:.1%})'

        figure.suptitle(title)
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
    def reliability_curve(inputs, label_tag, names):
        figure = pyplot.figure(figsize=(10, 10))
        axis_calib = pyplot.subplot2grid((3, 1), (0, 0), rowspan=2)
        axis_count = pyplot.subplot2grid((3, 1), (2, 0))
        axis_calib.plot([0, 1], [0, 1], "k:", label="Calibration parfaite")

        if not isinstance(names, list):
            names = [names]

        colors = ['#003f5c', '#58508d', '#bc5090']
        folds = inputs['Fold'].unique()

        mean_interp = np.linspace(0, 1, 100)
        for index, name in enumerate(names):
            interp_fops = []
            briers = []
            color = colors[index]
            for fold in folds:
                sub = inputs[inputs['Fold'] == fold]
                y_pred = np.array(sub[f'{name}_Prediction'].to_list())
                y_prob = np.array(sub[f'{name}_Probability'].to_list())
                y_pos_prob = y_prob[:, int(y_pred.max())]
                label = np.array(sub[label_tag].to_list())

                fraction_of_positives, mean_predicted_value = calibration_curve(label, y_pos_prob, n_bins=10)
                interp_fops.append(np.interp(mean_interp, mean_predicted_value, fraction_of_positives))
                briers.append(brier_score_loss(label, y_pos_prob, pos_label=y_pred.max()))

            # Compute mean and std
            interp_fops = np.array(interp_fops)
            mean_fop = np.mean(interp_fops, axis=0)
            std_fop = np.std(interp_fops, axis=0)
            briers = np.array(briers)
            mean_brier = np.mean(briers)
            std_brier = np.std(briers)
            axis_calib.plot(mean_interp, mean_fop, color=color, label=f"{name} (Brier {mean_brier:.2f}±{std_brier:.2f})")

            tprs_upper = np.minimum(mean_interp + std_fop, 1)
            tprs_lower = np.maximum(mean_interp - std_fop, 0)
            axis_calib.fill_between(mean_interp, tprs_lower, tprs_upper, color=color, alpha=.2, label='_Hidden')
            axis_count.hist(y_pos_prob, color=color, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

        axis_calib.set_ylabel("Fraction of positives")
        axis_calib.set_ylim([-0.05, 1.05])
        axis_calib.legend(loc="lower right")
        axis_calib.set_title('Calibration plots  (reliability curve)')

        axis_count.set_xlabel("Mean predicted value")
        axis_count.set_ylabel("Count")
        axis_count.legend(loc="upper center", ncol=2)

        pyplot.tight_layout()
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
    def steps_visualization(inputs, tags):
        # Fold needed for evaluation
        if 'Fold' not in inputs:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['label_encode', 'eval']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Find data
        folds = np.array(inputs[f'Fold'].to_list()).astype(int)
        steps = np.array(inputs[f'{tags["eval"]}_{Tools.STEPS}'].to_list()).astype(int)
        labels = np.array(inputs[tags["label_encode"]].to_list()).astype(int)
        predictions = np.array(inputs[f'{tags["eval"]}_{Tools.PREDICTION}'].to_list()).astype(int)
        # Process parameters
        parameters = np.array(inputs[f'{tags["eval"]}_{Tools.PARAMETERS}'].to_list())
        parameters = np.array([parameters[np.where(folds == fold)[0][0]] for fold in np.unique(folds)])
        parameters = np.moveaxis(parameters, 0, (len(parameters.shape) - 1))
        means = np.mean(parameters, axis=(len(parameters.shape)-1))
        stds = np.std(parameters, axis=(len(parameters.shape)-1))
        well_classified = labels == predictions

        # Compute data
        remain = len(predictions)
        data = []
        for current in np.unique(steps):
            current_prediction = predictions[steps == current]
            current_well = well_classified[steps == current]
            current_miss = ~current_well
            remain = remain - len(current_prediction)

            benign_mask = current_prediction == 0
            malignant_mask = current_prediction == 1

            data.append(np.array([len(np.where(current_well[benign_mask])[0]), # Benign
                                  len(np.where(current_miss[benign_mask])[0]), # False benign
                                  remain, # Remains
                                  len(np.where(current_miss[malignant_mask])[0]),
                                  len(np.where(current_well[malignant_mask])[0])]))

        dataframe = pandas.DataFrame(data, columns=['Benign', 'FBenign', 'Remain', 'FMalignant', 'Malignant'])

        return Views.__get_custom_horizontal_plot(dataframe, len(predictions), means, stds)

    @staticmethod
    def __get_color_map(values, encoder, settings):
        unique_values = np.unique(values)
        colors = []
        for value in unique_values:
            colors.append(settings.get_color(encoder.inverse_transform(value)[0]))
        return ListedColormap(colors)

    @staticmethod
    def __get_custom_horizontal_plot(dataframe, total, means, stds):

        def plot_rect(bottom, left, width, height, color='C0'):
            ax.add_patch(patches.Rectangle(
                (left, bottom), width, height, linewidth=1, edgecolor=color, facecolor=color))

        def process_data(data):
            left = np.zeros_like(data)
            mid = data[2] / 2
            left[0] = -np.sum(data[0:1]) - mid
            left[1] = -np.sum(data[1]) - mid
            left[2] = -mid
            left[3] = mid
            left[4] = np.sum(data[3]) + mid
            width = data
            return left, width

        # Create figure and axes
        fig, ax = pyplot.subplots(1)

        # Change color of plot edges
        ax.spines['left'].set_color('lightgray')
        ax.spines['right'].set_color('lightgray')
        ax.spines['top'].set_color('lightgray')

        # Hide y axis ticks
        pyplot.gca().tick_params(axis='y', colors='w')

        # Turn on gridlines and set color
        pyplot.grid(b=True, axis='both', color='lightgray', alpha=0.5, linewidth=1.5)

        # Add lines
        pyplot.axvline(x=0, c='lightgray')

        # Add x label
        pyplot.xlabel('Pourcentage', fontsize=14)

        # Define color scheme from negative to positive
        colors = ['forestgreen', 'palegreen', 'lightgray', 'salmon', 'firebrick']

        # Process data to plot
        rows = len(dataframe)
        array = [dataframe.iloc[i, :].values for i in reversed(np.arange(rows))]
        array_percentage = [dataframe.iloc[i, :].values / total for i in reversed(np.arange(rows))]

        num_modalities = len(array)

        # Define axis limits
        pyplot.xlim(-1.125, 1.125)
        pyplot.ylim(0.05, 0.2 * (num_modalities-1))

        # Define axis ticks ticks
        pyplot.xticks(np.arange(-1, 1.25, 0.25), np.arange(-100, 125, 25))
        pyplot.yticks(np.arange(0, 0.2 * (num_modalities), 0.2), np.arange(0, 0.2 * (num_modalities), 0.2))

        # Move gridlines to the back of the plot
        pyplot.gca().set_axisbelow(True)

        left = {}
        width = {}
        for i in range(num_modalities):
            left[i], width[i] = process_data(array_percentage[i])

        # Plot boxes
        height = 0.10
        bottom = 0.05
        gap = 0.0125
        for i in range(num_modalities):
            for j in range(len(array[i])):
                plot_rect(bottom=bottom + i * 0.2, left=left[i][j], width=width[i][j], height=height, color=colors[j])

        # Plot percentages
        for i in range(num_modalities):
            pyplot.text(-1, 0.2 * (i)+gap, str(array[i][0]), horizontalalignment='left',
                        verticalalignment='center')
            pyplot.text(-0.5, 0.2 * (i)+gap, str(array[i][1]), horizontalalignment='center',
                        verticalalignment='center')
            pyplot.text(0, 0.2 * (i)+gap, str(array[i][2]), horizontalalignment='center',
                        verticalalignment='center')
            pyplot.text(0.5, 0.2 * (i)+gap, str(array[i][3]), horizontalalignment='right',
                        verticalalignment='center')
            pyplot.text(1, 0.2 * (i)+gap, str(array[i][4]), horizontalalignment='right',
                        verticalalignment='center')

        # Plot threshs
        for i in range(num_modalities):
            rev_index = num_modalities-i-1
            if len(means.shape) == 1:
                pyplot.text(-0.15, 0.2 * (i) + gap + 1.5*height, f'{means[rev_index]:.2f}±{stds[rev_index]:.2f}', horizontalalignment='left',
                            verticalalignment='center')
            else:
                pyplot.text(-0.5, 0.2 * (i) + gap +  1.5*height, f'{means[rev_index,0]:.2f}±{stds[rev_index,0]:.2f}', horizontalalignment='left',
                            verticalalignment='center')
                pyplot.text(0.5, 0.2 * (i) + gap +  1.5*height, f'{means[rev_index,1]:.2f}±{stds[rev_index,1]:.2f}', horizontalalignment='left',
                            verticalalignment='center')

        # Plot category labels
        pyplot.text(-1.1, 0.2 * (num_modalities)+gap, 'Bénin', style='italic', horizontalalignment='left',
                    verticalalignment='center')
        # pyplot.text(-0.6, 0.2 * (num_modalities)+gap, 'Faux Bénin', style='italic', horizontalalignment='left',
        #             verticalalignment='center')
        pyplot.text(0, 0.2 * (num_modalities)+gap, 'Restant', style='italic', horizontalalignment='center',
                    verticalalignment='center')
        # pyplot.text(0.6, 0.2 * (num_modalities)+gap, 'Faux Malin', style='italic', horizontalalignment='right',
        #             verticalalignment='center')
        pyplot.text(1.1, 0.2 * (num_modalities)+gap, 'Malin', style='italic', horizontalalignment='right',
                    verticalalignment='center')

        return fig

    @staticmethod
    def __format_std(x, y, scores):
        try:
            std = np.std([score[y.name][x.name] for score in scores])
        # Manage case of accuracy, no second property
        except:
            std = np.std([score[y.name] for score in scores])
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

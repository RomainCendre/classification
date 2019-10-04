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
        mandatory = ['label_encode', 'result']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Data
        labels = np.array(inputs[tags['label_encode']].to_list())
        unique = np.unique(inputs[tags['label_encode']])
        probabilities = np.array(inputs[f'{tags["result"]}_Probabilities'].to_list())

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
    def report(inputs, tags, encode, is_train_mode=False):
        # Fold needed for evaluation
        if 'Fold' not in inputs:
            raise Exception('Need to build fold.')

        # Check mandatory fields
        mandatory = ['label_encode', 'prediction']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Prediction tag
        tag_pred = f'{tags["prediction"]}_Predictions'

        # Folds array
        folds = inputs['Fold']
        predictions = None
        scores = []
        for fold in np.unique(folds):
            # Create mask
            mask = folds == fold
            if is_train_mode:
                mask = ~mask

            # Inputs
            data = inputs[mask]
            data = np.array([data[tags['label_encode']], data[f'{tag_pred}_{fold}']]).transpose()

            # Remind data for mean
            if predictions is None:
                predictions = data
            else:
                predictions = np.concatenate((predictions, data))

            # Scores fold
            scores.append(classification_report(data[:, 0], data[:, 1], output_dict=True, target_names=encode.map_list))

        # Mean score
        report = pandas.DataFrame(classification_report(predictions[:, 0], predictions[:, 1],
                                                        output_dict=True, target_names=encode.map_list)).transpose()
        return report.apply(lambda x: pandas.DataFrame(x).apply(lambda y: Views.__format_std(x, y, scores), axis=1))

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

    def __parameters(self, result):
        unique_folds = result.get_unique_from_key('Fold')
        params = []
        for fold in unique_folds:
            filter_by = {'Fold': [fold]}
            best_params = str(result.get_from_key(key='BestParams', filters=filter_by)[0])
            features_number = str(result.get_from_key(key='FeaturesNumber', filters=filter_by)[0])
            params.append((best_params, features_number))
        return params


class ViewsTools:

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

# class VisualizationWriter:
#
#     def __init__(self, model, preprocess=None):
#         self.model = model
#         self.preprocess = preprocess
#
#     def __get_activation_map(self, seed_input, predict, image):
#         if image.ndim == 2:
#             image = repeat(image[:, :, newaxis], 3, axis=2)
#
#         grads = visualize_cam(self.model, len(self.model.layers) - 1, filter_indices=predict, seed_input=seed_input,
#                               backprop_modifier='guided')
#
#         jet_heatmap = uint8(cm.jet(grads)[..., :3] * 255)
#         return overlay(jet_heatmap, image)
#
#     def write_activations_maps(self, inputs, output_folder):
#
#         # Activation dir
#         activation_dir = join(output_folder, 'Activation/')
#         if not exists(activation_dir):
#             makedirs(activation_dir)
#
#         # Check for model type, will be changed in future
#         if not isinstance(self.model, KerasBatchClassifier):
#             return
#
#         # Extract data for fit
#         paths = inputs.get_datas()
#         labels = inputs.get_labels()
#         ulabels = inputs.get_unique_labels()
#
#         # Prepare data
#         generator = ResourcesGenerator(preprocessing_function=self.preprocess)
#         valid_generator = generator.flow_from_paths(paths, labels, batch_size=1, shuffle=False)
#
#         # Folds storage
#         for index in arange(len(valid_generator)):
#             x, y = valid_generator[index]
#
#             for label_index in ulabels:
#                 dir_path = join(activation_dir, inputs.decode('label', label_index))
#                 if not exists(dir_path):
#                     makedirs(dir_path)
#
#                 file_path = join(dir_path, '{number}.png'.format(number=index))
#
#                 try:
#                     activation = self.__get_activation_map(seed_input=x, predict=label_index,
#                                                            image=load_img(paths[index]))
#                     imsave(file_path, activation)
#                 except:
#                     print('Incompatible model or trouble occurred.')

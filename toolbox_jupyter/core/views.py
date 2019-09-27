import markups
import pickle
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, classification_report


class ViewTools:

    @staticmethod
    def write(plot, out_file):
        save_to = PdfPages(out_file)
        save_to.savefig(plot)
        save_to.close()
        pyplot.close()


class Views:

    @staticmethod
    def statistics(inputs, keys, name=None):
        figure, axes = pyplot.subplots(ncols=len(keys), figsize=(21, 7))
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
    def pca_projection(inputs, tags, settings, name=None):
        # Check mandatory fields
        mandatory = ['datum', 'label']
        if not isinstance(tags, dict) or not all(elem in mandatory for elem in tags.keys()):
            raise Exception(f'Expected tags: {mandatory}, but found: {tags}.')

        # Inputs
        labels = inputs.get('label', encode=False)
        ulabels = np.unique(labels)

        # Compute PCA
        pca = PCA(n_components=2, whiten=True)  # project to 2 dimensions
        projected = pca.fit_transform(inputs.get('datum'))
        figure = pyplot.figure()
        for label in ulabels:
            pyplot.scatter(projected[labels == label, 0], projected[labels == label, 1],
                           c=np.array(settings.get_color(label)), alpha=0.5, label=label, edgecolor='none')
        pyplot.axis('off')
        pyplot.legend(loc='lower right')
        if name:
            figure.suptitle(name)
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

        figure, axe = pyplot.subplots(ncols=1, figsize=(10, 10))
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
    def write_report(self, use_std=True, path=None):
        report = ''
        for result in self.results:
            if not result.is_valid_keys(['Label', 'Prediction']):
                print('Missing tag for global report.')
                return
            # Initialize converter of markup
            report += self.report_scores(result, use_std)+'\n\n'

        # Write to html way
        markup = markups.TextileMarkup()
        mk_report = markup.convert(report)
        if path is None:
            print(report)
        else:
            with open(path, mode='w', encoding='utf8') as text_file:
                text_file.write("%s" % mk_report.get_whole_html())


    def report_scores(self, result, use_std=True):
        dict_report = self.__get_report_values(result=result, use_std=use_std)
        headers = ['Labels', 'Precision', 'Recall', 'F1-score', 'Support']
        report = 'h1. ' + result.name + '\n\n'
        report += 'h2. Scores\n\n'
        report += '|_. ' + '|_. '.join(headers) + '|\n'

        # Label
        ulabels = result.get_unique_from_key('Label')
        ulabels = result.decode('label', ulabels)
        for ind, label in enumerate(ulabels):
            label_report = dict_report[label]
            report += '|' + label.capitalize()
            for key in label_report.keys():
                report += '|{value}'.format(value=label_report[key])
            report += '|\n'

        # Average
        avg_report = dict_report['weighted avg']
        report += '|' + 'weighted avg'.capitalize()
        for key in avg_report.keys():
            report += '|{value}'.format(value=avg_report[key])
        report += '|\n\n'

        report += 'h2. Parameters\n\n'
        report += '|_. Folds|_. HyperParameters|_. Number Features|\n'
        for fold, value in enumerate(self.__parameters(result=result)):
            report += '|Fold {fold}|{params}|{nb_feat}|\n'.format(fold=fold+1, params=value[0], nb_feat=value[1])

        return report

    def __parameters(self, result):
        unique_folds = result.get_unique_from_key('Fold')
        params = []
        for fold in unique_folds:
            filter_by = {'Fold': [fold]}
            best_params = str(result.get_from_key(key='BestParams', filters=filter_by)[0])
            features_number = str(result.get_from_key(key='FeaturesNumber', filters=filter_by)[0])
            params.append((best_params, features_number))
        return params

    def __get_report_values(self, result, use_std=True):
        report = self.__report_values_fold(result)

        if use_std is False:
            for label, val in report.items():
                for metrics in val.keys():
                    report[label][metrics] = '{mean:0.2f}'.format(mean=report[label][metrics])
        else:
            scores = []
            unique_folds = result.get_unique_from_key('Fold')
            for fold in unique_folds:
                scores.append(self.__report_values_fold(result=result, fold=fold))

            # Browse reference dict
            for label, val in report.items():
                for metrics in val.keys():
                    values = [score[label][metrics] for score in scores if label in score.keys()]
                    report[label][metrics] = '{mean:0.2f}±{std:0.2f}'.format(mean=report[label][metrics],
                                                                             std=np.std(values))

        # Return report
        return report

    def __report_values_fold(self, result, fold=None):
        if fold is None:
            labels = result.get_from_key(key='Label', flatten=True)
            predictions = result.get_from_key(key='Prediction', flatten=True)
        else:
            filter_by = {'Fold': [fold]}
            labels = result.get_from_key(key='Label', filters=filter_by, flatten=True)
            predictions = result.get_from_key(key='Prediction', filters=filter_by, flatten=True)
        return classification_report(result.decode('label', labels),
                                     result.decode('label', predictions), output_dict=True)


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


class PatchWriter:

    def __init__(self, inputs, settings):
        self.inputs = inputs
        self.settings = settings

    def write_patch(self, output_folder):
        # Check output folder
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)

        output_folder = output_folder/self.inputs.name
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
                center = ((end[0]+start[0])/2, (end[1]+start[1])/2)
                center = tuple(np.subtract(center, 10)), tuple(np.add(center, 10))
                predict = entity['PredictorTransform']
                color = self.settings.get_color(self.inputs.decode('label', predict))+(0.5,) #Add alpha
                color = tuple(np.multiply(color, 255).astype(int))
                draw = ImageDraw.Draw(image)
                draw.rectangle(center, fill=color)
                # draw.rectangle((start, end), outline="white")
            image.save(output_folder/'{ref}_{lab}.png'.format(ref=reference, lab=label[0]))


class ObjectManager:
    @staticmethod
    def save(obj, path):
        with open(path, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as input:
            return pickle.load(input)

from collections import Counter
from copy import deepcopy
from os import makedirs
import markups
import pandas
import pickle

from PIL import Image, ImageDraw
from matplotlib import pyplot, cm
from os.path import join, exists
from keras import backend as K
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imsave
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve, classification_report
from tensorboard.plugins import projector
import tensorflow


class DataProjectorWriter:

    # Have to improve this tool, see callbacks.py to get clues
    @staticmethod
    def project_data(inputs, output_folder):

        if not K.backend() == 'tensorflow':
            return

        # Write a batch to easily launch it
        DataProjectorWriter.write_batch(output_folder)

        sess = K.get_session()

        datas = inputs.get_datas()
        labels = inputs.get_labels(encode=False)

        # Write data
        data_path = join(output_folder, 'data.ckpt')
        tf_data = tensorflow.Variable(datas)
        saver = tensorflow.train.Saver([tf_data])
        sess.run(tf_data.initializer)
        saver.save(sess, data_path)

        # Write label as metadata
        metadata_path = join(output_folder, 'metadata.tsv')
        np.savetxt(metadata_path, labels, delimiter='\t', fmt='%s')

        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name
        # Link this tensor to its metadata(Labels) file
        embedding.metadata_path = metadata_path
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tensorflow.summary.FileWriter(output_folder), config)

    @staticmethod
    def write_batch(output_folder):
        file = open(join(output_folder, 'tb_launch.bat'), "w")
        file.write('tensorboard --logdir={}'.format("./"))
        file.close()


class StatisticsWriter:

    def __init__(self, keys, dir_name, name):
        self.keys = keys
        self.pdf = PdfPages(join(dir_name, name + "_stat.pdf"))

    def end(self):
        self.pdf.close()

    def write(self, inputs):
        figure, axes = pyplot.subplots(ncols=len(self.keys), figsize=(21, 7))
        # Browse each kind of parameter
        for index, key in enumerate(self.keys):
            if not inputs.is_valid_keys(keys=[key]):
                print('Key {key} is missing from data.'.format(key=key))
                continue

            elements = inputs.get_from_key(key=key)
            counter = Counter(list(elements))
            axes[index].pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
            axes[index].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            axes[index].set_title(key)
        figure.suptitle(inputs.name)
        self.pdf.savefig(figure)
        pyplot.close()


class ResultWriter:

    def __init__(self, results, settings):
        self.results = results
        if not isinstance(self.results, list):
            self.results = [self.results]

        self.settings = settings

    def write_results(self, dir_name, name, use_std=True):
        self.write_report(use_std=use_std, path=join(dir_name, '{name}_report.html'.format(name=name)))
        self.write_roc(path=join(dir_name, '{name}_rocs.pdf'.format(name=name)))
        self.write_misclassified(path=join(dir_name, '{name}_misclassified.csv'.format(name=name)))

    def write_misclassified(self, path=None):
        for result in self.results:
            if not result.is_valid_keys(['Label', 'Prediction', 'Reference']):
                print('Missing tag for misclassification report.')
                return

            labels = result.decode('label', np.concatenate(result.get_from_key(key='Label'), axis=0))
            predictions = result.decode('label', np.concatenate(result.get_from_key(key='Prediction'), axis=0))
            references = np.concatenate(result.get_from_key(key='Reference'), axis=0)
            misclassified = [index for index, (i, j) in enumerate(zip(labels, predictions)) if i != j]
            data = {'paths': references[misclassified],
                    'labels': labels[misclassified],
                    'predictions': predictions[misclassified]}
            pandas.DataFrame(data).to_csv(path_or_buf=path, index=False)

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
            with open(path, "w") as text_file:
                text_file.write("%s" % mk_report.get_document_body())

    def write_roc(self, path):

        with PdfPages(path) as pdf:
            for result in self.results:
                if not result.is_valid_keys(['Label', 'Prediction', 'Probability']):
                    print('Missing tag for Roc Curves report.')
                    return

                positives_indices = result.get_unique_from_key('Label')
                # Labels
                labels = result.get_from_key(key='Label')
                labels = np.concatenate(labels, axis=0)
                # Probabilities for drawing
                probabilities = result.get_from_key(key='Probability')
                probabilities = np.concatenate(probabilities, axis=0)

                figure, axe = pyplot.subplots(ncols=1, figsize=(10, 10))
                # Plot luck
                axe.plot([0, 1], [0, 1], linestyle='--', lw=2, color=self.settings.get_color('Luck'), label='Luck', alpha=.8)
                title = ''

                # Browse each label
                for positive_index in positives_indices:
                    positive_class = result.decode('label', positive_index)
                    fpr, tpr, threshold = roc_curve(labels,
                                                    probabilities[:, positive_index],
                                                    pos_label=positive_index)
                    axe.plot(fpr, tpr, lw=2, alpha=.8, color=self.settings.get_color(positive_class),
                             label='ROC {label} (AUC = {auc:.2f})'.format(label=positive_class, auc=auc(fpr, tpr)),
                             **self.settings.get_line(positive_class))

                    # Switch depend on the mode of display
                    title = 'Receiver operating characteristic'

                # Now set title and legend
                axe.set(adjustable='box',
                        aspect='equal',
                        xlabel='False Positive Rate (1-Specificity)',
                        ylabel='True Positive Rate (Sensitivity)',
                        title=title)
                axe.legend(loc='lower right')  # If better than random, no curve is display on bottom right part

                figure.suptitle(result.name)
                pdf.savefig(figure)
                pyplot.close()

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


class PCAProjection:
    def __init__(self, settings, dir_name, name):
        self.settings = settings
        self.pdf = PdfPages(join(dir_name, name + "_pca.pdf"))

    def end(self):
        self.pdf.close()

    def write_projection(self, inputs):
        data = inputs.get_datas()
        labels = inputs.get_labels(encode=False)
        ulabels = inputs.get_unique_labels(encode=False)
        pca = PCA(2, whiten=True)  # project to 2 dimensions
        projected = pca.fit_transform(data)
        figure = pyplot.figure()
        for label in ulabels:
            pyplot.scatter(projected[labels == label, 0], projected[labels == label, 1],
                           c=np.array(self.settings.get_color(label)), alpha=0.5, label=label, edgecolor='none')
        pyplot.xlabel('component 1')
        pyplot.ylabel('component 2')
        pyplot.legend(loc='lower right')
        figure.suptitle(inputs.name)
        self.pdf.savefig(figure)
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


class PatchWriter:

    def __init__(self, inputs, settings):
        self.inputs = inputs
        self.settings = settings

    def write_patch(self, folder):
        folder = join(join(folder, 'Patches_decision'), self.inputs.name)
        if not exists(folder):
            makedirs(folder)
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
            image.save(join(folder, '{ref}_{lab}.png'.format(ref=reference, lab=label[0])))


class ObjectManager:
    @staticmethod
    def save(obj, path):
        with open(path, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as input:
            return pickle.load(input)

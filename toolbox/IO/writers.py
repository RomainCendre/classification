from collections import Counter
from itertools import cycle
from math import ceil
from os import makedirs
import markups
import pandas
import pickle

from PIL import Image
from matplotlib import pyplot, cm
from os.path import join, exists
from keras import backend as K
from matplotlib.image import imsave
from numpy import std, repeat, newaxis, uint8, arange, savetxt, array, copy
from sklearn.metrics import auc, roc_curve, classification_report
from vis.utils.utils import load_img
from vis.visualization import visualize_cam, overlay
from tensorboard.plugins import projector
import tensorflow
from toolbox.core.classification import KerasBatchClassifier
from toolbox.core.generators import ResourcesGenerator


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
        savetxt(metadata_path, labels, delimiter='\t', fmt='%s')

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

    def __init__(self, inputs):
        self.inputs = inputs

    def write_result(self, keys, dir_name, name):
        self.write_stats(path=join(dir_name, name + "_stat.pdf"), keys=keys)

    def write_stats(self, keys, path=None):
        nb_chart = len(keys)

        # Browse each kind of parameter
        for index, key in enumerate(keys):
            elements = self.inputs.get_from_key(key=key)
            if elements is None:
                print('Key {key} is missing from data.'.format(key=key))
                continue

            counter = Counter(list(elements))
            pyplot.subplot(ceil(nb_chart/2), 2, index+1)
            pyplot.pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
            pyplot.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            pyplot.title(key)

        if path is None:
            pyplot.show()
        else:
            pyplot.savefig(path, format='pdf')
        pyplot.clf()


class ResultWriter:

    def __init__(self, inputs, results, settings):
        self.inputs = inputs
        self.results = results
        self.settings = settings

    def write_results(self, dir_name, name, pos_label=[], use_std=True):
        self.write_report(use_std=use_std, path=join(dir_name, "{name}_report.html".format(name=name)))
        self.write_roc(pos_label, path=join(dir_name, "{name}_rocs.pdf".format(name=name)), single_axe=False)
        self.write_roc(pos_label, path=join(dir_name, "{name}_roc.pdf".format(name=name)), single_axe=True)
        self.write_misclassified(path=join(dir_name, "{name}_misclassified.csv".format(name=name)))

    def write_misclassified(self, path=None):
        if not self.results.is_valid_keys(['Label', 'Prediction', 'Reference']):
            print('Missing tag for misclassification report.')
            return

        labels = self.inputs.decode('label', self.results.get_data(key='Label'))
        predictions = self.inputs.decode('label', self.results.get_data(key='Prediction'))
        references = self.results.get_data(key='Reference')
        misclassified = [index for index, (i, j) in enumerate(zip(labels, predictions)) if i != j]
        data = {'paths': references[misclassified],
                'labels': labels[misclassified],
                'predictions': predictions[misclassified]}
        pandas.DataFrame(data).to_csv(path_or_buf=path, index=False)

    def write_report(self, use_std=True, path=None):
        if not self.results.is_valid_keys(['Label', 'Prediction']):
            print('Missing tag for global report.')
            return

        # Initialize converter of markup
        markup = markups.TextileMarkup()

        # Get report
        report = ''
        report += self.report_scores(use_std)

        # Write to html way
        mk_report = markup.convert(report)
        if path is None:
            print(report)
        else:
            with open(path, "w") as text_file:
                text_file.write("%s" % mk_report.get_document_body())

    def write_roc(self, positives_classes=[], single_axe=False, path=None):
        if not self.results.is_valid_keys(['Label', 'Prediction', 'Probability']):
            print('Missing tag for Roc Curves report.')
            return

        if not positives_classes:
            positives_indices = self.results.get_unique_values('Label')
        else:
            positives_indices = self.inputs.encode_label(positives_classes)

        labels = self.results.get_data(key='Label')
        probabilities = self.results.get_data(key='Probability')
        lines = ['-', '-.', ':']
        linecycler = cycle(lines)
        colors = self.settings.get_color('draw')
        if single_axe:
            figure, axe = pyplot.subplots(ncols=1, figsize=(21, 7), sharex=True, sharey=True)
            axe.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
            for index, positive_index in enumerate(positives_indices):
                # Get AUC results for current positive class
                positive_class = self.inputs.decode('label', positive_index)
                fpr, tpr, threshold = roc_curve(labels,
                                                probabilities[:, positive_index],
                                                pos_label=positive_index)

                axe.plot(fpr, tpr, next(linecycler), lw=2, alpha=.8, color=colors[positive_class],
                         label='ROC {label} (AUC = {auc:.2f})'.format(label=positive_class, auc=auc(fpr, tpr)))
                axe.set(adjustable='box',
                        aspect='equal',
                        xlabel='False Positive Rate (1-Specificity)',
                        ylabel='True Positive Rate (Sensitivity)',
                        title='Receiver operating characteristic')
                axe.legend(loc='lower right')  # If better than random, no curve is display on bottom right part
        else:
            figure, axes = pyplot.subplots(ncols=len(positives_indices), figsize=(21, 7), sharex=True, sharey=True)
            if len(positives_classes) == 1:
                axes = [axes]

            for index, axe in enumerate(axes):
                # Get AUC results for current positive class
                positive_index = positives_indices[index]
                positive_class = self.inputs.decode('label', positive_index)
                fpr, tpr, threshold = roc_curve(labels,
                                                probabilities[:, positive_index],
                                                pos_label=positive_index)

                axe.plot(fpr, tpr, label=r'ROC %s (AUC = %0.2f)' % (self.results.name, auc(fpr, tpr)), lw=2, alpha=.8)
                axe.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
                axe.set(adjustable='box',
                        aspect='equal',
                        xlabel='False Positive Rate (1-Specificity)',
                        ylabel='True Positive Rate (Sensitivity)',
                        title='Receiver operating characteristic - Label {label}'.format(label=positive_class))
                axe.legend(loc='lower right')  # If better than random, no curve is display on bottom right part

        if path is None:
            pyplot.show()
        else:
            pyplot.savefig(path, format='pdf')
        pyplot.clf()

    def report_scores(self, use_std=True):
        dict_report = self.__get_report_values(use_std=use_std)
        headers = ['Labels', 'Precision', 'Recall', 'F1-score', 'Support']
        report = 'h1. ' + self.results.name + '\n\n'
        report += '|_. ' + '|_. '.join(headers) + '|\n'

        # Label
        ulabels = self.results.get_unique_values('Label')
        ulabels = self.inputs.decode('label', ulabels)
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
        report += '|\n'

        return report

    def __get_report_values(self, use_std=True):
        report = self.__report_values_fold()

        if use_std is False:
            for label, val in report.items():
                for metrics in val.keys():
                    report[label][metrics] = '{mean:0.2f}'.format(mean=report[label][metrics])
        else:
            scores = []
            unique_folds = self.results.get_unique_values('Fold')
            for fold in unique_folds:
                scores.append(self.__report_values_fold(fold=fold))

            # Browse reference dict
            for label, val in report.items():
                for metrics in val.keys():
                    values = [score[label][metrics] for score in scores if label in score.keys()]
                    report[label][metrics] = '{mean:0.2f}±{std:0.2f}'.format(mean=report[label][metrics],
                                                                             std=std(values))

        # Return report
        return report

    def __report_values_fold(self, fold=None):
        if fold is None:
            labels = self.results.get_data(key='Label')
            predictions = self.results.get_data(key='Prediction')
        else:
            filter_by = {'Fold': [fold]}
            labels = self.results.get_data(key='Label', filter_by=filter_by)
            predictions = self.results.get_data(key='Prediction', filter_by=filter_by)
        return classification_report(self.inputs.decode('label', labels),
                                     self.inputs.decode('label', predictions), output_dict=True)


class VisualizationWriter:

    def __init__(self, model, preprocess=None):
        self.model = model
        self.preprocess = preprocess

    def __get_activation_map(self, seed_input, predict, image):
        if image.ndim == 2:
            image = repeat(image[:, :, newaxis], 3, axis=2)

        grads = visualize_cam(self.model, len(self.model.layers) - 1, filter_indices=predict, seed_input=seed_input,
                              backprop_modifier='guided')

        jet_heatmap = uint8(cm.jet(grads)[..., :3] * 255)
        return overlay(jet_heatmap, image)

    def write_activations_maps(self, inputs, output_folder):

        # Activation dir
        activation_dir = join(output_folder, 'Activation/')
        if not exists(activation_dir):
            makedirs(activation_dir)

        # Check for model type, will be changed in future
        if not isinstance(self.model, KerasBatchClassifier):
            return

        # Extract data for fit
        paths = inputs.get_datas()
        labels = inputs.get_labels()
        ulabels = inputs.get_unique_labels()

        # Prepare data
        generator = ResourcesGenerator(preprocessing_function=self.preprocess)
        valid_generator = generator.flow_from_paths(paths, labels, batch_size=1, shuffle=False)

        # Folds storage
        for index in arange(len(valid_generator)):
            x, y = valid_generator[index]

            for label_index in ulabels:
                dir_path = join(activation_dir, inputs.decode('label', label_index))
                if not exists(dir_path):
                    makedirs(dir_path)

                file_path = join(dir_path, '{number}.png'.format(number=index))

                try:
                    activation = self.__get_activation_map(seed_input=x, predict=label_index,
                                                           image=load_img(paths[index]))
                    imsave(file_path, activation)
                except:
                    print('Incompatible model or trouble occurred.')


class PatchWriter:

    def __init__(self, inputs):
        self.inputs = inputs

    def write_patch(self, folder, patch_size=250):
        print(folder)
        references = list(set(self.inputs.get_from_key('Reference')))

        for index, reference in enumerate(references):
            entities = self.inputs.to_sub_input({'Reference': [reference]})
            path = list(set(entities.get_from_key('Full_path')))
            label = list(set(entities.get_from_key('Label')))
            image = array(Image.open(path[0]).convert('L'))
            image = repeat(image[:, :, newaxis], 3, axis=2)
            predict_map = copy(image)
            for entity in entities.data.data_set:
                X = entity.data['Patch_Position_X']
                Y = entity.data['Patch_Position_Y']
                predict = entity.data['PredictorTransform']
                color = self.inputs.get_style('patches', self.inputs.decode('label', predict.item(0)))
                predict_map[X:X+patch_size, Y:Y+patch_size, :] = color
            imsave(join(folder, '{ref}_{lab}'.format(ref=reference, lab=label[0])), overlay(predict_map, image))


class ObjectManager:
    @staticmethod
    def save(obj, path):
        with open(path, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as input:
            return pickle.load(input)

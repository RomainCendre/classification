from collections import Counter
from math import ceil
import markups
import pandas
import pickle
from matplotlib import pyplot
from os.path import join

from numpy import std
from sklearn.metrics import auc, roc_curve, classification_report


class StatisticsWriter:

    def __init__(self, data_set):
        self.data_set = data_set

    def write_result(self, keys, dir_name, name, filter_by={}):
        self.write_stats(path=join(dir_name, name + "_stat.pdf"), keys=keys, filter_by=filter_by)

    def write_stats(self, keys, path=None, filter_by={}):
        nb_chart = len(keys)

        # Browse each kind of parameter
        for index, key in enumerate(keys):
            counter = Counter(list(self.data_set.get_data(key=key, filter_by=filter_by)))
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

    def __init__(self, result):
        self.result = result

    def write_results(self, dir_name, name, pos_label=[], use_std=True):
        self.write_report(use_std=use_std, path=join(dir_name, "{name}_report.html".format(name=name)))
        self.write_roc(pos_label, path=join(dir_name, name + "{name}_roc.pdf".format(name=name)))
        self.write_misclassified(path=join(dir_name, name + "{name}_misclassified.csv".format(name=name)))

    def write_misclassified(self, path=None):
        if not self.result.is_valid_keys(['Label', 'Prediction', 'Reference']):
            print('Missing tag for misclassification report.')
            return

        labels = self.result.get_data(key='Label')
        predictions = self.result.get_data(key='Prediction')
        references = self.result.get_data(key='Reference')
        misclassified = [index for index, (i, j) in enumerate(zip(labels, predictions)) if i == j]
        data = {'paths': references[misclassified],
                'labels': labels[misclassified],
                'predictions': predictions[misclassified]}
        pandas.DataFrame(data).to_csv(path_or_buf=path)

    def write_report(self, use_std=True, path=None):
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

    def write_roc(self, positives_classes=[], path=None):
        if not positives_classes:
            positives_classes = self.result.get_unique_values('Label')

        labels = self.result.get_data(key='Label')
        probabilities = self.result.get_data(key='Probability')

        figure, axes = pyplot.subplots(ncols=len(positives_classes), figsize=(21, 7), sharex=True, sharey=True)
        for index, axe in enumerate(axes):
            # Get AUC results for current positive class
            positive_class = positives_classes[index]
            positive_index = self.result.map_index.index(positive_class)
            fpr, tpr, threshold = roc_curve(labels,
                                            probabilities[:, positive_index],
                                            pos_label=positive_class)

            axe.plot(fpr, tpr, label=r'ROC %s (AUC = %0.2f)' % (self.result.name, auc(fpr, tpr)), lw=2, alpha=.8)
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
        report = 'h1. ' + self.result.name + '\n\n'
        report += '|_. ' + '|_. '.join(headers) + '|\n'

        # Label
        ulabels = self.result.get_unique_values('Label')
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
            unique_folds = self.result.get_unique_values('Fold')
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
            labels = self.result.get_data(key='Label')
            predictions = self.result.get_data(key='Prediction')
        else:
            filter_by = {'Fold': [fold]}
            labels = self.result.get_data(key='Label', filter_by=filter_by)
            predictions = self.result.get_data(key='Prediction', filter_by=filter_by)
        return classification_report(labels, predictions, output_dict=True)


class ObjectManager:
    @staticmethod
    def save(obj, path):
        with open(path, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as input:
            return pickle.load(input)

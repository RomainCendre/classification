from collections import defaultdict
from itertools import chain

from numpy import std, average, asarray
from numpy.ma import array
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.utils.multiclass import unique_labels


class Results:
    """Class that manage a result spectrum files.

    In this class we afford to manage spectrum results to write it on files.

    Attributes:

    """

    def __init__(self, labels, folds, predictions, map_index, probabilities, name=''):
        """Make an initialisation of SpectrumReader object.

        Take a string that represent delimiter

        Args:
             labels (:obj:'ndarray'): The truth.
             folds (:obj:'ndarray'): The folds used to test data.
             predictions (:obj:'ndarray'): The predictions made by process.
             map_index
             probabilities (:obj:'ndarray'): The probabilities of each class.
        """
        self.labels = labels
        self.ulabels = unique_labels(self.labels)
        self.folds = folds
        self.map_index = map_index
        self.name = name
        self.predictions = predictions
        self.probabilities = probabilities

    def __report_values(self, use_std=True):
        report = classification_report(self.labels, self.predictions, output_dict=True)

        if use_std is False:
            for label, val in report.items():
                for metrics in val.keys():
                    report[label][metrics] = '{mean:0.2f}'.format(mean=report[label][metrics])
        else:
            scores = []
            for fold in self.folds:
                scores.append(classification_report(self.labels[fold], self.predictions[fold], output_dict=True))

            # Browse reference dict
            for label, val in report.items():
                for metrics in val.keys():
                    values = [score[label][metrics] for score in scores]
                    report[label][metrics] = '{mean:0.2f}±{std:0.2f}'.format(mean=report[label][metrics],
                                                                             std=std(values))

        # Return report
        return report

    def report_values_fold(self, fold=None):
        return classification_report(self.labels[fold], self.predictions[fold], output_dict=True)

    def report_scores(self, use_std=True):
        print(classification_report(self.labels, self.predictions))
        dict_report = self.__report_values(use_std=use_std)
        headers = ["Labels", "Precision", "Recall", "F1-score", "Support"]
        report = 'h1. ' + self.name + '\n\n'
        report += '|_. ' + '|_. '.join(headers) + '|\n'

        # Label
        for ind, label in enumerate(self.ulabels):
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

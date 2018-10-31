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

    def std_scores(self):
        scores = []
        for fold in self.folds:
            scores.append(precision_recall_fscore_support(self.labels[fold], self.predictions[fold],
                                                          labels=self.ulabels, average=None))
        scores = array(scores)
        scores = std(scores, axis=0)

        return scores.tolist()

    def mean_scores(self):
        return precision_recall_fscore_support(self.labels, self.predictions,
                                               labels=self.ulabels, average=None)

    def average_scores(self):
        p, r, f1, s = precision_recall_fscore_support(self.labels, self.predictions,
                                                      labels=self.ulabels, average=None)
        return [average(p, weights=s), average(r, weights=s), average(f1, weights=s), sum(s)]

    def report_scores(self, use_std=True):
        print(classification_report(self.labels, self.predictions))
        ap, ar, af1, asup = self.mean_scores()
        sp, sr, sf1, ssup = self.std_scores()
        headers = ["Labels", "Precision", "Recall", "F1-score", "Support"]
        report = 'h1. ' + self.name + '\n\n'
        report += '|_. ' + '|_. '.join(headers) + '|\n'
        for ind, label in enumerate(self.ulabels):
            report += '|' + label
            report += '| {:.2f}'.format(ap[ind])
            if use_std:
                report += '± {:.2f}'.format(sp[ind])
            report += '| {:.2f}'.format(ar[ind])
            if use_std:
                report += '± {:.2f}'.format(sr[ind])
            report += '| {:.2f}'.format(af1[ind])
            if use_std:
                report += '± {:.2f}'.format(sf1[ind])
            report += '| {:d}'.format(asup[ind]) + '|\n'
        # Average
        p, r, f1, s = self.average_scores()
        report += '|Total|'
        report += "{:.2f}".format(p) + '|'
        report += "{:.2f}".format(r) + '|'
        report += "{:.2f}".format(f1) + '|'
        report += "{:d}".format(s) + '|\n\n'
        return report

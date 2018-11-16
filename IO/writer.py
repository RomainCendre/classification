from collections import Counter
from math import ceil
import markups
from matplotlib import pyplot
from os.path import join
from sklearn.metrics import auc, roc_curve


class StatisticsWriter:

    def __init__(self, data_set):
        self.data_set = data_set

    def write_result(self, metas, dir_name, name, filter_by={}):
        self.write_stats(path=join(dir_name, name + "_stat.pdf"), metas=metas, filter_by=filter_by)

    def write_stats(self, metas, path=None, filter_by={}):
        nb_chart = len(metas)

        # Browse each kind of parameter
        for index, meta in enumerate(metas):
            counter = Counter(list(self.data_set.get_meta(meta=meta, filter_by=filter_by)))
            pyplot.subplot(ceil(nb_chart/2), 2, index+1)
            pyplot.pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
            pyplot.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            pyplot.title(meta)

        if path is None:
            pyplot.show()
        else:
            pyplot.savefig(path, format='pdf')
        pyplot.clf()


class ResultWriter:

    def __init__(self, result):
        self.result = result

    def write_result(self, pos_label, dir_name):
        self.write_report(path=join(dir_name, self.result.name + ".html"))
        self.write_roc(pos_label, path=join(dir_name, self.result.name + ".png"))

    def write_results(self, dir_name, name, pos_label=[], use_std=True):
        self.write_report(use_std=use_std, path=join(dir_name, name + ".html"))
        self.write_roc(pos_label, path=join(dir_name, name + ".pdf"))

    def write_report(self, use_std=True, path=None):
        # Initialize converter of markup
        markup = markups.TextileMarkup()

        # Get report
        report = ''
        report += self.result.report_scores(use_std)

        # Write to html way
        mk_report = markup.convert(report)
        if path is None:
            print(report)
        else:
            with open(path, "w") as text_file:
                text_file.write("%s" % mk_report.get_document_body())

    def write_roc(self, positives_classes=[], path=None):
        if not positives_classes:
            positives_classes = self.result.ulabels

        figure, axes = pyplot.subplots(ncols=len(positives_classes), figsize=(21, 7), sharex=True, sharey=True)
        for index, axe in enumerate(axes):
            # Get AUC results for current positive class
            positive_class = positives_classes[index]
            positive_index = self.result.map_index.index(positive_class)
            fpr, tpr, threshold = roc_curve(self.result.labels,
                                            self.result.probabilities[:, positive_index],
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

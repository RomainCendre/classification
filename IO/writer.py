from collections import Counter
from math import ceil
import markups
from matplotlib import pyplot
from os.path import join
from sklearn.metrics import auc, roc_curve


class StatisticsWriter:

    def __init__(self, spectra):
        self.spectra = spectra

    def write_result(self, dir_name, name):
        self.write_stats(path=join(dir_name, name + "_stat.pdf"))

    def write_stats(self, path=None):
        # Get meta data to create chart
        meta = self.spectra.get_meta_info()
        nb_chart = len(meta.items())

        # Browse each kind of parameter
        for index, (title, list) in enumerate(meta.items()):
            keys = Counter(list).keys()
            values = Counter(list).values()
            pyplot.subplot(ceil(nb_chart/2), 2, index+1)
            pyplot.pie(values, labels=keys, autopct='%1.1f%%', startangle=90)
            pyplot.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            pyplot.title(title)

        if path is None:
            pyplot.show()
        else:
            pyplot.savefig(path, format='pdf')
        pyplot.clf()


class ResultsWriter:

    def __init__(self, result):
        self.result = result

    def write_result(self, pos_label, dir_name):
        self.write_report(path=join(dir_name, self.result.name + ".html"))
        self.write_roc(pos_label, path=join(dir_name, self.result.name + ".png"))

    def write_results(self, pos_label, dir_name, name, use_std=True):
        self.write_report(use_std=use_std, path=join(dir_name, name + ".html"))
        self.write_roc(pos_label, path=join(dir_name, name + ".pdf"))

    def write_report(self, use_std=True, path=None):
        markup = markups.TextileMarkup()
        report = ''
        for res in self.result:
            report += res.report_scores(use_std)

        mk_report = markup.convert(report)
        if path is None:
            print(report)
        else:
            with open(path, "w") as text_file:
                text_file.write("%s" % mk_report.get_document_body())

    def write_roc(self, pos_label, path=None):
        if isinstance(self.result, list):
            for res in self.result:
                pos_index = res.map_index.index(pos_label)
                fpr, tpr, threshold = roc_curve(res.labels, res.probabilities[:, pos_index],
                                                pos_label=pos_label)
                auc_val = auc(fpr, tpr)
                pyplot.plot(fpr, tpr, label=r'ROC %s (AUC = %0.2f)' % (res.name, auc_val), lw=2, alpha=.8)
        else:
            pos_index = self.result.map_index.index(pos_label)
            fpr, tpr, threshold = roc_curve(self.result.labels, self.result.probabilities[:, pos_index], pos_label=pos_label)
            auc_val = auc(fpr, tpr)
            pyplot.plot(fpr, tpr, label=r'ROC %s (AUC = %0.2f)' % (self.result.name, auc_val), lw=2, alpha=.8)

        pyplot.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
        pyplot.xlabel('False Positive Rate (1-Specificity)')
        pyplot.ylabel('True Positive Rate (Sensitivity)')
        pyplot.title('Receiver operating characteristic')
        pyplot.legend(loc="lower right")
        if path is None:
            pyplot.show()
        else:
            pyplot.savefig(path, format='pdf')
        pyplot.clf()

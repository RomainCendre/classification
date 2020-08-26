from collections import Counter

import numpy
import pandas
import seaborn as seaborn
from matplotlib import pyplot
from toolbox.classification.parameters import Dermatology
from toolbox.views.common import ViewsTools, Views

# Prepare data
inputs = Dermatology.images(data_type='Full', modality='Microscopy', use_unknown=True)

## ELISA
inputs_elisa = inputs[inputs['ID_Table'] == 'Elisa']
inputs_elisa = inputs_elisa.groupby('ID_Lesion').first()
inputs_elisa['Age'] = pandas.to_numeric(inputs_elisa['Age'])

# Prepare plots
ViewsTools.plot_size((12, 8))
figure = pyplot.figure()

# Display different informations
axes = figure.add_subplot(1, 3, 1)
seaborn.violinplot(x='Sex', y='Age', data=inputs_elisa)

axes = figure.add_subplot(1, 3, 2)
seaborn.violinplot(x='Sex', y='Age', data=inputs_elisa, hue='Binary_Diagnosis', split=True)

axes = figure.add_subplot(1, 3, 3)
unique = (0, 9, 6)
common = (6, 6, 6)
ind = numpy.arange(3)    # the x locations for the groups
p1 = pyplot.bar(ind, common, 0.35)
p2 = pyplot.bar(ind, unique, 0.35, bottom=common)

pyplot.ylabel('Nombre experts')
pyplot.xticks(ind, ('All', 'Clinical+Dermoscopy', 'RCM'))
# pyplot.yticks(numpy.arange(0, 81, 10))
pyplot.legend((p1[0], p2[0]), ('Commun', 'Uniques'))
figure.show()
figure.savefig('elisa_statistics.pdf', bbox_inches='tight')

## FULL
cat = seaborn.catplot(x='Label', order=['Malignant', 'Benign', 'Normal', 'Unknown'], kind="count", hue='ID_Table', data=inputs)
# pyplot.legend(loc='upper left')
pyplot.show()
cat.savefig('full_statistics.pdf', bbox_inches='tight')

## Lesions
inputs = inputs[inputs['ID_Image']=='0M']
cat = seaborn.catplot(x='Binary_Diagnosis', order=['Malignant', 'Benign'], kind="count", hue='ID_Table', data=inputs)
# pyplot.legend(loc='upper left')
pyplot.show()
cat.savefig('lesions_statistics.pdf', bbox_inches='tight')

inputs_patch = Dermatology.images(data_type='Patch', modality='Microscopy', use_unknown=True)
## WHOLE
cat = seaborn.catplot(x='Label', order=['Malignant', 'Benign', 'Normal', 'Unknown'], kind="count", hue='ID_Table', data=inputs_patch)
# pyplot.legend(loc='upper left')
pyplot.show()
cat.savefig('patch_statistics.pdf', bbox_inches='tight')
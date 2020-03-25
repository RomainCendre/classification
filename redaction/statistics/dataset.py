from collections import Counter

import numpy
import pandas
import seaborn as seaborn
from matplotlib import pyplot
from toolbox.classification.parameters import Dermatology
from toolbox.views.common import ViewsTools, Views

# Prepare data
inputs = Dermatology.images(data_type='Full')
inputs = inputs[inputs['TableID'] == 'Elisa']
inputs = inputs.groupby('ID').first()
inputs['Age'] = pandas.to_numeric(inputs['Age'])

# Prepare plots
ViewsTools.plot_size((12, 8))
figure = pyplot.figure()

axes = figure.add_subplot(2, 3, 1)
seaborn.violinplot(x='Sex', y='Age', data=inputs)

axes = figure.add_subplot(2, 3, 2)
seaborn.violinplot(x='Sex', y='Age', data=inputs, hue='Binary_Diagnosis', split=True)

axes = figure.add_subplot(2, 3, 3)
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
figure.savefig("statistics.pdf", bbox_inches='tight')

# ViewsTools.plot_size((12, 8))
# figure = pyplot.figure()
#
# axes = figure.add_subplot(2, 3, 1)
# counter = Counter(list(inputs['Binary_Diagnosis']))
# axes.pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
# axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# axes = figure.add_subplot(2, 3, 2)
# counter = Counter(list(inputs['Age']))
# # axes.pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
# counts.plot(kind='bar', stacked=True)
# axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# axes = figure.add_subplot(2, 3, 3)
# counter = Counter(list(inputs['Sex']))
# axes.pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
# axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# axes = figure.add_subplot(2, 3, 4)
# counter = Counter(list(inputs['Area']))
# axes.pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
# axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
#
# axes = figure.add_subplot(2, 3, 5)
# counter = Counter(list(inputs['Diagnosis']))
# axes.pie(list(counter.values()), labels=list(counter.keys()), autopct='%1.1f%%', startangle=90)
# axes.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#
# figure.show()
#
# ViewsTools.plot_size((12, 8))
# figure = pyplot.figure()
#
# axes = figure.add_subplot(2, 3, 1)
# seaborn.stripplot(x='Sex', y='Age', data=inputs, jitter=True)
#
# axes = figure.add_subplot(2, 3, 2)
# seaborn.stripplot(x='Binary_Diagnosis', y='Age', data=inputs, jitter=True)
#
# axes = figure.add_subplot(2, 3, 3)
# seaborn.stripplot(x='Binary_Diagnosis', y='Age', data=inputs, jitter=True, hue='Sex', dodge=True)
# # axes = figure.add_subplot(2, 3, 1)
# # seaborn.boxplot(x='Binary_Diagnosis', y='Sex', data=inputs)
#
# figure.show()

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

fontsize = 12
x = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
y = [3803, 3858, 3906, 3958, 4018, 4061, 4094, 4112, 4131, 4124, 4128, 4157, 4160, 4100, 4104, 4077, 4065, 4003, 3979]

fig = plt.figure()
plt.box(on=None)
plt.gca().yaxis.grid(True)
fig.suptitle('Evolution du nombre de praticiens entre 1999 et 2017', fontsize=fontsize)
plt.bar(x, y)
plt.xlabel('Années', fontsize=fontsize)
plt.ylabel('Nombre de praticiens', fontsize=fontsize)
# plt.xlim(1999, 2017)
plt.yticks([3600, 3800, 4000, 4200, 4400])
plt.xticks([2000, 2005, 2010, 2015])
plt.ylim(3600, 4400)
plt.show()
plt.savefig("foo.pdf", bbox_inches='tight')
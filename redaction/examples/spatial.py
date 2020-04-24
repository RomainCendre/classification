import numpy
from numpy import inf
import pandas
import seaborn as seaborn
from matplotlib import pyplot
from toolbox.classification.parameters import Dermatology
from toolbox.views.common import ViewsTools, Views
from PIL import Image
from mahotas import texture
from skimage.feature import greycomatrix, greycoprops
from skimage.exposure import rescale_intensity

benin = numpy.array(Image.open('benin.bmp').convert('L'))
malin = numpy.array(Image.open('malin.bmp').convert('L'))
originales = [benin, malin]

benin_glcm = greycomatrix(benin, distances=[1], angles=[0, numpy.pi/4.0, numpy.pi/2.0, 3*numpy.pi/4.0], levels=256, symmetric=True, normed=False)
malin_glcm = greycomatrix(malin, distances=[1], angles=[0, numpy.pi/4.0, numpy.pi/2.0, 3*numpy.pi/4.0], levels=256, symmetric=True, normed=False)
glcm = [benin_glcm, malin_glcm]

images = ['Benin', 'Malin']
directions = ['Horizontal', 'Diagonale_1', 'Vertical', 'Diagonale_2']

fig = pyplot.figure()
fig, axes = pyplot.subplots(2, 5, figsize=(8, 4), dpi=1200)
for im_index, image in enumerate(images):
    axes[im_index, 0].imshow(originales[im_index], cmap='gray', vmin=0, vmax=255)
    axes[im_index, 0].axis('off')
    for index, direction in enumerate(directions):
        data = glcm[im_index][:,:,0,index]
        im = axes[im_index, index+1].imshow(data, cmap='gray', vmin=0, vmax=2500)
        axes[im_index, index+1].set_title(f'{direction}')
        axes[im_index, index+1].axis('off')
fig.subplots_adjust(right=0.8)
pyplot.axis('off')
pyplot.show()

fig.savefig('example_GLCM.pdf', bbox_inches='tight')
import numpy
from numpy import inf
import pandas
import seaborn as seaborn
from matplotlib import pyplot
from PIL import Image

# fonction de normalisation
def norma(mat):
     mat1 = mat.real
     mat1 -= mat1.min()
     mat1 *= 255. / mat1.max()
     return mat1

benin = numpy.array(Image.open('benin.bmp').convert('L'))
malin = numpy.array(Image.open('malin.bmp').convert('L'))
originales = [benin, malin]

# Calcul FFT
benin_fft = numpy.fft.fft2(benin)
benin_fft = numpy.fft.fftshift(benin_fft)
malin_fft = numpy.fft.fft2(malin)
malin_fft = numpy.fft.fftshift(malin_fft)

# # module de la transformée de Fourier de l'image
# benin_fft_abs = abs(benin_fft)
# malin_fft_abs = abs(malin_fft)
#
# # phase de la transformée de Fourier de l'image
# benin_fft_phase = benin_fft / benin_fft_abs
# malin_fft_phase = malin_fft / malin_fft_abs

fft_titles = ['Module', 'Phase']


images = ['Benin', 'Malin']
directions = ['Horizontal', 'Diagonale_1', 'Vertical', 'Diagonale_2']

fig = pyplot.figure()
fig, axes = pyplot.subplots(2, 3, figsize=(8,4), dpi=300)

axes[0, 0].imshow(benin, cmap='gray')
axes[0, 0].axis('off')

axes[0, 1].imshow(numpy.log(1+numpy.abs(benin_fft)), cmap='gray')
# axes[0, 1].imshow(numpy.log10(benin_fft_abs), cmap='gray')
axes[0, 1].set_title('fft module')

axes[0, 2].imshow(numpy.angle(benin_fft), cmap='gray')
axes[0, 2].set_title('fft phase')


axes[1, 0].imshow(malin, cmap='gray')
axes[1, 0].axis('off')

axes[1, 1].imshow(numpy.log(1+numpy.abs(malin_fft)), cmap='gray')
axes[1, 1].set_title('fft module')

axes[1, 2].imshow(numpy.angle(malin_fft), cmap='gray')
axes[1, 2].set_title('fft phase')

pyplot.show()

fig.savefig('example_frequency.pdf', bbox_inches='tight')
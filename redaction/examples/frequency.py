import numpy
import pywt
from numpy import inf
import pandas
import seaborn as seaborn
from matplotlib import pyplot
from PIL import Image
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

benin = numpy.array(Image.open('benin.bmp').convert('L'))
malin = numpy.array(Image.open('malin.bmp').convert('L'))

# Calcul FFT
benin_fft = numpy.fft.fft2(benin)
benin_fft = numpy.fft.fftshift(benin_fft)
malin_fft = numpy.fft.fft2(malin)
malin_fft = numpy.fft.fftshift(malin_fft)
fft_titles = ['Module', 'Phase']

images = ['Benin', 'Malin']
directions = ['Horizontal', 'Diagonale_1', 'Vertical', 'Diagonale_2']

fig = pyplot.figure()
fig, axes = pyplot.subplots(2, 3, figsize=(8, 4), dpi=600)

axes[0, 0].imshow(benin, cmap='gray')
axes[0, 0].axis('off')
axes[0, 1].imshow(numpy.log(1+numpy.abs(benin_fft)), cmap='gray')
axes[0, 1].set_title('Module')
axes[0, 1].axis('off')
axes[0, 2].imshow(numpy.angle(benin_fft), cmap='gray')
axes[0, 2].set_title('Phase')
axes[0, 2].axis('off')
axes[1, 0].imshow(malin, cmap='gray')
axes[1, 0].axis('off')
axes[1, 1].imshow(numpy.log(1+numpy.abs(malin_fft)), cmap='gray')
axes[1, 1].axis('off')
axes[1, 2].imshow(numpy.angle(malin_fft), cmap='gray')
axes[1, 2].axis('off')

pyplot.show()
fig.savefig('example_fft.pdf', bbox_inches='tight')

benin = benin.astype(numpy.float32)
malin = malin.astype(numpy.float32)

max_lev = 3       # how many levels of decomposition to draw
label_levels = 3  # how many levels to explicitly label on the plots

fig, axes = pyplot.subplots(2, 4, figsize=[14, 8], dpi=600)
for level in range(0, max_lev + 1):
    # compute the 2D DWT
    c = pywt.wavedec2(benin, 'db4', mode='periodization', level=level)
    # normalize each coefficient array independently for better visibility
    c[0] /= numpy.abs(c[0]).max()
    for detail_level in range(level):
        c[detail_level + 1] = [d/numpy.abs(d).max() for d in c[detail_level + 1]]
    # show the normalized coefficients
    arr, slices = pywt.coeffs_to_array(c)
    axes[0, level].imshow(arr, cmap='gray')
    axes[0, level].set_title('Coefficients\n({} niveaux)'.format(level))
    axes[0, level].set_axis_off()

for level in range(0, max_lev + 1):
    # compute the 2D DWT
    c = pywt.wavedec2(malin, 'db4', mode='periodization', level=level)
    # normalize each coefficient array independently for better visibility
    c[0] /= numpy.abs(c[0]).max()
    for detail_level in range(level):
        c[detail_level + 1] = [d/numpy.abs(d).max() for d in c[detail_level + 1]]
    # show the normalized coefficients
    arr, slices = pywt.coeffs_to_array(c)
    axes[1, level].imshow(arr, cmap='gray')
    # axes[1, level].set_title('Coefficients\n({} level)'.format(level))
    axes[1, level].set_axis_off()

pyplot.show()
fig.savefig('example_wavelet.pdf', bbox_inches='tight')
from os import makedirs
from time import time
from os.path import expanduser

from numpy import array
from sklearn.model_selection import GroupKFold

from IO.dermatology import Reader
from tools.limitations import Parameters
import mahotas
from PIL import Image

outer_cv = GroupKFold(n_splits=5)

if __name__ == '__main__':

    # Configure GPU consumption
    Parameters.set_gpu(percent_gpu=0.5)

    image = 'C:\\Users\\Romain\\Desktop\\Viewer.png'
    im = array(Image.open(image))
    textures = mahotas.features.haralick(im)
    print(textures)


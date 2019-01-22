from os import makedirs
from os.path import dirname, normpath, exists, join
from tempfile import gettempdir

from toolbox.IO.dermatology import Reader
from toolbox.IO.writers import VisualizationWriter
from toolbox.core.models import DeepModels
from toolbox.core.structures import Inputs

if __name__ == '__main__':

    here_path = dirname(__file__)
    temp_path = gettempdir()

    # Output dir
    output_dir = normpath('{temp}/dermatology/'.format(temp=temp_path))
    if not exists(output_dir):
        makedirs(output_dir)
    print('Output directory: {out}'.format(out=output_dir))

    # Activation dir
    activation_dir = join(output_dir, 'Activation/')
    if not exists(activation_dir):
        makedirs(activation_dir)

    # Load data
    thumbnail_folder = normpath('{here}/data/dermatology/Thumbnails/'.format(here=here_path))
    data_set = Reader().scan_folder_for_images(thumbnail_folder)
    inputs = Inputs(data_set, data_tag='Data', label_tag='Label')

    # Visualization test
    model, preprocess, extractor = DeepModels.get_confocal_model(inputs)
    VisualizationWriter(model=model, preprocess=preprocess).write_activations_maps(directory=activation_dir,
                                                                                   inputs=inputs)
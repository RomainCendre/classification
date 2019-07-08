import numpy as np

import os

from keras.utils import Sequence
from keras_preprocessing import get_keras_submodule
from keras_preprocessing.image import Iterator, ImageDataGenerator, load_img, img_to_array, array_to_img
from numpy import asarray
from PIL import Image as pil_image

backend = get_keras_submodule('backend')


class ResourcesGenerator(ImageDataGenerator):

    def flow_from_paths(self, filenames, labels=None,
                        target_size=None, color_mode='rgb',
                        classes=None, class_mode='categorical',
                        batch_size=32, shuffle=True, seed=None,
                        save_to_dir=None,
                        save_prefix='',
                        save_format='png',
                        subset=None,
                        interpolation='nearest'):
        local_mode = class_mode
        if class_mode == 'both':
            class_mode = None
        iterator = ResourcesIterator(
            filenames, self, labels,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            interpolation=interpolation)
        iterator.class_mode = local_mode
        return iterator


class ResourcesIterator(Iterator, Sequence):

    def __init__(self, filenames, image_data_generator, labels=None,
                 target_size=None, color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 subset=None,
                 interpolation='nearest'):
        if data_format is None:
            data_format = backend.image_data_format()

        self.image_data_generator = image_data_generator
        if color_mode not in {'rgb', 'rgba', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb", "rgba", or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format

        if target_size is not None:
            self.target_size = tuple(target_size)
            self.image_shape = self._build_shape(self.target_size)
        else:
            self.target_size = None

        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        # First, count the number of samples and classes.
        self.samples = len(filenames)

        # List of existing classes
        if labels is None:
            labels = np.zeros((self.samples,), dtype=int)
            if not classes:
                classes = list(set(labels))
            self.num_classes = len(classes)
            self.class_indices = dict(zip(classes, range(len(classes))))
            print('Found {sample} images.'.format(sample=self.samples))
        else:
            if not classes:
                classes = list(set(labels))

            self.num_classes = len(classes)
            self.class_indices = dict(zip(classes, range(len(classes))))

            print('Found %d images belonging to %d classes.' % (self.samples, self.num_classes))

        # Second, build an index of the images
        self.classes = list(map(self.class_indices.get, labels))
        self.classes = asarray([self.classes[index] for index, filename in enumerate(filenames) if filename.endswith(tuple(white_list_formats))])
        self.filenames = [filename for filename in filenames if filename.endswith(tuple(white_list_formats))]

        super(ResourcesIterator, self).__init__(self.samples,
                                                 batch_size,
                                                 shuffle,
                                                 seed)

    def _get_batches_of_transformed_samples(self, index_array):

        target_size = self._get_shape(index_array)
        batch_x = np.zeros((len(index_array),) + self._build_shape(target_size), dtype=backend.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(fname,
                           color_mode=self.color_mode,
                           target_size=target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_shape(self, index_array):
        if self.target_size is not None:
            return self.target_size
        else:
            max_width, max_height = 0, 0
            for i in index_array:
                fname = self.filenames[i]
                with pil_image.open(fname) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)

            return (max_height, max_width)

    def _build_shape(self, target_size):
        if self.color_mode == 'rgba':
            if self.data_format == 'channels_last':
                return target_size + (4,)
            else:
                return (4,) + target_size
        elif self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                return target_size + (3,)
            else:
                return (3,) + target_size
        else:
            if self.data_format == 'channels_last':
                return target_size + (1,)
            else:
                return (1,) + target_size

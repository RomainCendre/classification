from tensorflow.keras.layers import InputSpec,Layer
from tensorflow.keras import backend as K
from numpy.linalg import norm


class NormPooling2D(Layer):
    """Abstract class for different global pooling 2D layers.
    """

    def __init__(self, order=1, data_format=None, **kwargs):
        super(NormPooling2D, self).__init__(**kwargs)
        self.order = order
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], input_shape[3])
        else:
            return (input_shape[0], input_shape[1])

    def call(self, inputs):
        norm(inputs, self.order)
        if self.data_format == 'channels_last':
            return K.max(inputs, axis=[1, 2])
        else:
            return K.max(inputs, axis=[2, 3])

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(NormPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.random_uniform_variable(shape=(1, self.output_dim), low=0, high=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        base_config = super().get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

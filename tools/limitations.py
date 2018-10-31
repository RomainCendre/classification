import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class Parameters:

    @staticmethod
    def set_gpu(percent_gpu=1):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = percent_gpu
        set_session(tf.Session(config=config))

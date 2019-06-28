from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D, regularizers
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras import backend as K
import numpy as np
import os
from PIL import Image as pil_image
import matplotlib.pyplot as plt

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
from sklearn.model_selection import GroupKFold

from toolbox.core.generators import ResourcesGenerator
from toolbox.core.transforms import OrderedEncoder

from toolbox.core.parameters import DermatologyDataset, LocalParameters


def get_inputs():
    # Input patch
    image_inputs = DermatologyDataset.images(modality='Microscopy')
    image_inputs.set_encoders({'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant'])})
    patch_filter = {'Type': ['Patch']}
    image_inputs.set_filters(patch_filter)

    # Data from microscopy
    x = image_inputs.get_datas()
    y = image_inputs.get_labels()
    groups = image_inputs.get_groups()
    train, test = next(GroupKFold(2).split(x, y, groups))
    return x[train], x[test], y[train], y[test]


def get_generators(x_train, x_test, y_train, y_test):
    # Build generators
    generator = ResourcesGenerator(rescale=1. / 255)
    train_generator = generator.flow_from_paths(x_train, y_train, color_mode='grayscale', target_size=(252, 252),
                                                class_mode='input',
                                                batch_size=batch_size)
    validation_generator = generator.flow_from_paths(x_test, y_test, target_size=(252, 252),
                                                     color_mode='grayscale',
                                                     class_mode='input',
                                                     batch_size=batch_size, shuffle=False)
    return train_generator, validation_generator


def ae():
    encoding_dim = 32
    # Encoding network
    input_img = Input(shape=original_dim)
    flat = Flatten()(input_img)
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(flat)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(original_dim[0]*original_dim[1], activation='sigmoid')(encoded)
    output = Reshape(original_dim)(decoded)

    vae = Model(input_img, output)
    vae.compile(optimizer='adadelta', loss='binary_crossentropy')

    return vae


def cae():
    encoding_dim = 32
    input_img = Input(shape=original_dim)
    flat = Flatten()(input_img)
    # add a Dense layer with a L1 activity regularizer
    encoded = Dense(encoding_dim, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(flat)
    decoded = Dense(original_dim[0]*original_dim[1], activation='sigmoid')(encoded)
    output = Reshape(original_dim)(decoded)

    vae = Model(input_img, output)
    vae.compile(optimizer='adadelta', loss='binary_crossentropy')

    return vae


def deep_cae():
    encoding_dim = 32
    input_img = Input(shape=original_dim)
    flat = Flatten()(input_img)
    encoded = Dense(128, activation='relu')(flat)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu',
                    activity_regularizer=regularizers.l1(10e-5))(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(original_dim[0]*original_dim[1], activation='sigmoid')(decoded)
    output = Reshape(original_dim)(decoded)
    vae = Model(input_img, output)
    vae.compile(optimizer='adadelta', loss='binary_crossentropy')

    return vae


def conv_ae():
    input_img = Input(shape=original_dim)
    # Encoding network
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Decoding network
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    vae = Model(input_img, decoded)
    vae.compile(optimizer='adadelta', loss='binary_crossentropy')

    return vae


def vae():
    inputs = Input(shape=original_dim, name='encoder_input')
    flat = Flatten()(inputs)
    x = Dense(intermediate_dim, activation='relu')(flat)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # encoder, from inputs to latent space
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim[0]*original_dim[1], activation='sigmoid')(x)
    outputs = Reshape(original_dim)(outputs)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    def vae_loss(y_true, y_pred):
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= original_dim[0]*original_dim[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    vae.compile(optimizer='adam', loss=vae_loss)

    return vae, encoder


def hybrid_conv_ae():
    input_img = Input(shape=original_dim)
    # Encoding network
    x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    encoder = Conv2D(32, (2, 2), activation='relu', padding="same", strides=2)(x)
    # Decoding network
    x = Conv2D(32, (2, 2), activation='relu', padding="same")(encoder)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoder)


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# Configure GPU consumption
LocalParameters.set_gpu(percent_gpu=0.5)
# network parameters
image_size = 252
original_dim = (image_size, image_size, 1)
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 50

x_train, x_test, y_train, y_test = get_inputs()
train_generator, validation_generator = get_generators(x_train, x_test, y_train, y_test)

# autoencoder = ae()
# autoencoder = cae()
# autoencoder = deep_cae()
# autoencoder = conv_ae()
autoencoder, encoder = vae()

autoencoder.fit_generator(train_generator,
                          nb_epoch=epochs,
                          validation_data=validation_generator,
                          verbose=1)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

decoded_imgs = autoencoder.predict_generator(validation_generator)
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    x_current = pil_image.open(x_test[i])
    plt.imshow(x_current)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i, :, :, 0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
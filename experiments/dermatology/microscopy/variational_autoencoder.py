from __future__ import print_function
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator

# Compute VAE loss
from sklearn.model_selection import KFold

from toolbox.core.generators import ResourcesGenerator
from toolbox.core.parameters import DermatologyDataset
from toolbox.core.transforms import OrderedEncoder


def my_vae_loss(y_true, y_pred):
    xent_loss = img_rows * img_cols * metrics.binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


K.set_image_data_format('channels_first')
K.set_image_dim_ordering('th')

print("Image data format: ", K.image_data_format())
print("Image dimension ordering: ", K.image_dim_ordering())
print("Backend: ", K.backend())

# input image dimensions
img_rows, img_cols, img_chns = 250, 250, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 100
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

latent_dim = 2
intermediate_dim = 128
epsilon_std = 1.0
epochs = 5

print("Original image size: ", original_img_size)
x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 128 * 128, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 128, 128)
else:
    output_shape = (batch_size, 128, 128, filters)

print('Output shape 1: ', output_shape)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 256, 256)
else:
    output_shape = (batch_size, 256, 256, filters)

print('Output shape 2: ', output_shape)

decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# instantiate VAE model
vae = Model(x, x_decoded_mean_squash)

# Compute VAE loss
xent_loss = img_rows * img_cols * metrics.binary_crossentropy(
    K.flatten(x),
    K.flatten(x_decoded_mean_squash))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='rmsprop', loss=my_vae_loss)
vae.summary()

train_datagen = ImageDataGenerator(data_format='channels_first',
                                   rescale=1. / 255)

test_datagen = ImageDataGenerator(data_format='channels_first',
                                  rescale=1. / 255)
# Input patch
image_inputs = DermatologyDataset.images(modality='Microscopy')
image_inputs.set_encoders({'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant'])})
patch_filter = {'Type': ['Patch']}
image_inputs.set_filters(patch_filter)

# Data from microscopy
x = image_inputs.get_datas()
y = image_inputs.get_labels()
train, test = next(KFold(2).split(x, y))
x_train = x[train]
x_test = x[train]
y_train = y[train]
y_test = y[test]

# Build generators
generator = ResourcesGenerator(data_format='channels_first',
                               rescale=1./255)
train_generator = generator.flow_from_paths(x_train, y_train, color_mode='grayscale',
                                            class_mode='input',
                                            batch_size=batch_size)
validation_generator = generator.flow_from_paths(x_test, y_test, color_mode='grayscale',
                                                 class_mode='input',
                                                 batch_size=batch_size)

vae.fit_generator(train_generator,
                  steps_per_epoch=38585 // batch_size,
                  epochs=epochs,
                  validation_data=validation_generator,
                  validation_steps=5000 // batch_size)

# '''Example of VAE on MNIST dataset using MLP
#
# The VAE has a modular design. The encoder, decoder and VAE
# are 3 models that share weights. After training the VAE model,
# the encoder can be used to generate latent vectors.
# The decoder can be used to generate MNIST digits by sampling the
# latent vector from a Gaussian distribution with mean = 0 and std = 1.
#
# # Reference
#
# [1] Kingma, Diederik P., and Max Welling.
# "Auto-Encoding Variational Bayes."
# https://arxiv.org/abs/1312.6114
# '''
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# from keras.layers import Lambda, Input, Dense
# from keras.models import Model
# from keras.datasets import mnist
# from keras.losses import mse, binary_crossentropy
# from keras.utils import plot_model
# from keras import backend as K
#
# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os
#
#
# # reparameterization trick
# # instead of sampling from Q(z|X), sample epsilon = N(0,I)
# # z = z_mean + sqrt(var) * epsilon
# from sklearn.model_selection import KFold
#
# from toolbox.core.parameters import DermatologyDataset
# from toolbox.core.transforms import OrderedEncoder
#
#
# def sampling(args):
#     """Reparameterization trick by sampling from an isotropic unit Gaussian.
#
#     # Arguments
#         args (tensor): mean and log of variance of Q(z|X)
#
#     # Returns
#         z (tensor): sampled latent vector
#     """
#
#     z_mean, z_log_var = args
#     batch = K.shape(z_mean)[0]
#     dim = K.int_shape(z_mean)[1]
#     # by default, random_normal has mean = 0 and std = 1.0
#     epsilon = K.random_normal(shape=(batch, dim))
#     return z_mean + K.exp(0.5 * z_log_var) * epsilon
#
#
# def plot_results(models,
#                  data,
#                  batch_size=128,
#                  model_name="vae_mnist"):
#     """Plots labels and MNIST digits as a function of the 2D latent vector
#
#     # Arguments
#         models (tuple): encoder and decoder models
#         data (tuple): test data and label
#         batch_size (int): prediction batch size
#         model_name (string): which model is using this function
#     """
#
#     encoder, decoder = models
#     x_test, y_test = data
#     os.makedirs(model_name, exist_ok=True)
#
#     filename = os.path.join(model_name, "vae_mean.png")
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = encoder.predict(x_test,
#                                    batch_size=batch_size)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.savefig(filename)
#     plt.show()
#
#     filename = os.path.join(model_name, "digits_over_latent.png")
#     # display a 30x30 2D manifold of digits
#     n = 30
#     digit_size = 28
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[i * digit_size: (i + 1) * digit_size,
#                    j * digit_size: (j + 1) * digit_size] = digit
#
#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = (n - 1) * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap='Greys_r')
#     plt.savefig(filename)
#     plt.show()
#
# # Input patch
# image_inputs = DermatologyDataset.images(modality='Microscopy')
# image_inputs.set_encoders({'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant'])})
# patch_filter = {'Type': ['Patch']}
# image_inputs.set_filters(patch_filter)
#
# # Data from microscopy
# x = image_inputs.get_datas()
# y = image_inputs.get_labels()
# train, test = next(KFold(2).split(x, y))
# x_train = x[train]
# x_test = x[train]
# y_train = y[train]
# y_test = y[test]
#
# image_size = x_train.shape[1]
# original_dim = image_size * image_size
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
#
# # network parameters
# input_shape = (original_dim, )
# intermediate_dim = 512
# batch_size = 128
# latent_dim = 2
# epochs = 50
#
# # VAE model = encoder + decoder
# # build encoder model
# inputs = Input(shape=input_shape, name='encoder_input')
# x = Dense(intermediate_dim, activation='relu')(inputs)
# z_mean = Dense(latent_dim, name='z_mean')(x)
# z_log_var = Dense(latent_dim, name='z_log_var')(x)
#
# # use reparameterization trick to push the sampling out as input
# # note that "output_shape" isn't necessary with the TensorFlow backend
# z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
#
# # instantiate encoder model
# encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
#
# # build decoder model
# latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
# x = Dense(intermediate_dim, activation='relu')(latent_inputs)
# outputs = Dense(original_dim, activation='sigmoid')(x)
#
# # instantiate decoder model
# decoder = Model(latent_inputs, outputs, name='decoder')
# decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
#
# # instantiate VAE model
# outputs = decoder(encoder(inputs)[2])
# vae = Model(inputs, outputs, name='vae_mlp')
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     help_ = "Load h5 model trained weights"
#     parser.add_argument("-w", "--weights", help=help_)
#     help_ = "Use mse loss instead of binary cross entropy (default)"
#     parser.add_argument("-m",
#                         "--mse",
#                         help=help_, action='store_true')
#     args = parser.parse_args()
#     models = (encoder, decoder)
#     data = (x_test, y_test)
#
#     # VAE loss = mse_loss or xent_loss + kl_loss
#     if args.mse:
#         reconstruction_loss = mse(inputs, outputs)
#     else:
#         reconstruction_loss = binary_crossentropy(inputs,
#                                                   outputs)
#
#     reconstruction_loss *= original_dim
#     kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#     kl_loss = K.sum(kl_loss, axis=-1)
#     kl_loss *= -0.5
#     vae_loss = K.mean(reconstruction_loss + kl_loss)
#     vae.add_loss(vae_loss)
#     vae.compile(optimizer='adam')
#     vae.summary()
#     plot_model(vae,
#                to_file='vae_mlp.png',
#                show_shapes=True)
#
#     if args.weights:
#         vae.load_weights(args.weights)
#     else:
#         # train the autoencoder
#         vae.fit(x_train,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 validation_data=(x_test, None))
#         vae.save_weights('vae_mlp_mnist.h5')
#
#     plot_results(models,
#                  data,
#                  batch_size=batch_size,
#                  model_name="vae_mlp")

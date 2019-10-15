import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Flatten, Reshape
from keras.models import Model, Sequential
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import GroupKFold

from toolbox.core.generators import ResourcesGenerator
from toolbox.core.parameters import DermatologyDataset
from toolbox.core.transforms import OrderedEncoder

input_dim = (252, 252, 1)
original_dim = input_dim[0] * input_dim[1]
intermediate_dim = 256
latent_dim = 2
batch_size = 200
epochs = 50
epsilon_std = 1.0


def get_inputs():
    # Input patch
    image_inputs = DermatologyDataset.images(modality='Microscopy')
    image_inputs.set_encoders({'label': OrderedEncoder().fit(['Normal', 'Benign', 'Malignant'])})
    patch_filter = {'Type': ['Patch']}
    image_inputs.set_filters(patch_filter)

    # Data from microscopy_old
    x = image_inputs.get('datum')
    y = image_inputs.get('label')
    groups = image_inputs.get_groups()
    train, test = next(GroupKFold(2).split(x, y, groups))
    return x[train], x[test], y[train], y[test]


def get_generators(x_train, x_test, y_train, y_test):
    # Build generators
    generator = ResourcesGenerator(rescale=1. / 255)
    train_generator = generator.flow_from_paths(x_train, y_train, color_mode='grayscale', target_size=(252, 252),
                                                class_mode='both',
                                                batch_size=batch_size)
    validation_generator = generator.flow_from_paths(x_test, y_test, target_size=(252, 252),
                                                     color_mode='grayscale',
                                                     class_mode='both',
                                                     batch_size=batch_size, shuffle=False)
    return train_generator, validation_generator


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


class ClassificationConstraint(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(ClassificationConstraint, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)
        davies_bouldin_score(x_feat, y_true)
        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(original_dim, activation='sigmoid'),
    Reshape(input_dim)
])

x = Input(shape=input_dim)
flat = Flatten()(x)
h = Dense(intermediate_dim, activation='relu')(flat)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(x)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

x_pred = decoder(z)

x_train, x_test, y_train, y_test = get_inputs()
train_generator, validation_generator = get_generators(x_train, x_test, y_train, y_test)

vae = Model(inputs=[x, eps], outputs=[z, x_pred])

# vae.compile(optimizer='rmsprop', loss=nll)
# vae.fit_generator(train_generator,
#                   shuffle=True,
#                   epochs=epochs,
#                   validation_data=validation_generator)
vae.compile(optimizer='rmsprop', loss=[davies_bouldin_score, nll])
vae.fit_generator(train_generator,
                  shuffle=True,
                  epochs=epochs,
                  validation_data=validation_generator)

encoder = Model(x, z_mu)

# display a 2D plot of the digit classes in the latent space
z_test = encoder.predict_generator(validation_generator)
plt.figure(figsize=(6, 6))
plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
            alpha=.4, s=3 ** 2, cmap='viridis')
plt.colorbar()
plt.show()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)
x_decoded = decoder.predict(z_grid.reshape(n * n, 2))
x_decoded = x_decoded.reshape(n, n, digit_size, digit_size)

plt.figure(figsize=(10, 10))
plt.imshow(np.block(list(map(list, x_decoded))), cmap='gray')
plt.show()

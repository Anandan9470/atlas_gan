# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, merge
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.models import model_from_json
import keras.backend as K

from functools import partial

import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors

import numpy as np

path = "/media/anandan/3474068674064B56/CERN/Program/atlas_sim_gan/"

def get_data():

    data = np.loadtxt(path+"data/vectorized_cylindrical_230dim.csv", delimiter=',')

    while(True):

        batch = data[np.random.choice(data.shape[0], 128, replace=False)]
        batch = batch/65536
        yield batch

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((1, 230))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):

        self.img_dim = 230
        self.latent_dim = 20

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=(self.img_dim,))
        energy_labels = Input(shape=(1,))

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator([z_disc, energy_labels])

        # Discriminator determines validity of the real and fake images
        fake = self.critic([fake_img, energy_labels])
        valid = self.critic([real_img, energy_labels])

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_img, energy_labels])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, energy_labels, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator([z_gen, energy_labels])
        # Discriminator determines validity
        valid = self.critic([img, energy_labels])
        # Defines generator model
        self.generator_model = Model([z_gen, energy_labels], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        model.add(Dense(50, input_dim=self.latent_dim+1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(200))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.img_dim))
        #model.summary()

        noise = Input(shape=(self.latent_dim,))
        energy = Input(shape=(1,))
        merged = merge([noise, energy], mode='concat', concat_axis=-1)
        img = model(merged)

        return Model([noise, energy], img)

    def build_critic(self):

        model = Sequential()
        model.add(Dense(231, input_dim=self.img_dim+1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(231))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(231))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        #model.summary()

        img = Input(shape=(self.img_dim,))
        energy = Input(shape=(1,))
        merged = merge([img, energy], mode='concat',concat_axis=-1)
        validity = model(merged)

        return Model([img, energy], validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        data_gen = get_data()

        # Adversarial ground truths
        valid = np.random.uniform(-1.1, -0.9, (batch_size, 1))
        fake =  np.random.uniform(0.9, 1.1, (batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                X_train = data_gen.__next__()
                total_energy = np.log10(np.ones((batch_size, 1))*65536)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Sample generator input
                noise = np.random.normal(-1, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([X_train,total_energy,noise],
                                                          [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch([noise,total_energy], valid)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(-1, 1, (r*c, self.latent_dim))
        total_energy = np.log10(np.ones((r*c, 1))*65536)
        gen_imgs = self.generator.predict([noise, total_energy])

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):

                img = np.reshape(gen_imgs[cnt], newshape=(10,23), order='F')

                num_levels = 20
                vmin, vmax = img.min(), img.max()
                midpoint = 0
                levels = np.linspace(vmin, vmax, num_levels)
                midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)
                vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.5, 1])
                colors = plt.cm.seismic(vals)
                cmap, norm = from_levels_and_colors(levels, colors)

                im = axs[i,j].imshow(img, cmap=cmap, norm=norm, interpolation='none')
                fig.colorbar(im, ax=axs[i,j])
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig("images/sample_%d.png" % epoch)
        plt.close()

    def save_model_to_disk(self):

        noise = np.random.normal(-1, 1, (10000, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        np.savetxt("./saved_model/samples.csv", gen_imgs, delimiter=',')

        # serialize model to JSON
        generator_model_json = self.generator.to_json()
        with open("./saved_model/generator_model.json", "w") as json_file:
            json_file.write(generator_model_json)
        # serialize weights to HDF5
        self.generator.save_weights("./saved_model/model.h5")
        print("Saved model to disk")

    def load_model_from_disk(self):

        # load json and create model
        json_file = open("./saved_model/generator_model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("./saved_model/model.h5")
        print("Loaded model from disk")

if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=10000, batch_size=128, sample_interval=100)
    wgan.save_model_to_disk()
    wgan.load_model_from_disk()



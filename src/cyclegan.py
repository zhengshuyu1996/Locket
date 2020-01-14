from __future__ import print_function, division
import scipy
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONHASHSEED'] = '0'

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras import Sequential, Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
from utils import DataLoader
import tensorflow as tf 

class Config():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64


class CycleGAN():
    def __init__(self, config):
        # Input shape
        self.config = config
        self.img_shape = (self.config.img_rows, self.config.img_cols, self.config.channels)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.config.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)
        
        self.build_summary()
        print('End building CycleGAN...')

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            # d = InstanceNormalization()(d)
            d = BatchNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            # u = InstanceNormalization()(u)
            u = BatchNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.config.gf)
        d2 = conv2d(d1, self.config.gf*2)
        d3 = conv2d(d2, self.config.gf*4)
        d4 = conv2d(d3, self.config.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.config.gf*4)
        u2 = deconv2d(u1, d2, self.config.gf*2)
        u3 = deconv2d(u2, d1, self.config.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.config.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                # d = InstanceNormalization()(d)
                d = BatchNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.config.df, normalization=False)
        d2 = d_layer(d1, self.config.df*2)
        d3 = d_layer(d2, self.config.df*4)
        d4 = d_layer(d3, self.config.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def build_summary(self):
        # Start tensorboard 
        self.sess = tf.Session()
        os.makedirs('logs/', exist_ok=True)
        logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

        self.writer = tf.summary.FileWriter(logdir, self.sess.graph)

        self.D_LOSS = tf.placeholder(tf.float32, [])
        self.D_ACC = tf.placeholder(tf.float32, [])
        self.G_LOSS = tf.placeholder(tf.float32, [])
        tf.summary.scalar("D_LOSS", self.D_LOSS)
        tf.summary.scalar("D_ACC", self.D_ACC)
        tf.summary.scalar("G_LOSS", self.G_LOSS)
        
        self.merged = tf.summary.merge_all()
        # End tensorboard

    def train(self, epochs, AB_train, AB_val, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        print('Start training...')
        step = 0
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(AB_train.get_batch(batch_size)):
                step += 1
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time


                summary = self.sess.run(self.merged, feed_dict={self.D_LOSS: d_loss[0], self.D_ACC: 100*d_loss[1], self.G_LOSS: g_loss[0]}) 
                self.writer.add_summary(summary, step)


                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, AB_train.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(AB_val, epoch, batch_i)

        # self.save_models()

    def sample_images(self, AB_val, epoch, batch_i):
        os.makedirs('images/', exist_ok=True)
        r, c = 2, 3

        imgs_A, imgs_B = AB_val
        # imgs_A, imgs_B = next(AB_val.get_batch(batch_size=1))

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B], axis=1)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        cnt = 0
        for k in range(len(imgs_A)):
            fig, axs = plt.subplots(r, c)
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt])
                    axs[i,j].set_title(titles[j])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("images/%d_%d_%d.png" % (epoch, batch_i, k))
            plt.close()

    def save_models(self, save_path):
        save_model(self.g_AB, save_path+'g_AB')
        save_model(self.g_BA, save_path+'g_BA')
        save_model(self.d_B, save_path+'d_B')
        save_model(self.d_A, save_path+'d_A')



if __name__ == '__main__':
    config = Config()
    gan = CycleGAN(config)
    # Configure data loader
    dir_A = '../datasets/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/training_set/painting/'
    dir_B = '../datasets/matting_samples/clip/'
    AB_train = DataLoader(dir_A, dir_B, img_res=(256,256))
    AB_val = DataLoader(dir_A, dir_B, is_testing=True, img_res=(256,256))
    sample_num = 10
    A_sample = AB_val.get_dataset_A(sample_num)
    B_sample = AB_val.get_dataset_B(sample_num)

    gan.train(AB_train=AB_train, AB_val=[A_sample, B_sample], epochs=200, batch_size=8, sample_interval=200)
    gan.save_models('../models/')



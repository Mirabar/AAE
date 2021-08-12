import argparse
import tensorflow as tf
import utils
import models
import numpy as np
import os
import pickle
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

tf.executing_eagerly()


def user_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', help='Number of training epochs', type=int, required=False, default=20)
    parser.add_argument('-bs', '--batch_size', help='Number of samples in a mini-batch', type=int, required=False,
                        default=64)
    parser.add_argument('-tc', '--training_checkpoins', help='checkpoint directory',
                        type=str, required=False, default='training_checkpoints')
    parser.add_argument('-d', '--dest_dir', help='destination directory',
                        type=str, required=False, default='.')
    parser.add_argument('-r', '--restore_chkpt', help='continue with trained model',
                        type=int, required=False, default=0, choices=[0, 1])
    parser.add_argument('-m', '--mode', help='whether to perform train or test',
                        type=str, required=False, default='train')

    args = parser.parse_args()
    arguments = vars(args)

    return arguments


class VAA():

    def __init__(self, real_data, n_classes, epochs, batch_size, checkpoint_prefix, save_dir):

        self.d_iter = 5
        self.save_dir = save_dir
        self.real_data = real_data
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_prefix = checkpoint_prefix
        self.generator = models.aae_generator(in_sh=(30,))
        self.discriminator = models.aae_discriminator(in_sh=(30,))
        self.encoder = models.encoder(out_dim=30)
        self.vae_history = []
        self.g_history = []
        self.d_history = []
        self.vae_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.d_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.g_opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.checkpoint = tf.train.Checkpoint(vae_optimizer=self.vae_opt,
                                              discriminator_optimizer=self.d_opt,
                                              generator_optimizer=self.g_opt,
                                              generator=self.generator,
                                              discriminator=self.discriminator,
                                              encoder=self.encoder)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_prefix, max_to_keep=2)
        self.scale = 10

    def train_step(self, x):

        z = utils.z_sampler(x.shape[0], concat=True)

        with tf.GradientTape() as vae_tape, tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:

            z_ = self.encoder(x, training=True)
            x_ = self.generator(z_, training=True)

            real_pred = self.discriminator(z, training=True)
            fake_pred = self.discriminator(z_, training=True)

            if self.iter == self.d_iter:
                self.g_loss = -tf.reduce_mean(fake_pred)

            self.d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + self.calc_penalty(z, z_)
            self.vae_loss = tf.reduce_mean((x - x_) ** 2)

        if self.iter == self.d_iter:
            g_grad = g_tape.gradient(self.g_loss, self.encoder.trainable_variables)
            self.g_opt.apply_gradients(zip(g_grad, self.encoder.trainable_variables))
            self.iter = 0
        else:
            d_grad = d_tape.gradient(self.d_loss, self.discriminator.trainable_variables)
            vae_grad = vae_tape.gradient(self.vae_loss, self.encoder.trainable_variables +
                                         self.generator.trainable_variables)
            self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
            self.vae_opt.apply_gradients(zip(vae_grad,
                                             self.encoder.trainable_variables + self.generator.trainable_variables))

    def train(self, restore=False):

        if restore:
            self.checkpoint.restore(self.manager.latest_checkpoint)

        for epoch in range(self.epochs):

            tf.print(f'Epoch {epoch + 1}')
            self.iter = 0
            for batch in self.real_data:
                self.train_step(batch)
                self.iter += 1
            if (epoch + 1) % 3 == 0:
                tf.print(f'vae loss {float(self.vae_loss)}')
                tf.print(f'g loss {float(self.g_loss)}')
                tf.print(f'discriminator loss {float(self.d_loss)}')
                self.manager.save()

            self.vae_history.append(self.vae_loss)
            self.d_history.append(self.d_loss)

        return {'vae_loss': self.vae_history, 'd_loss': self.d_history}

    def calc_penalty(self, real_img, fake_img):

        eps = tf.random.uniform([real_img.shape[0], 1], 0.0, 0.1)
        x_hat = eps * real_img + (1 - eps) * fake_img
        with tf.GradientTape() as penalty_tape:
            penalty_tape.watch(x_hat)
            pred = self.discriminator(x_hat)

        ddx = penalty_tape.gradient(pred, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1) + 1e-8)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * self.scale)

        return ddx

    def test(self, clip_val=0.6):

        self.clip_val = clip_val

        self.checkpoint.restore(self.manager.latest_checkpoint)
        y_list = []
        z_list = []

        for X, y in self.real_data:
            z = self.encoder(X, training=False)
            z_list.append(z)
            y_list.append(y)

        all_z = np.concatenate(z_list, axis=0)
        all_y = np.concatenate(y_list, axis=0)

        utils.cluster_viz(all_z, all_y, self.save_dir)

        acc_cn, acc_n, ari_n, nmi_n = utils.cluster_latent(all_z[:, :-self.n_classes], all_y)
        print(f'zn cluster eval- ARI: {ari_n}, NMI: {nmi_n}, ACC: {acc_n}, class ACC: {acc_cn}')
        acc_c, acc, ari, nmi = utils.cluster_latent(all_z, all_y)
        print(f'cluster eval- ARI: {ari}, NMI: {nmi}, ACC: {acc}, class ACC: {acc_c}')


def main(args):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    if args['mode'] == 'train':
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = train_images / 255
        BUFFER_SIZE = 60000
        BATCH_SIZE = args['batch_size']
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        if not os.path.exists(args['dest_dir']):
            os.mkdir(args['dest_dir'])

        checkpoint_dir = args['dest_dir'] + '/' + args['training_checkpoins']
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        vaa = VAA(real_data=train_dataset,
                  n_classes=len(np.unique(train_labels)),
                  epochs=args['epochs'],
                  batch_size=BATCH_SIZE,
                  checkpoint_prefix=checkpoint_prefix, save_dir=args['dest_dir'])

        history = vaa.train(restore=args['restore_chkpt'])
        pickle.dump(history, open(args['dest_dir'] + '/history.pkl', 'wb'))

    elif args['mode'] == 'test':

        checkpoint_dir = args['dest_dir'] + '/' + args['training_checkpoins']
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
        test_images = test_images / 255
        BUFFER_SIZE = 60000
        BATCH_SIZE = args['batch_size']
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).shuffle(BUFFER_SIZE).batch(
            BATCH_SIZE)

        vaa = VAA(real_data=test_dataset,
                  n_classes=len(np.unique(train_labels)),
                  epochs=args['epochs'],
                  batch_size=BATCH_SIZE,
                  checkpoint_prefix=checkpoint_prefix, save_dir=args['dest_dir'])

        vaa.test()


if __name__ == '__main__':
    args = user_input()
    main(args)

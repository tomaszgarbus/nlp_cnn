# https://arxiv.org/pdf/1509.01626.pdf
import os
import random
from typing import Optional

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from chart_generator import ChartGenerator
from models.model import Model


class CharCNN(Model):
    def __init__(self,
                 session,
                 learning_rate=0.01,
                 input_length=1014,
                 regularization_rate=0.1,
                 alphabet_size=70,
                 ckpt_file: Optional[str] = None):
        Model.__init__(self)

        self.sess = session
        self.learning_rate = learning_rate
        self.input_length = input_length
        self.alphabet_size = alphabet_size
        self.reg_rate = regularization_rate
        if ckpt_file:
            self.ckpt_file = ckpt_file
        else:
            ckpt_dir = os.path.join('tmp', 'models', str(self))
            os.makedirs(ckpt_dir, exist_ok=True)
            self.ckpt_file = os.path.join(ckpt_dir, 'char_cnn.ckpt')

        self._build_model()
        self._add_training_objectives()
        self._load_or_init()

    def __str__(self):
        return "CharCNN"

    def _build_model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.input_length, self.alphabet_size))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=())
        tf.assert_equal(tf.shape(self.x)[0], tf.shape(self.y)[0])

        signal = self.x

        # Convolutional layers
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_rate)
        # initializer = tf.initializers.random_normal(mean=0.0, stddev=0.02)
        # conv_features = [256] * 6
        # conv_kernels = [7, 7, 3, 3, 3, 3]
        # conv_pools = [3, 3, None, None, None, 3]
        conv_features = [32] * 3
        conv_kernels = [7, 3, 3]
        conv_pools = [3, 3, 3]
        for features, kernel, pooling in zip(conv_features, conv_kernels, conv_pools):
            signal = tf.layers.conv1d(inputs=signal, filters=features, kernel_size=kernel,
                                      kernel_regularizer=self.regularizer)
            if pooling is not None:
                # Non-overlapping pooling thus strides=pool_size.
                signal = tf.layers.max_pooling1d(inputs=signal, pool_size=pooling, strides=pooling)

            # Non-linearity
            signal = tf.nn.leaky_relu(signal)

        print(signal.shape)
        # Fully connected layers
        signal = tf.layers.flatten(signal)
        fully_connected = [16, 1]  # The paper says 1024 but seems like an overkill for MR dataset.
        for i, units in enumerate(fully_connected):
            if i > 0:
                signal = tf.nn.leaky_relu(signal)
            signal = tf.layers.dropout(inputs=signal, rate=0.5, training=self.tf_is_traing_pl)
            signal = tf.layers.dense(inputs=signal, units=units,
                                     kernel_regularizer=self.regularizer)

        self.output = signal
        self.sigmoid_output = tf.nn.sigmoid(self.output)

    def _add_training_objectives(self) -> None:
        # Adds regularization.
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        self.loss = tf.losses.log_loss(self.y, self.sigmoid_output) #+ reg_term
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9).\
            minimize(loss=self.loss)
        self.accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(self.sigmoid_output) - self.y))

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size=128, nb_epochs=20, shuffle: bool = True, val_x=None,
            val_y=None, ckpt_freq=100) -> None:
        chart = ChartGenerator()
        print(x.shape, y.shape)
        assert x.shape[0] == y.shape[0]
        iters_per_epoch = x.shape[0] // batch_size + (nb_epochs % x.shape[0] > 0)
        for epoch_no in range(nb_epochs):
            if shuffle:
                order = list(range(len(x)))
                random.shuffle(order)
                x = x[order]
                y = y[order]
            losses = []
            accs = []
            stds = []
            trange = tqdm(range(iters_per_epoch), desc="Epoch %d/%d" % (epoch_no+1, nb_epochs))
            for iter in trange:
                batch_x = x[iter * batch_size: (iter + 1) * batch_size]
                batch_y = y[iter * batch_size: (iter + 1) * batch_size]
                loss, acc, sigmo_outs, _ = self.sess.run([self.loss,
                                                          self.accuracy,
                                                          self.sigmoid_output,
                                                          self.optimizer],
                                                         feed_dict={self.x: batch_x, self.y: batch_y})
                losses.append(loss)
                accs.append(acc)
                stds.append(np.std(sigmo_outs))
            print("loss: " + str(loss) + " acc: " + str(acc) + " mean_loss: " + str(np.mean(losses)) + " mean_acc: " +
                  str(np.mean(accs)) + " outs_std: " + str(np.std(sigmo_outs)))
            if val_x is not None and val_y is not None:
                val_acc = self.evaluate(val_x, val_y, batch_size)
                print("val_acc : " + str(val_acc))
                chart.log_values(epoch_no+1, {'train_acc': np.mean(accs),
                                              'val_acc': val_acc,
                                              'loss': np.mean(losses),
                                              'outs_std': np.mean(stds)})
            if epoch_no > 0 and epoch_no % ckpt_freq == 0:
                self.save()
        self.save()
        chart.show_chart(title=str(self))
        chart.save_chart(fname='tmp/charts/' + str(self) + '.jpg', title=str(self))

    def predict(self, x: np.ndarray, batch_size=128, round_value=True) -> np.ndarray:
        """
        :param x:
        :param batch_size:
        :param round_value: Whether values should be round to {0, 1}. Set to false for ensembling.
        :return:
        """
        predictions = []
        iters = x.shape[0] // batch_size + (x.shape[0] % batch_size > 0)
        for iter in range(iters):
            batch_x = x[iter * batch_size: (iter + 1) * batch_size]
            outs = self.sess.run([self.sigmoid_output], feed_dict={self.x: batch_x, self.tf_is_traing_pl: False})[0]
            if round_value:
                outs = np.round(outs)
            predictions.append(outs)
        predictions = np.concatenate(predictions, axis=0)
        return predictions
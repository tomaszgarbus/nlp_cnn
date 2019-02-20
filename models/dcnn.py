# https://arxiv.org/pdf/1404.2188.pdf
import os
from typing import Optional

import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm

from chart_generator import ChartGenerator
from models.model import Model


class DCNN(Model):
    def __init__(self,
                 sess,
                 initial_embeddings,
                 embeddings_dim=50,
                 learning_rate=0.1,
                 static=False,
                 ckpt_file: Optional[str] = None):
        Model.__init__(self)

        self.sess = sess
        self.initial_embeddings = initial_embeddings
        self.embeddings_dim = embeddings_dim
        self.learning_rate = learning_rate
        self.static = static
        self.top_k = 4
        if ckpt_file:
            self.ckpt_file = ckpt_file
        else:
            ckpt_dir = os.path.join('tmp', 'models', str(self))
            os.makedirs(ckpt_dir, exist_ok=True)
            self.ckpt_file = os.path.join(ckpt_dir, 'dcnn.ckpt')

        self._build_model()
        self._add_training_objectives()
        self._load_or_init()

    def __str__(self):
        desc = ["DCNN",
                "dimension: " + str(self.embeddings_dim),
                "static" if self.static else "non-static",
                ]
        return ", ".join(desc)

    def _build_model(self):
        self.x = tf.placeholder(dtype=tf.int32, shape=(None, None))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        # Maps integers to embeddings.
        self.embeddings = tf.Variable(initial_value=self.initial_embeddings, trainable=not self.static,
                                      expected_shape=(None, self.embeddings_dim))
        signal = tf.gather(self.embeddings, self.x)

        # First convolutional layer.
        # Padding to make it a wide convolution.
        paddings = tf.constant([[0, 0], [6, 6], [0, 0]])
        signal = tf.pad(signal, paddings, mode='CONSTANT')
        signal = tf.layers.conv1d(signal,
                                  filters=6,
                                  kernel_size=7,
                                  padding='valid')

        # First dynamic k-max pooling.
        s = tf.cast(tf.shape(self.x)[1], dtype=tf.float32)
        # k1 = tf.maximum(self.top_k, tf.cast(tf.ceil(1/2 * s), tf.int32))
        k1 = 8  # TODO: actual dynamic k
        # Selects the most activated features.
        excitements = tf.reduce_mean(signal, axis=2)
        _, indices = tf.nn.top_k(excitements, k1, sorted=False)
        signal = tf.gather_nd(signal, indices)

        # Non-linearity
        signal = tf.nn.sigmoid(signal)

        # Second convolutional layer.
        paddings = tf.constant([[0, 0], [4, 4], [0, 0]])
        signal = tf.pad(signal, paddings, mode='CONSTANT')
        signal = tf.layers.conv1d(signal,
                                  filters=14,
                                  kernel_size=5,
                                  padding='valid')

        # Second dynamic k-max pooling.
        # Selects the most activated features.
        excitements = tf.reduce_mean(signal, axis=2)
        _, indices = tf.nn.top_k(excitements, self.top_k, sorted=False)
        signal = tf.gather_nd(signal, indices)

        # TODO: folding
        # TODO: do as in paper
        signal = tf.reduce_mean(signal, axis=1)
        dense = tf.layers.dense(signal, units=1)
        self.output = tf.reduce_mean(dense)
        # self.output = tf.layers.dense(signal, units=1)
        self.sigmoid_output = tf.sigmoid(self.output)

    def _add_training_objectives(self) -> None:
        # Adds regularization.
        self.loss = tf.losses.log_loss(self.y, self.sigmoid_output)
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
        self.accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(self.sigmoid_output) - self.y))

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size=20, nb_epochs=20, shuffle: bool = True, val_x=None,
            val_y=None, ckpt_freq=100) -> None:
        chart = ChartGenerator()
        print(x.shape, y.shape)
        assert x.shape[0] == y.shape[0]
        iters_per_epoch = (x.shape[0] // batch_size) + (x.shape[0] % batch_size != 0)
        for epoch_no in range(nb_epochs):
            if shuffle:
                order = list(range(len(x)))
                random.shuffle(order)
                x = x[order]
                y = y[order]
            losses = []
            accs = []
            trange = tqdm(range(iters_per_epoch), desc="Epoch %d/%d" % (epoch_no + 1, nb_epochs))
            for iter in trange:
                feed_x = x[iter * batch_size : (iter+1) * batch_size]
                feed_y = y[iter * batch_size: (iter + 1) * batch_size]
                loss, acc, sigmo_outs, _ = self.sess.run([self.loss,
                                                          self.accuracy,
                                                          self.sigmoid_output,
                                                          self.optimizer],
                                                         feed_dict={self.x: feed_x, self.y: feed_y})
                losses.append(loss)
                accs.append(acc)
                # print(loss, acc)
                # print(sigmo_outs)
            print("loss: " + str(loss) + " acc: " + str(acc) + " mean_loss: " + str(np.mean(losses)) + " mean_acc: " +
                  str(np.mean(accs)))
            if val_x is not None and val_y is not None:
                val_acc = self.evaluate(val_x, val_y)
                print("val_acc : " + str(val_acc))
                chart.log_values(epoch_no+1, {'train_acc': np.mean(accs),
                                              'test_acc': val_acc,
                                              'loss': np.mean(losses)})
            if epoch_no > 0 and epoch_no % ckpt_freq == 0:
                self.save()
        self.save()
        chart.show_chart(title=str(self))
        chart.save_chart(fname='tmp/charts/' + str(self) + '.jpg', title=str(self))

    def predict(self, x: np.ndarray, batch_size=50, round_value=True) -> np.ndarray:
        """
        :param x:
        :param batch_size: IGNORED (TODO: unignore)
        :param round_value: Whether values should be round to {0, 1}. Set to false for ensembling.
        :return:
        """
        predictions = []
        iters = x.shape[0]
        for iter in range(iters):
            feed_x = np.array([x[iter]])
            feed_x = np.array([feed_x[feed_x.nonzero()]])
            outs = self.sess.run([self.sigmoid_output], feed_dict={self.x: feed_x})[0]
            if round_value:
                outs = np.round(outs)
            predictions.append([outs])
        predictions = np.concatenate(predictions, axis=0)
        print(predictions, predictions.mean())
        return predictions
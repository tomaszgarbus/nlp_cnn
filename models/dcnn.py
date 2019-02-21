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
                 regularization=0.1,
                 embeddings_dim=50,
                 learning_rate=0.1,
                 h1=7,
                 h2=5,
                 static=False,
                 ckpt_file: Optional[str] = None):
        Model.__init__(self)

        self.sess = sess
        self.initial_embeddings = initial_embeddings
        self.embeddings_dim = embeddings_dim
        self.learning_rate = learning_rate
        self.static = static
        self.regularization_rate = regularization
        self.top_k = 3
        self.h1 = h1
        self.h2 = h2
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
        self.k1 = tf.placeholder(dtype=tf.int32, shape=())
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=())
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularization_rate)

        # Maps integers to embeddings.
        self.embeddings = tf.Variable(initial_value=self.initial_embeddings, trainable=not self.static,
                                      expected_shape=(None, self.embeddings_dim))
        signal = tf.gather(self.embeddings, self.x)

        # First convolutional layer.
        # Padding to make it a wide convolution.
        paddings = tf.constant([[0, 0], [self.h1-1, self.h1-1], [0, 0]])
        signal = tf.pad(signal, paddings, mode='CONSTANT')
        signal = tf.layers.conv1d(signal,
                                  filters=6,
                                  kernel_size=self.h1,
                                  kernel_regularizer=self.regularizer,
                                  padding='valid')

        def _add_dynamic_k_max_pooling(signal, k):
            # Selects the most activated features.
            excitements = tf.reduce_mean(signal, axis=2)
            _, indices = tf.nn.top_k(excitements, k, sorted=False)
            rows_nos = tf.range(start=0, limit=tf.cast(tf.shape(signal)[0], tf.int32))
            indices = tf.map_fn(fn=lambda row_no: tf.map_fn(fn=lambda a: tf.convert_to_tensor([row_no, a]),
                                                            elems=indices[row_no],
                                                            dtype=tf.int32),
                                elems=rows_nos, dtype=tf.int32)
            signal = tf.gather_nd(signal, indices)
            return signal

        # First dynamic k-max pooling.
        k1 = self.k1  # TODO: actual dynamic k
        signal = _add_dynamic_k_max_pooling(signal, k1)

        # Non-linearity
        signal = tf.nn.tanh(signal)
        # Dropout
        signal = tf.layers.dropout(inputs=signal, rate=0.5, training=self.tf_is_traing_pl)

        # Second convolutional layer.
        paddings = tf.constant([[0, 0], [self.h2-1, self.h2-1], [0, 0]])
        signal = tf.pad(signal, paddings, mode='CONSTANT')
        signal = tf.layers.conv1d(signal,
                                  filters=15,
                                  kernel_size=self.h2,
                                  kernel_regularizer=self.regularizer,
                                  padding='valid')

        # Second dynamic k-max pooling.
        signal = _add_dynamic_k_max_pooling(signal, self.top_k)

        # TODO: folding
        # TODO: do as in paper
        signal = tf.layers.flatten(signal)
        dense = tf.layers.dense(signal, units=1, kernel_regularizer=self.regularizer)
        self.output = dense
        self.sigmoid_output = tf.sigmoid(self.output)

    def _add_training_objectives(self) -> None:
        # Adds regularization.
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
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
                feed_x = x[iter * batch_size: (iter+1) * batch_size]
                feed_y = y[iter * batch_size: (iter + 1) * batch_size]
                loss, acc, sigmo_outs, _ = self.sess.run([self.loss,
                                                          self.accuracy,
                                                          self.sigmoid_output,
                                                          self.optimizer],
                                                         feed_dict={self.x: feed_x,
                                                                    self.y: feed_y,
                                                                    self.k1: 6})
                losses.append(loss)
                accs.append(acc)
                # print(loss, acc)
                # print(sigmo_outs)
            print("loss: " + str(loss) + " acc: " + str(acc) + " mean_loss: " + str(np.mean(losses)) + " mean_acc: " +
                  str(np.mean(accs)) + " outs_std: " + str(np.std(sigmo_outs)) + " outs_mean: " + str(np.mean(sigmo_outs)))
            if val_x is not None and val_y is not None:
                val_acc = self.evaluate(val_x, val_y)
                print("val_acc : " + str(val_acc))
                chart.log_values(epoch_no+1, {'train_acc': np.mean(accs),
                                              'val_acc': val_acc,
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
        iters = x.shape[0] // batch_size + (x.shape[0] % batch_size != 0)
        for iter in range(iters):
            feed_x = np.array(x[iter * batch_size: (iter+1) * batch_size])
            outs = self.sess.run([self.sigmoid_output], feed_dict={self.x: feed_x, self.k1: 6, self.tf_is_traing_pl: False})[0]
            if round_value:
                outs = np.round(outs)
            predictions.append(outs)
        predictions = np.concatenate(predictions, axis=0)
        # print(predictions, predictions.mean())
        return predictions
# https://arxiv.org/pdf/1408.5882.pdf
import random
from typing import List, Optional

import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

from chart_generator import ChartGenerator
from models.model import Model

VOCABULARY_SIZE = 400000


class YoonKimCnnStatic(Model):
    def __init__(self,
                 session,
                 initial_embeddings: Optional[np.ndarray],
                 static: str='non-static',
                 input_length=20,
                 embeddings_dim=50,
                 learning_rate=1,
                 num_filters=100,
                 regularization_rate=0.01,
                 ckpt_file: Optional[str] = None):
        Model.__init__(self)

        self.sess = session
        self.learning_rate = learning_rate
        self.input_length = input_length
        assert static in ['non-static', 'static', 'rand', 'both']
        self.static = static
        self.embeddings_dim = embeddings_dim
        if initial_embeddings is None:
            assert static == 'rand'
            self.initial_embeddings = np.random.rand(VOCABULARY_SIZE, self.embeddings_dim).astype(np.float32)
        else:
            self.initial_embeddings = initial_embeddings
        self.num_filters = num_filters
        self.regularization_rate = regularization_rate
        if ckpt_file:
            self.ckpt_file = ckpt_file
        else:
            ckpt_dir = os.path.join('tmp', 'models', str(self))
            os.makedirs(ckpt_dir, exist_ok=True)
            self.ckpt_file = os.path.join(ckpt_dir, 'yoon_kim.ckpt')

        self._build_model()
        self._add_training_objectives()
        self._load_or_init()

    def __str__(self):
        desc = ["Kim Yoon CNN",
                "filters: " + str(self.num_filters),
                "dimension: " + str(self.embeddings_dim),
                self.static,
                "regularization: " + str(self.regularization_rate)
                ]
        return ", ".join(desc)

    def _build_model(self) -> None:
        self.x = tf.placeholder(tf.int32, shape=(None, self.input_length))
        self.y = tf.placeholder(tf.float32, shape=(None, 1,))
        self.tf_is_traing_pl = tf.placeholder_with_default(True, shape=())
        if self.static == 'both':
            self.embeddings_static = tf.Variable(initial_value=self.initial_embeddings, trainable=False,
                                                 expected_shape=(None, self.embeddings_dim))
            self.embeddings_nonstatic = tf.Variable(initial_value=self.initial_embeddings, trainable=True,
                                                    expected_shape=(None, self.embeddings_dim))
            signal_static = tf.gather(self.embeddings_static, self.x)
            signal_nonstatic = tf.gather(self.embeddings_nonstatic, self.x)
            signal = tf.concat([signal_static, signal_nonstatic], axis=2)
            assert signal.shape[1:] == (self.input_length, self.embeddings_dim * 2)
        else:
            self.embeddings = tf.Variable(initial_value=self.initial_embeddings, trainable=(self.static != 'static'),
                                          expected_shape=(None, self.embeddings_dim))
            signal = tf.gather(self.embeddings, self.x)
            assert signal.shape[1:] == (self.input_length, self.embeddings_dim)

        # Feature maps layer.
        self.feature_maps_list = []
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularization_rate)
        for filter_size in [3, 4, 5]:
            initializer = tf.initializers.random_normal(mean=0.0, stddev=np.sqrt(2/(self.input_length * self.embeddings_dim)))
            self.feature_maps_list.append(tf.layers.conv1d(inputs=signal,
                                                           filters=self.num_filters,
                                                           kernel_size=filter_size,
                                                           use_bias=True,
                                                           kernel_initializer=initializer,
                                                           kernel_regularizer=self.regularizer,
                                                           padding='same',
                                                           activation=tf.nn.tanh))
        self.feature_maps = tf.concat(self.feature_maps_list, axis=2)

        # Max pooling layer.
        max_pool_over_time = tf.layers.max_pooling1d(inputs=self.feature_maps, pool_size=self.input_length, strides=1)
        dropout = tf.layers.dropout(inputs=max_pool_over_time, rate=0.5, training=self.tf_is_traing_pl)

        # Dense layer.
        self.dense = tf.layers.dense(inputs=dropout, units=1, kernel_regularizer=self.regularizer)
        flatten = tf.layers.flatten(self.dense)
        # self.output = tf.nn.sigmoid(flatten)
        self.output = flatten
        self.sigmoid_output = tf.sigmoid(self.output)

    def _add_training_objectives(self) -> None:
        # Adds regularization.
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        self.loss = tf.losses.log_loss(self.y, self.sigmoid_output) + reg_term
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
        self.accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(self.sigmoid_output) - self.y))

    def predict(self, x: np.ndarray, batch_size=50, round_value=True) -> np.ndarray:
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

    def find_most_exciting(self, x: np.ndarray, int_to_word: List[str]) -> None:
        """
        :param x: Dataset to extract most exciting fragments from.
        :param int_to_word: Mapping from integers to words.
        """
        # Iterates all filter sizes.
        for map_idx, filter_size in [(0, 3), (1, 4), (2, 5)]:
            # Fetches the feature maps.
            excitements = self.sess.run(self.feature_maps_list, feed_dict={self.x: x, self.tf_is_traing_pl: False})
            most_exciting = [[] for _ in range(self.num_filters)]
            for filter_no in range(self.num_filters):
                # For a given filter, fetches 5 most excited values.
                tmp_ex = []
                for sample_no in range(excitements[map_idx].shape[0]):
                    for pos in range(self.input_length):
                        tmp_ex.append((excitements[map_idx][sample_no][pos][filter_no], sample_no, pos))
                most_exciting[filter_no] = sorted(tmp_ex, reverse=True)[:5]
            # For each filter, prints top 5 values as words lists, with raw excitement value.
            for filter_no in range(self.num_filters):
                print("Filter %d from those of size %d" % (filter_no, filter_size))
                for val, sample, pos in most_exciting[filter_no]:
                    words = []
                    for i in range(pos - filter_size//2, pos + filter_size // 2 + filter_size % 2):
                        if 0 <= i < self.input_length:
                            words.append(int_to_word[x[sample][i]])
                        else:
                            words.append("")
                    print(val, " ".join(words))

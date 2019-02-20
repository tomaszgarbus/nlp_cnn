import numpy as np
import tensorflow as tf


class Model:
    sess = None
    ckpt_file = None

    def __init__(self):
        pass

    def __str__(self):
        raise NotImplementedError()

    def _load_or_init(self):
        saver = tf.train.Saver()
        try:
            saver.restore(self.sess, save_path=self.ckpt_file)
        except tf.errors.NotFoundError as e:
            self.sess.run(tf.global_variables_initializer())

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path=self.ckpt_file)

    def predict(self, x: np.ndarray, batch_size=50, round_value=True) -> np.ndarray:
        raise NotImplementedError()

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size=50) -> float:
        """
        :return: Accuracy
        """
        predictions = self.predict(x, batch_size=batch_size, round_value=True)
        acc = 1.0 - np.mean(np.abs(predictions - y))
        return acc
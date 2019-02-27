import numpy as np
import tensorflow as tf

from chart_generator import ChartGenerator
import random
from tqdm import tqdm


class Model:
    sess = None
    ckpt_file = None
    loss = None
    x = None
    y = None
    accuracy = None
    sigmoid_output = None
    optimizer = None

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

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size=50, nb_epochs=20, shuffle: bool = True, val_x=None,
            val_y=None, ckpt_freq=100) -> None:
        chart = ChartGenerator()
        print(x.shape, y.shape)
        assert x.shape[0] == y.shape[0]
        iters_per_epoch = x.shape[0] // batch_size + (x.shape[0] % nb_epochs > 0)
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
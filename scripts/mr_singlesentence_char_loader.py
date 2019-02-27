import numpy as np
from typing import Tuple
import os

DATA_PATH = 'data/rt-polaritydata'
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


class MovieReviewSingleSentenceDatasetCharLoader:  # TODO: shorten this abomination
    def __init__(self, alphabet: str, input_length=1014):
        self.input_length = input_length
        self.alphabet = alphabet

    def _file_to_onehot_chars(self, fname: str) -> np.ndarray:
        with open(fname, 'r') as file:
            lines = file.read().strip().split('\n')
        ret = np.zeros(shape=(len(lines), self.input_length, len(self.alphabet)), dtype=np.int32)
        for i, line in enumerate(lines):
            ret[i] = self._line_to_onehot_chars(line)
        return ret

    def _line_to_onehot_chars(self, line: str) -> np.ndarray:
        ret = np.zeros(shape=(1, self.input_length, len(self.alphabet)), dtype=np.int32)
        for i, char in enumerate(line):
            if i >= self.input_length:
                break
            if char in self.alphabet:
                ret[0, i, self.alphabet.index(char)] = 1
        return ret

    def load_file(self, fname) -> np.ndarray:
        embeddings = self._file_to_onehot_chars(os.path.join(DATA_PATH, fname))
        return embeddings

    def load_sets(self, aug=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: (train_x, train_y, val_x, val_y, test_x, test_y)
        """
        train_x_pos = self.load_file('rt-polarity-utf8.train.pos' + ('.aug' if aug else ''))
        train_x_neg = self.load_file('rt-polarity-utf8.train.neg' + ('.aug' if aug else ''))
        train_x = np.concatenate([train_x_pos, train_x_neg])
        train_y = np.array([1] * len(train_x_pos) + [0] * len(train_x_neg))
        train_y = train_y.reshape((len(train_y), 1))

        val_x_pos = self.load_file('rt-polarity-utf8.val.pos')
        val_x_neg = self.load_file('rt-polarity-utf8.val.neg')
        val_x = np.concatenate([val_x_pos, val_x_neg])
        val_y = np.array([1] * len(val_x_pos) + [0] * len(val_x_neg))
        val_y = val_y.reshape((len(val_y), 1))

        test_x_pos = self.load_file('rt-polarity-utf8.test.pos')
        test_x_neg = self.load_file('rt-polarity-utf8.test.neg')
        test_x = np.concatenate([test_x_pos, test_x_neg])
        test_y = np.array([1] * len(test_x_pos) + [0] * len(test_x_neg))
        test_y = test_y.reshape((len(test_y), 1))

        return train_x, train_y, val_x, val_y, test_x, test_y
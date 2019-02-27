import numpy as np
from typing import List, Dict, Tuple
import os

DATA_PATH = 'data/rt-polaritydata'
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


class MovieReviewSingleSentenceDatasetLoader:
    def __init__(self, dim, input_length):
        self.dim = dim
        self.input_length = input_length
        self.word_to_int, self.int_to_embedding, self.glove_dict, self.int_to_word = self._load_glove()

    def _file_to_word_ids(self, fname: str) -> np.ndarray:
        with open(fname, 'r') as file:
            lines = file.read().strip().split('\n')
        ret = np.zeros(shape=(len(lines), self.input_length), dtype=np.int32)
        for i, line in enumerate(lines):
            ret[i] = self._line_to_word_ids(line.split(' '))
        return ret

    def _line_to_word_ids(self, line: List[str]) -> np.ndarray:
        ret = np.zeros(shape=(1, self.input_length), dtype=np.int32)
        for i, word in enumerate(line):
            if i >= self.input_length:
                break
            if word in self.word_to_int:
                ret[0, i] = self.word_to_int[word]
        return ret

    def _load_glove(self) -> (Dict[str, int], np.ndarray, Dict[str, np.ndarray], List[str]):
        word_to_int = {}
        int_to_word = []
        int_to_embedding = []
        fname = "glove.6B.%dd.txt" % self.dim
        with open(fname, 'r') as glove_file:
            content = glove_file.read()
            for line in content.split('\n'):
                elems = line.split(' ')
                token = elems[0]
                if len(elems) > 1:
                    embedding = np.array(list(map(float, elems[1:])))
                    word_to_int[token] = len(int_to_embedding)
                    int_to_embedding.append(embedding)
                    int_to_word.append(token)
        int_to_embedding = np.array(int_to_embedding, dtype=np.float32)
        combined = {}
        for w in word_to_int:
            combined[w] = int_to_embedding[word_to_int[w]]
        return word_to_int, int_to_embedding, combined, int_to_word

    def load_file(self, fname) -> np.ndarray:
        embeddings = self._file_to_word_ids(os.path.join(DATA_PATH, fname))
        return embeddings

    def load_word_ids(self, aug=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: (train_x, train_y, val_x, val_y, test_x, test_y)
        """
        train_x_pos = self.load_file('rt-polarity-utf8.train.pos' + ('.aug' if aug else ''))
        train_x_neg = self.load_file('rt-polarity-utf8.train.neg' + ('.aug' if aug else ''))
        train_x = np.concatenate([train_x_pos, train_x_neg])
        train_y = np.array([1] * len(train_x_pos) + [0] * len(train_x_neg))
        train_y = train_y.reshape((len(train_y), 1))
        del train_x_pos, train_x_neg

        val_x_pos = self.load_file('rt-polarity-utf8.val.pos')
        val_x_neg = self.load_file('rt-polarity-utf8.val.neg')
        val_x = np.concatenate([val_x_pos, val_x_neg])
        val_y = np.array([1] * len(val_x_pos) + [0] * len(val_x_neg))
        val_y = val_y.reshape((len(val_y), 1))
        del val_x_pos, val_x_neg

        test_x_pos = self.load_file('rt-polarity-utf8.test.pos')
        test_x_neg = self.load_file('rt-polarity-utf8.test.neg')
        test_x = np.concatenate([test_x_pos, test_x_neg])
        test_y = np.array([1] * len(test_x_pos) + [0] * len(test_x_neg))
        test_y = test_y.reshape((len(test_y), 1))
        del test_x_pos, test_x_neg

        return train_x, train_y, val_x, val_y, test_x, test_y

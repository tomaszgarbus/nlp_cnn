import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from models.yoon_kim_cnn import YoonKimCnnStatic

DIM = 100
INPUT_LENGTH = 40
DATA_PATH = 'data/rt-polaritydata'
TEST_SPLIT = 0.2


class DataLoader:
    def __init__(self):
        self.word_to_int, self.int_to_embedding, self.glove_dict, self.int_to_word = self._load_glove(DIM)

    def _file_to_word_ids(self, fname: str) -> np.ndarray:
        with open(fname, 'r') as file:
            lines = file.read().strip().split('\n')
        ret = np.zeros(shape=(len(lines), INPUT_LENGTH), dtype=np.int32)
        for i, line in enumerate(lines):
            ret[i] = self._line_to_word_ids(line.split(' '))
        return ret

    def _line_to_word_ids(self, line: List[str]) -> np.ndarray:
        ret = np.zeros(shape=(1, INPUT_LENGTH), dtype=np.int32)
        for i, word in enumerate(line):
            if i >= INPUT_LENGTH:
                break
            if word in self.word_to_int:
                ret[0, i] = self.word_to_int[word]
        return ret

    def _build_embeddings(self, fname: str) -> np.ndarray:
        with open(fname, 'r') as file:
            lines = file.read().strip().split('\n')
        ret = np.zeros(shape=(len(lines), INPUT_LENGTH, DIM))
        for i, line in enumerate(lines):
            ret[i] = self._line_embedding(line.split(' '))
        return ret

    def _line_embedding(self, line: List[str]) -> np.ndarray:
        ret = np.zeros(shape=(1, INPUT_LENGTH, DIM))
        for i, word in enumerate(line):
            if i >= INPUT_LENGTH:
                break
            if word in self.glove_dict:
                ret[0, i] = self.glove_dict[word]
        return ret

    @staticmethod
    def _load_glove(dims=DIM) -> (Dict[str, int], np.ndarray, Dict[str, np.ndarray], List[str]):
        word_to_int = {}
        int_to_word = []
        int_to_embedding = []
        fname = "glove.6B.%dd.txt" % dims
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

    def load_word_ids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: (train_x, train_y, test_x, test_y)
        """
        pos_file = os.path.join(DATA_PATH, 'rt-polarity-utf8.pos')
        neg_file = os.path.join(DATA_PATH, 'rt-polarity-utf8.neg')
        pos_embeddings = self._file_to_word_ids(pos_file)
        neg_embeddings = self._file_to_word_ids(neg_file)
        test_count = int(len(pos_embeddings) * TEST_SPLIT)
        train_count = len(pos_embeddings) - test_count
        train_x = np.concatenate([pos_embeddings[:train_count], neg_embeddings[:train_count]])
        train_y = np.array([1] * train_count + [0] * train_count).reshape((2 * train_count, 1))
        test_x = np.concatenate([pos_embeddings[-test_count:], neg_embeddings[-test_count:]])
        test_y = np.array([1] * test_count + [0] * test_count).reshape((2 * test_count, 1))
        return train_x, train_y, test_x, test_y

    def load_embeddings(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: (train_x, train_y, test_x, test_y)
        """
        pos_file = os.path.join(DATA_PATH, 'rt-polarity-utf8.pos')
        neg_file = os.path.join(DATA_PATH, 'rt-polarity-utf8.neg')
        pos_embeddings = self._build_embeddings(pos_file)
        neg_embeddings = self._build_embeddings(neg_file)
        test_count = int(len(pos_embeddings) * TEST_SPLIT)
        train_count = len(pos_embeddings) - test_count
        train_x = np.concatenate([pos_embeddings[:train_count], neg_embeddings[:train_count]])
        train_y = np.array([1] * train_count + [0] * train_count).reshape((2 * train_count, 1))
        test_x = np.concatenate([pos_embeddings[-test_count:], neg_embeddings[-test_count:]])
        test_y = np.array([1] * test_count + [0] * test_count).reshape((2 * test_count, 1))
        return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    dloader = DataLoader()
    print("Built data loader.")
    train_x, train_y, test_x, test_y = dloader.load_word_ids()
    print("Loaded dataset.")
    with tf.Session() as sess:
        cnn = YoonKimCnnStatic(sess,
                               embeddings_dim=DIM,
                               input_length=INPUT_LENGTH,
                               learning_rate=0.05,
                               num_filters=200,
                               regularization_rate=0.1,
                               static='rand',
                               initial_embeddings=None)
                               # initial_embeddings=dloader.int_to_embedding)
        print("Initialized or loaded model (" + str(cnn) + ").")
        cnn.fit(train_x, train_y, nb_epochs=600, val_x=test_x, val_y=test_y, ckpt_freq=200)
        cnn.find_most_exciting(train_x, dloader.int_to_word)
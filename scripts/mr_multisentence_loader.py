import numpy as np
from typing import List, Dict, Tuple
from utils import listfiles
import os

DATA_PATH = 'data/txt_sentoken'
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


class MovieReviewMultiSentenceDatasetLoader:
    def __init__(self, dim, input_length):
        self.dim = dim
        self.input_length = input_length
        self.word_to_int, self.int_to_embedding, self.glove_dict, self.int_to_word = self._load_glove()

    def _file_to_word_ids(self, fname: str) -> np.ndarray:
        with open(fname, 'r') as file:
            lines = file.read().strip().split('\n')
        # Concatenates all lines
        line = ' '.join(lines)
        line = line.split(' ')
        print(len(line))
        ret = np.zeros(shape=(self.input_length,), dtype=np.int32)
        for i, word in enumerate(line):
            if i >= self.input_length:
                break
            if word in self.word_to_int:
                ret[i] = self.word_to_int[word]
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

    def _load_dir(self, dir: str) -> Tuple[np.ndarray, np.ndarray]:
        pos_files = listfiles(os.path.join(DATA_PATH, dir, 'pos'))
        pos_x = np.zeros(shape=(len(pos_files), self.input_length))
        for i, file in enumerate(pos_files):
            pos_x[i] = self._file_to_word_ids(os.path.join(DATA_PATH, dir, 'pos', file))

        neg_files = listfiles(os.path.join(DATA_PATH, dir, 'neg'))
        neg_x = np.zeros(shape=(len(neg_files), self.input_length))
        for i, file in enumerate(neg_files):
            neg_x[i] = self._file_to_word_ids(os.path.join(DATA_PATH, dir, 'neg', file))

        x = np.concatenate([pos_x, neg_x])
        y = np.array([1] * len(pos_files) + [0] * len(neg_files))
        y = y.reshape((len(y), 1))
        return x, y

    def load_word_ids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: (train_x, train_y, val_x, val_y, test_x, test_y)
        """
        train_x, train_y = self._load_dir('train')
        val_x, val_y = self._load_dir('val')
        test_x, test_y = self._load_dir('test')
        return train_x, train_y, val_x, val_y, test_x, test_y

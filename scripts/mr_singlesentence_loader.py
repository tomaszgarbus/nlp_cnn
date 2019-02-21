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

    def load_word_ids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: (train_x, train_y, val_x, val_y, test_x, test_y)
        """
        pos_file = os.path.join(DATA_PATH, 'rt-polarity-utf8.pos')
        neg_file = os.path.join(DATA_PATH, 'rt-polarity-utf8.neg')
        pos_embeddings = self._file_to_word_ids(pos_file)
        neg_embeddings = self._file_to_word_ids(neg_file)
        val_count = int(len(pos_embeddings) * VAL_SPLIT)
        test_count = int(len(pos_embeddings) * TEST_SPLIT)
        train_count = len(pos_embeddings) - test_count - val_count

        val_beg = train_count
        test_beg = val_beg + val_count

        train_x = np.concatenate([pos_embeddings[:val_beg], neg_embeddings[:val_beg]])
        train_y = np.array([1] * train_count + [0] * train_count).reshape((2 * train_count, 1))

        val_x = np.concatenate([pos_embeddings[val_beg:test_beg], neg_embeddings[val_beg:test_beg]])
        val_y = np.array([1] * val_count + [0] * val_count).reshape((2 * val_count, 1))

        test_x = np.concatenate([pos_embeddings[test_beg:], neg_embeddings[test_beg:]])
        test_y = np.array([1] * test_count + [0] * test_count).reshape((2 * test_count, 1))
        return train_x, train_y, val_x, val_y, test_x, test_y
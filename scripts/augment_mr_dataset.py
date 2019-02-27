import numpy as np
from typing import List, Dict, Tuple
import os

DATA_PATH = 'data/rt-polaritydata'
THESAURUS_PATH = 'data/th_en_US_v2.dat'
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

""" Parameters of geometric distribution. """
P = 0.5
Q = 0.5


class MRAugmenter:
    def __init__(self):
        self._init_thesaurus()

    def _init_thesaurus(self):
        self._thesaurus = {}
        with open(THESAURUS_PATH, 'r') as file:
            th_raw = file.read().strip().split('\n')
            i = 1
            while i < len(th_raw):
                word, meanings_count = th_raw[i].split('|')
                meanings_count = int(meanings_count)
                synonyms = []
                for j in range(meanings_count):
                    meaning = th_raw[i+j+1]
                    for synonym in meaning.split('|')[1:]:
                        if synonym.find('(') != -1:
                            synonym = synonym[:synonym.find('(')]
                        if synonym not in synonyms and synonym != word:
                            synonyms.append(synonym)
                if len(word) > 2:
                    self._thesaurus[word] = synonyms
                i = i + meanings_count + 1
        print(self._thesaurus['dog'])

    def lines_from_file(self, fname):
        with open(fname, 'r') as file:
            return file.read().strip().split('\n')

    def augment_line(self, line: str, cycles: int = 5) -> List[str]:
        ret = []
        for c in range(cycles):
            np.random.seed(100 + c)
            words = line.split(' ')
            candidates = []
            for i, word in enumerate(words):
                if word in self._thesaurus:
                    candidates.append(i)
            to_replace = min(len(candidates), np.random.geometric(p=P))
            to_replace = np.random.permutation(candidates)[:to_replace]
            for idx in to_replace:
                word = words[idx]
                synonym_idx = min(len(self._thesaurus[word])-1, np.random.geometric(p=Q)-1)
                synonym = self._thesaurus[word][synonym_idx]
                words[idx] = synonym.lower()
            new_line = ' '.join(words)
            ret.append(new_line)
            print(line, ":::::::", new_line)
        return ret

    def augment(self, save=True) -> None:
        pos_lines = self.lines_from_file(os.path.join(DATA_PATH, 'rt-polarity-utf8.train.pos'))
        neg_lines = self.lines_from_file(os.path.join(DATA_PATH, 'rt-polarity-utf8.train.neg'))
        aug_pos_lines = pos_lines.copy()
        for line in pos_lines:
            aug_pos_lines += self.augment_line(line)
        aug_neg_lines = neg_lines.copy()
        for line in neg_lines:
            aug_neg_lines += self.augment_line(line)
        if save:
            with open(os.path.join(DATA_PATH, 'rt-polarity-utf8.train.pos.aug'), 'w') as fpos:
                fpos.write('\n'.join(aug_pos_lines))
            with open(os.path.join(DATA_PATH, 'rt-polarity-utf8.train.neg.aug'), 'w') as fneg:
                fneg.write('\n'.join(aug_neg_lines))
        print(len(aug_pos_lines), len(aug_neg_lines))


if __name__ == '__main__':
    augmenter = MRAugmenter()
    augmenter.augment(save=False)
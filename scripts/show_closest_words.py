import tensorflow as tf
from typing import List, Tuple
import numpy as np

from models.yoon_kim_cnn import YoonKimCnnStatic
from scripts.subj_loader import SubjDatasetLoader

DIM = 200
INPUT_LENGTH = 40


def nearest_neigbors(wordvector: np.ndarray, embeddings: np.ndarray, count=5) -> List[Tuple[float, int]]:
    """
    :return: List of pairs (distance, word int)
    """
    distances = np.zeros(shape=(len(embeddings),))
    for i in range(len(distances)):
        distances[i] = np.linalg.norm(wordvector - embeddings[i])
        # if i % 100 == 0:
        #     print(i, len(distances))
    print("2")
    sorted_distances = np.argsort(distances, axis=0)
    print("3")
    ret = []
    for i in range(count):
        ret.append((distances[sorted_distances[i]], sorted_distances[i]))
    print("4")
    return ret


if __name__ == '__main__':
    dloader = SubjDatasetLoader(dim=DIM, input_length=INPUT_LENGTH)
    orig_embeddings = dloader.int_to_embedding
    print("Built data loader.")
    train_x, train_y, val_x, val_y, test_x, test_y = dloader.load_word_ids()
    print("Loaded dataset.")
    with tf.Session() as sess:
        cnn = YoonKimCnnStatic(sess,
                               embeddings_dim=DIM,
                               input_length=INPUT_LENGTH,
                               learning_rate=0.05,
                               num_filters=100,
                               regularization_rate=0.1,
                               static='rand',
                               initial_embeddings=None)
                               # initial_embeddings=dloader.int_to_embedding)
        trained_embeddings = sess.run([cnn.embeddings])[0]

    print("Got original and trained embeddings")
    while True:
        print("Please input word")
        word = input().strip().lower()
        print("thank")

        wordvector = orig_embeddings[dloader.word_to_int[word]]
        orig_closest = nearest_neigbors(wordvector, orig_embeddings)
        print(orig_closest)
        orig_closest = list(map(lambda d: (d[0], dloader.int_to_word[d[1]]), orig_closest))
        print(orig_closest)

        wordvector = trained_embeddings[dloader.word_to_int[word]]
        trained_closest = nearest_neigbors(wordvector, trained_embeddings)
        trained_closest = list(map(lambda d: (d[0], dloader.int_to_word[d[1]]), trained_closest))
        print(trained_closest)

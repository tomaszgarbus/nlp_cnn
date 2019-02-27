from models.model import Model
from models.yoon_kim_cnn import YoonKimCnnStatic
import tensorflow as tf
import numpy as np
from scripts.mr_singlesentence_loader import MovieReviewSingleSentenceDatasetLoader


if __name__ == '__main__':
    dloader = MovieReviewSingleSentenceDatasetLoader(dim=200, input_length=40)
    _, _, _, _, test_x, test_y = dloader.load_word_ids()
    with tf.Session() as sess:
        model1 = YoonKimCnnStatic(sess,
                                  embeddings_dim=200,
                                  input_length=40,
                                  learning_rate=0.05,
                                  num_filters=100,
                                  regularization_rate=0.1,
                                  static='static',
                                  initial_embeddings=dloader.int_to_embedding)
        preds1 = model1.predict(test_x, round_value=False)
    with tf.Session() as sess:
        model2 = YoonKimCnnStatic(sess,
                                  embeddings_dim=200,
                                  input_length=40,
                                  learning_rate=0.05,
                                  num_filters=100,
                                  regularization_rate=0.1,
                                  static='non-static',
                                  initial_embeddings=dloader.int_to_embedding)

        preds2 = model2.predict(test_x, round_value=False)
    preds = np.round(preds1 + preds2) / 2
    acc = 1.0 - np.mean(np.abs(preds - test_y))
    print(acc)

import tensorflow as tf

from models.yoon_kim_cnn import YoonKimCnnStatic
from scripts.mr_singlesentence_loader import MovieReviewSingleSentenceDatasetLoader

DIM = 200
INPUT_LENGTH = 40


if __name__ == '__main__':
    dloader = MovieReviewSingleSentenceDatasetLoader(dim=DIM, input_length=INPUT_LENGTH)
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
                               static='both',
                               # initial_embeddings=None)
                               initial_embeddings=dloader.int_to_embedding)
        print("Initialized or loaded model (" + str(cnn) + ").")
        print("Test acc: " + str(cnn.evaluate(test_x, test_y)))
        cnn.fit(train_x, train_y, nb_epochs=600, val_x=val_x, val_y=val_y, ckpt_freq=200)

        cnn.find_most_exciting(train_x, dloader.int_to_word)
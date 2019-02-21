import tensorflow as tf

from models.dcnn import DCNN
from scripts.mr_singlesentence_loader import MovieReviewSingleSentenceDatasetLoader

DIM = 200
INPUT_LENGTH = 40


if __name__ == '__main__':
    dloader = MovieReviewSingleSentenceDatasetLoader(dim=DIM, input_length=INPUT_LENGTH)
    print("Built data loader.")
    train_x, train_y, val_x, val_y, test_x, test_y = dloader.load_word_ids()
    print("Loaded dataset.")
    with tf.Session() as sess:
        cnn = DCNN(sess,
                   initial_embeddings=dloader.int_to_embedding,
                   embeddings_dim=DIM,
                   learning_rate=0.5,
                   )
        print("Initialized or loaded model (" + str(cnn) + ").")
        cnn.fit(train_x, train_y, nb_epochs=200, batch_size=50, val_x=test_x, val_y=test_y, ckpt_freq=200)

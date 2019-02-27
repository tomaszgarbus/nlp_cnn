import tensorflow as tf

from models.char_cnn import CharCNN
from scripts.mr_singlesentence_char_loader import MovieReviewSingleSentenceDatasetCharLoader

INPUT_LENGTH = 250
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}"

if __name__ == '__main__':
    dloader = MovieReviewSingleSentenceDatasetCharLoader(alphabet=ALPHABET, input_length=INPUT_LENGTH)
    train_x, train_y, val_x, val_y, test_x, test_y = dloader.load_sets(aug=True)
    with tf.Session() as sess:
        cnn = CharCNN(sess, learning_rate=0.01, input_length=INPUT_LENGTH)
        cnn.fit(x=train_x, y=train_y, batch_size=128, nb_epochs=200, val_x=val_x, val_y=val_y)
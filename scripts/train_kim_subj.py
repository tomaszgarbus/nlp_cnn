import tensorflow as tf

from models.yoon_kim_cnn import YoonKimCnnStatic
from scripts.subj_loader import SubjDatasetLoader

DIM = 200
INPUT_LENGTH = 40


if __name__ == '__main__':
    dloader = SubjDatasetLoader(dim=DIM, input_length=INPUT_LENGTH)
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
                               static='non-static',
                               # initial_embeddings=None)
                               initial_embeddings=dloader.int_to_embedding)
        print("Initialized or loaded model (" + str(cnn) + ").")
        print("Test acc: " + str(cnn.evaluate(test_x, test_y)))
        cnn.fit(train_x, train_y, nb_epochs=250, val_x=val_x, val_y=val_y, ckpt_freq=250)
        print("Test acc: " + str(cnn.evaluate(test_x, test_y)))
        cnn.find_most_exciting(x=train_x, int_to_word=dloader.int_to_word)

from collections import Counter
from itertools import chain
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.data import Dataset

from src.helpers import create_train_test_splits, tokenize_sentences, build_target_context

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class FruitFly(tf.keras.models.Model):

    def __init__(self, vocab_size: int, n_kenyon: int = 100, batch_size: int = 256, optimizer=None,
                 window_size: int = 3):
        super().__init__()
        self.projection_layer = InputLayer(vocab_size * 2, dtype=tf.float64)
        self.kenyon_cells = Dense(n_kenyon, dtype=tf.float64)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        if optimizer is None:
            optimizer = Adam(lr=tf.Variable(0.01, dtype=tf.float64),
                             beta_1=tf.Variable(0.9, dtype=tf.float64),
                             beta_2=tf.Variable(0.999, dtype=tf.float64),
                             epsilon=tf.Variable(1e-7, dtype=tf.float64)
                             )
            optimizer.iterations
            optimizer.decay = tf.Variable(0.0)
        self.optimizer = optimizer
        self.window_size = window_size
        self.tokenizer = None
        self.word_index = None

    def call(self, X, word_prob):
        X = self.projection_layer(X, dtype=X.dtype)
        y = self.kenyon_cells(X)
        loss = self.get_loss(X, self.kenyon_cells.weights[0], y, word_prob)
        return y, loss

    def get_word_prob(self, word_counts, index_word):
        prob = np.zeros((self.vocab_size * 2, ))
        for i in range(self.vocab_size):
            prob[i] = word_counts[index_word[i + 1]]
            prob[i + self.vocab_size] = word_counts[index_word[i + 1]]
        prob = prob / (sum(prob) / 2)
        return prob

    def train(self, sentences: list, number_epochs: int = 10):
        if isinstance(sentences[0], str):
            sentences = [s.split(" ") for s in sentences]
        assert isinstance(sentences, list)
        assert isinstance(sentences[0], list)
        word_index, tokenizer = tokenize_sentences(sentences, num_words=self.vocab_size + 1)
        word_prob = self.get_word_prob(tokenizer.word_counts, tokenizer.index_word)
        y, X = build_target_context(sentences, windows_size=self.window_size)
        self.tokenizer, self.word_index = tokenizer, word_index
        target, context = tokenizer.texts_to_matrix(y), tokenizer.texts_to_matrix(X)
        X = np.concatenate((context[:, 1:], target[:, 1:]), axis=1)

        splits = create_train_test_splits(X)
        buffer_size = splits[0].shape[0]
        dataset = Dataset.from_tensor_slices((splits[0], )).shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(self.batch_size)
        steps_per_epoch = max(buffer_size // self.batch_size, 1)
        for epoch in range(number_epochs):
            loss_per_batch = []
            for (batch, (input, )) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self._train_step(input, word_prob)
                loss_per_batch.append(batch_loss)
                if batch % 10 == 0 and batch > 1:
                    print("Batch loss: {}".format(np.mean(loss_per_batch[-10:])))
            epoch_loss = np.mean(loss_per_batch)
            print("Epoch loss: {}".format(epoch_loss))

    def get_loss(self, x, w, kn, p):
        """

        :param w: weights between projection layer and kenyon cells
        :param kn: values of kenyon cells
        :param p: probability of word
        :return:
        """
        x = tf.cast(x, p.dtype)
        x = x / p
        ken_cell_indices = tf.math.argmax(kn, axis=1)
        inner_product_squared = {i: tf.square(tf.tensordot(w[:, i], w[:, i], axes=1)) for i in range(w.shape[1])}
        energy = - tf.reduce_sum(
            [
                tf.tensordot(w[:, ken_cell_indices[j]], x[j, :], axes=1) /
                inner_product_squared[int(ken_cell_indices[j])]
                for j in range(x.shape[0])
            ]
        )
        return energy

    def _train_step(self, X, word_prob):
        with tf.GradientTape() as g:
            y, loss = self.call(X, word_prob)
        variables = self.trainable_variables
        gradients = g.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss


if __name__ == "__main__":
    train_data_share = 0.2
    data = pd.read_feather(Path("..", "data", "products.feather"))
    data = data.iloc[np.random.choice(np.arange(data.shape[0]), size=int(train_data_share*data.shape[0])), :]
    fruitfly = FruitFly(3000, n_kenyon=100)
    fruitfly.train(data["StockCode"].tolist())



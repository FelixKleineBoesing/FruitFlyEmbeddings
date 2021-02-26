import tensorflow as tf
import numpy as np
import os

from tensorflow.data import Dataset
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

from src.helpers import create_train_test_splits

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class FruitFly(tf.keras.models.Model):

    def __init__(self, vocab_size: int, n_kenyon: int = 100, batch_size: int = 32, optimizer=None):
        super().__init__()
        self.projection_layer = InputLayer(vocab_size * 2, dtype=tf.float64)
        self.kenyon_cells = Dense(n_kenyon, dtype=tf.float64)
        self.batch_size = batch_size
        if optimizer is None:
            optimizer = Adam(lr=tf.Variable(0.01),
                             beta_1=tf.Variable(0.9),
                             beta_2=tf.Variable(0.999),
                             epsilon=tf.Variable(1e-7)
                             )
            optimizer.iterations
            optimizer.decay = tf.Variable(0.0)
        self.optimizer = optimizer

    def call(self, X, word_prob):
        X = self.projection_layer(X, dtype=X.dtype)
        y = self.kenyon_cells(X)
        loss = self.get_loss(X, self.kenyon_cells.weights[0], y, word_prob)
        return y, loss

    def get_word_prob(self, X):
        context_target_border = int(X.shape[1] / 2)
        counts = tf.math.reduce_sum(X[:, :context_target_border], axis=0)
        probs = counts / tf.reduce_sum(counts)
        return tf.tile(probs, multiples=[2])

    def train(self, X, number_epochs: int = 10):
        word_prob = self.get_word_prob(X)
        input_splits = create_train_test_splits(X)
        buffer_size = input_splits[0].shape[0]
        dataset = Dataset.from_tensor_slices((input_splits[0], )).shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(self.batch_size)
        steps_per_epoch = max(buffer_size // self.batch_size, 1)
        for epoch in range(number_epochs):
            loss_per_batch = []
            for (batch, (input, )) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self._train_step(input, word_prob)
                loss_per_batch.append(batch_loss)
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

    data = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0]], dtype=np.float)
    fruitfly = FruitFly(5, n_kenyon=3)
    fruitfly.train(data)

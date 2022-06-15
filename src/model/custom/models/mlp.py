# 서드파티
import tensorflow as tf
import numpy as np


class ExampleModel(tf.keras.Model):

    def __init__(self,
                 input_dim: int = 32,
                 output_dim: int = 2):
        super(ExampleModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(input_dim, activation='relu')
        self.d2 = tf.keras.layers.Dense(50, activation='relu')
        self.d3 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, x, training=None):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x

    @staticmethod
    def loss_object(y, y_hat):
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return scce(y, y_hat)


def train_step(model, x, y, train_acc):
    adam = tf.keras.optimizers.Adam()
    with tf.GradientTape() as tape:
        y_hat = model(x, training=True)
        loss = model.loss_object(y, y_hat)
        gradients = tape.gradient(loss, model.trainable_variables)
        adam.apply_gradients(zip(gradients, model.trainable_variables))
    print(f'train_accuracy: {train_acc(y, y_hat):.3f}')


def test_step(model, text, label, test_acc):
    y_hat = model(text, training=False)
    print(f'test_accuracy: {test_acc(label, y_hat):.3f}')


def main():
    input_dim = 32
    model = ExampleModel(input_dim=input_dim)

    epochs = 100
    batch_size = 2

    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accy')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

    for _ in range(epochs):
        train_x = np.random.random((batch_size, input_dim))
        train_y = np.random.random((batch_size,)).astype(np.uint32)
        print(repr(train_x))
        print(repr(train_y))
        train_step(model, train_x, train_y, train_acc=train_acc)

    test_x = np.random.random((10, input_dim))
    test_y = np.random.random((10,)).astype(np.uint32)
    test_step(model, test_x, test_y, test_acc=test_acc)


if __name__ == '__main__':
    main()

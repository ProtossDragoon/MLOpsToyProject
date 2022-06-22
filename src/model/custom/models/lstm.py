# 서드파티
import tensorflow as tf
import numpy as np

# 프로젝트
from src.environment.tpu import ColabTPUEnvironmentManager
from src.preprocessing.sms import SMSDataPreprocessingManager


class LSTMModel(tf.keras.Model):

    def __init__(self,
                 word_embedding_dim: int = None,
                 sentence_max_len_dim: int = None,
                 output_dim: int = 2):
        super(LSTMModel, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.sentence_max_len_dim = sentence_max_len_dim
        self.output_dim = output_dim
        self.l1 = tf.keras.layers.LSTM(80, return_sequences=True)
        self.l2 = tf.keras.layers.LSTM(30, return_sequences=True)
        self.l3 = tf.keras.layers.LSTM(output_dim, activation='softmax')

    def call(self, x, training=None):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

    @staticmethod
    def loss_object(y, y_hat):
        if ColabTPUEnvironmentManager.is_tpu_env():
            #FIXME https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction
            reduction = tf.keras.losses.Reduction.NONE
        else:
            reduction = tf.keras.losses.Reduction.AUTO
        scce = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=reduction)
        return scce(y, y_hat)


def main():
    # 데이터 로드
    prep_manager = SMSDataPreprocessingManager(
        feature_column_name='message',
        label_column_name='label'
    )
    path = './data/sample/spam.csv'
    df = prep_manager.read_sample_data(path)

    # 기본 전처리
    prep_manager.remove_stopwords(df)
    prep_manager.sentence_to_lowercase(df)
    train_df, test_df = prep_manager.split(df, 0.8)

    # 훈련 데이터 전처리
    train_x, train_y = prep_manager.get_xy(train_df, {'spam': 0., 'ham': 1.})
    train_x_onehot = prep_manager.get_onehot(train_x)

    # 테스트 데이터 전처리
    test_x, test_y = prep_manager.get_xy(test_df, {'spam': 0., 'ham': 1.})
    test_x_onehot = prep_manager.get_onehot(test_x)

    # 데이터 요약
    print(f'train data spec:')
    print(f'x: {train_x_onehot.shape}({train_x_onehot.dtype})')
    print(f'y: {train_y.shape}({train_y.dtype})')
    print(f'test data spec:')
    print(f'x: {test_x_onehot.shape}({test_x_onehot.dtype})')
    print(f'y: {test_y.shape}({test_y.dtype})')

    # 모델 정의
    word_embedding_dim = train_x_onehot.shape[-1]
    sentence_max_len_dim = train_x_onehot.shape[-2]
    model = LSTMModel(
        word_embedding_dim=word_embedding_dim,
        sentence_max_len_dim=sentence_max_len_dim,
    )

    epochs = 3
    batch_size = 2

    tfds_train = tf.data.Dataset.from_tensor_slices(
        (train_x_onehot, train_y)).batch(batch_size)
    tfds_test = tf.data.Dataset.from_tensor_slices(
        (test_x_onehot, test_y)).batch(1)

    if ColabTPUEnvironmentManager.is_tpu_env():
        tfds_train = ColabTPUEnvironmentManager.get_tpu_strategy(
        ).experimental_distribute_dataset(tfds_train)
        tfds_test = ColabTPUEnvironmentManager.get_tpu_strategy(
        ).experimental_distribute_dataset(tfds_test)

    # 메트릭 정의
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accy')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

    # 학습 루프 정의
    def train_step(x, y):
        adam = tf.keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            y_hat = model(x, training=True)
            loss = model.loss_object(y, y_hat)
            gradients = tape.gradient(loss, model.trainable_variables)
        adam.apply_gradients(zip(gradients, model.trainable_variables))
        train_acc.update_state(y, y_hat)

    @tf.function
    def distributed_train_step(x, y):
        strategy = ColabTPUEnvironmentManager.get_tpu_strategy()
        strategy.run(train_step, args=(x, y))

    def test_step(x, y):
        y_hat = model(x, training=False)
        test_acc.update_state(y, y_hat)

    @tf.function
    def distributed_test_step(x, y):
        strategy = ColabTPUEnvironmentManager.get_tpu_strategy()
        strategy.run(test_step, args=(x, y))

    # 학습
    for epoch in range(1, epochs+1):
        for step, (train_x, train_y) in enumerate(tfds_train, 1):
            print(f'epoch: {epoch}, step: {step}', end='\t')
            # print(f'x: {repr(train_x.shape)}, y: {repr(train_y.shape)}')
            if ColabTPUEnvironmentManager.is_tpu_env():
                distributed_train_step(train_x, train_y)
            train_step(train_x, train_y)
            print(f'train_accuracy: {train_acc.result():.3f}')

    # 평가
    for step, (test_x, test_y) in enumerate(tfds_test, 1):
        if ColabTPUEnvironmentManager.is_tpu_env():
            distributed_test_step(test_x, test_y)
        test_step(test_x, test_y)
        print(f'train_accuracy: {test_acc.result():.3f}')

    # 요약
    print(f'train_acc: {train_acc.result():.3f}, '
          f'test_acc: {test_acc.result():.3f}')


if __name__ == '__main__':
    if ColabTPUEnvironmentManager.is_tpu_env():
        with ColabTPUEnvironmentManager.get_tpu_strategy().scope():
            main()
    else:
        main()

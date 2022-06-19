# 서드파티
import tensorflow as tf
import numpy as np

# 프로젝트
from src.environment.tpu import ColabTPUEnvironmentManager
from src.preprocessing.sms import SMSDataPreprocessingManager


class MLPModel(tf.keras.Model):

    def __init__(self,
                 input_dim: int = 32,
                 output_dim: int = 2):
        super(MLPModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1 = tf.keras.layers.Dense(input_dim, activation='relu')
        self.l2 = tf.keras.layers.Dense(50, activation='relu')
        self.l3 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, x, training=None):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
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
    train_x_dtm = prep_manager.get_dtm(train_x)
    train_x_tfidf = prep_manager.get_tfidf(train_x_dtm)

    # 테스트 데이터 전처리
    test_x, test_y = prep_manager.get_xy(test_df, {'spam': 0., 'ham': 1.})
    test_x_dtm = prep_manager.get_dtm(test_x)
    test_x_tfidf = prep_manager.get_tfidf(test_x_dtm)

    # 데이터 요약
    print(f'train data spec:')
    print(f'x: {train_x_tfidf.shape}({train_x_tfidf.dtype})')
    print(f'y: {train_y.shape}({train_y.dtype})')
    print(f'test data spec:')
    print(f'x: {test_x_tfidf.shape}({test_x_tfidf.dtype})')
    print(f'y: {test_y.shape}({test_y.dtype})')

    # 모델 정의
    input_dim = train_x_tfidf.shape[-1]
    model = MLPModel(input_dim=input_dim)

    epochs = 3
    batch_size = 2

    tfds_train = tf.data.Dataset.from_tensor_slices(
        (train_x_tfidf, train_y)).batch(batch_size)
    tfds_test = tf.data.Dataset.from_tensor_slices(
        (test_x_tfidf, test_y)).batch(1)

    # 메트릭 정의
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accy')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')

    # 학습
    for epoch in range(1, epochs+1):
        for step, (train_x, train_y) in enumerate(tfds_train, 1):
            print(f'epoch: {epoch}, step: {step}', end='\t')
            # print(f'x: {repr(train_x.shape)}, y: {repr(train_y.shape)}')
            train_step(model, train_x, train_y, train_acc=train_acc)

    # 평가
    for step, (test_x, test_y) in enumerate(tfds_test, 1):
        test_step(model, test_x, test_y, test_acc=test_acc)

    # 요약
    print(f'train_acc: {train_acc.result():.3f},'
          f'test_acc: {test_acc.result():.3f}')


if __name__ == '__main__':
    if ColabTPUEnvironmentManager.is_tpu_env():
        with ColabTPUEnvironmentManager.get_tpu_strategy():
            main()
    else:
        main()

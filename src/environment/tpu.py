# 내장
import sys
import os

# 서드파티
import tensorflow as tf


class COLABEnvironmentManager():

    @classmethod
    def is_colab_env(cls):
        if 'google.colab' in sys.modules:
            print(f'COLAB environment detected.')
            return True
        return False


class ColabTPUEnvironmentManager(COLABEnvironmentManager):

    tpu_resolver = None
    tpu_strategy = None

    @classmethod
    def is_tpu_env(cls):
        if not cls.is_colab_env():
            return False
        if os.environ.get('COLAB_TPU_ADDR') is None:
            return False
        print(f'TPU environment detected.')
        return True

    @classmethod
    def set_tpu_resolver(cls):
        cls.tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ.get('COLAB_TPU_ADDR'))
        tf.config.experimental_connect_to_cluster(cls.tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(cls.tpu_resolver)

    @classmethod
    def get_tpu_strategy(cls):
        if cls.tpu_resolver is None:
            cls.set_tpu_resolver()
        cls.strategy = tf.distribute.TPUStrategy(cls.tpu_resolver)
        return cls.strategy
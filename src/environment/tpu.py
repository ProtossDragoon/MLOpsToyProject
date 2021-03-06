# 내장
import sys
import os

# 서드파티
import tensorflow as tf


class COLABEnvironmentManager():

    _print_colab_env_manager_once = True

    @classmethod
    def is_colab_env(cls):
        try:
            import google.colab
        except ModuleNotFoundError:
            return False
        else:
            c = COLABEnvironmentManager
            if c._print_colab_env_manager_once:
                print(f'COLAB environment detected.')
                c._print_colab_env_manager_once = False
            return True


class ColabTPUEnvironmentManager(COLABEnvironmentManager):

    _print_colab_tpu_env_manager_once = True

    tpu_resolver = None
    tpu_strategy = None

    @classmethod
    def is_tpu_env(cls):
        if not cls.is_colab_env():
            return False
        if os.environ.get('COLAB_TPU_ADDR') is None:
            return False
        c = ColabTPUEnvironmentManager
        if c._print_colab_tpu_env_manager_once:
            print(f'TPU environment detected.')
            c._print_colab_tpu_env_manager_once = False
        return True

    @classmethod
    def set_tpu_resolver(cls):
        c = ColabTPUEnvironmentManager
        c.tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ.get('COLAB_TPU_ADDR'))
        tf.config.experimental_connect_to_cluster(c.tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(c.tpu_resolver)

    @classmethod
    def get_tpu_strategy(cls):
        c = ColabTPUEnvironmentManager
        if c.tpu_resolver is None:
            cls.set_tpu_resolver()
        if c.tpu_strategy is None:
            c.tpu_strategy = tf.distribute.TPUStrategy(c.tpu_resolver)
        return c.tpu_strategy

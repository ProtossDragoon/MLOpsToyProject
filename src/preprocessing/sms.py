# 내장
import string

# 서드파티
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# 프로젝트
from src.preprocessing.common import PreprocessingManager


class SMSDataPreprocessingManager(PreprocessingManager):

    def __init__(
        self,
        feature_column_name: str = 'message',
        label_column_name: str = 'label'
    ) -> None:
        super().__init__(feature_column_name, label_column_name)

    def read_sample_data(
        self,
        path: str,
        ratio: int = 0.1,
    ):
        df_sms = pd.read_csv(path, encoding='latin-1')
        df_sms.dropna(how="any", inplace=True, axis=1)
        df_sms.columns = [self.label_column_name, self.feature_column_name]
        return df_sms[:int(len(df_sms)*ratio)]

    def read_entire_data(
        self,
        path: str
    ):
        df_sms = pd.read_csv(path, encoding='latin-1')
        df_sms.dropna(how="any", inplace=True, axis=1)
        df_sms.columns = [self.label_column_name, self.feature_column_name]
        return df_sms

    def split(
        self,
        df: pd.DataFrame,
        ratio: int = 0.8,
    ) -> list:
        cnt = int(len(df)*ratio)
        train_df = df[:cnt]
        test_df = df[cnt:]
        return train_df, test_df

    def remove_stopwords(
        self,
        df: pd.DataFrame,
    ) -> None:
        nltk.download('stopwords')

        def fn(msg):
            STOPWORDS = stopwords.words('english') + \
                ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
            # 1. 철자 단위 검사 후 문장부호(punctuation) 제거
            nopunc = [char for char in msg if char not in string.punctuation]
            nopunc = ''.join(nopunc)
            # 2. 불용어(stop word) 제거
            return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
        df[self.feature_column_name] = df[self.feature_column_name].apply(fn)

    def sentence_to_lowercase(
        self,
        df: pd.DataFrame,
    ) -> None:
        def fn(sentence: str) -> list:
            return sentence.lower()
        df[self.feature_column_name] = df[self.feature_column_name].apply(fn)

    def get_xy(
        self,
        df: pd.DataFrame,
        label_map: dict = None,
    ) -> tuple:
        if label_map:
            self.label_map = label_map
            for key, val in label_map.items():
                condition = (df[self.label_column_name] == key)
                df.loc[condition, self.label_column_name] = val
        return (df[self.feature_column_name], df[self.label_column_name].astype(np.float32))

    def get_onehot(
        self,
        x: pd.Series,
    ) -> np.ndarray:
        if not hasattr(self, 'onehot_encoder'):
            self.onehot_encoder = Tokenizer()
            self.onehot_encoder.fit_on_texts(x)
            self.cnt_unique_words = len(self.onehot_encoder.word_counts)
            self.sentence_max_len = None
            print(f'{self.cnt_unique_words}-kind of words were detected.')

        def encode(x):
            encoded = self.onehot_encoder.texts_to_sequences(x)
            if self.sentence_max_len is None:
                self.sentence_max_len = max([len(s) for s in encoded])
                print(f'Maximum length of sentence: {self.sentence_max_len}')
            padded = pad_sequences(encoded, maxlen=self.sentence_max_len)
            #NOTE: Tokenizer reserves 0 as an "out of scope" index
            return to_categorical(padded, num_classes=self.cnt_unique_words+1)
        x_onehot = encode(x)
        print(f'One-hot matrix shape: {x_onehot.shape}')
        return x_onehot

    def get_dtm(
        self,
        x: pd.Series,
    ) -> np.ndarray:
        if not hasattr(self, 'count_vectorizer'):
            self.count_vectorizer = CountVectorizer()
            self.count_vectorizer.fit(x)
        x_dtm = self.count_vectorizer.transform(x)
        print(f'DTM matrix shape: {x_dtm.toarray().shape}')
        return x_dtm.toarray().astype(np.float32)

    def get_tfidf(
        self,
        x_dtm: np.ndarray,
    ) -> np.ndarray:
        if not hasattr(self, 'tfidf_transformer'):
            self.tfidf_transformer = TfidfTransformer()
            self.tfidf_transformer.fit(x_dtm)
        x_tfidf = self.tfidf_transformer.transform(x_dtm)
        print(f'TF-IDF matrix shape: {x_tfidf.toarray().shape}')
        return x_tfidf.toarray().astype(np.float32)

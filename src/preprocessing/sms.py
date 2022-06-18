# 내장
import string

# 서드파티
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
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
                df.loc[df[self.label_column_name] == key, self.label_column_name] = val
        return (df[self.feature_column_name], df[self.label_column_name].astype(np.float32))

    def get_dtm(
        self,
        x: np.array,
    ):
        if not hasattr(self, 'count_vectorizer'):
            self.count_vectorizer = CountVectorizer()
            self.count_vectorizer.fit(x)
        x_dtm = self.count_vectorizer.transform(x)
        print(f'DTM matrix shape: {x_dtm.toarray().shape}')
        return x_dtm.toarray().astype(np.float32)

    def get_tfidf(
        self,
        x_dtm,
    ):
        if not hasattr(self, 'tfidf_transformer'):
            self.tfidf_transformer = TfidfTransformer()
            self.tfidf_transformer.fit(x_dtm)
        x_tfidf = self.tfidf_transformer.transform(x_dtm)
        print(f'TF-IDF matrix shape: {x_tfidf.toarray().shape}')
        return x_tfidf.toarray().astype(np.float32)

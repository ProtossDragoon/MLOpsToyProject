import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import StringInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

# Convert text to numbers
from sklearn.feature_extraction.text import CountVectorizer 
from NBModel import NBModel
@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class SpamClassifier(BentoService):
    """ 
    A minimum prediction service exposing a Scikit-learn model
    """
    def __init__(self):
        super().__init__()
        model = NBModel()
        self.count_vector = model.getCountVector()

    @api(input=StringInput(), batch=True)
    def predict(self, str_input: str ):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        test = pd.Series(str_input)
        str_vector = self.count_vector.transform(test)
        return self.artifacts.model.predict(str_vector)

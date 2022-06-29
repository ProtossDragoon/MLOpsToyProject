# 내장
import os

# 서드파티
from fastapi import APIRouter
from pandas import DataFrame
from starlette.responses import JSONResponse
from ludwig.api import LudwigModel
from ludwig.serve import convert_input

# 프로젝트
from src.preprocessing.sms import SMSDataPreprocessingManager
from src.model.ludwig.config import LudwigConfigManager


router = APIRouter()


@router.post("/")
def do_inference(text):
    model_path = './models/ludwig_automl_example/model'
    model = LudwigModel.load(model_path, backend="local")

    def prep(text: str) -> dict:
        k = model.config["input_features"][0]["column"]
        return {k: text}

    def postp(df: DataFrame) -> dict:
        return df.to_dict("records")[0]

    input = prep(text)
    df, _ = model.predict(dataset=[input], data_format=type(input))
    return JSONResponse(postp(df))

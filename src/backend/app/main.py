# 서드파티
from fastapi import FastAPI

# 프로젝트
from src.backend.app.api.api_v1.api import api_router
from src.backend.app.core.config import settings


app = FastAPI()
app.include_router(api_router, prefix=settings.API_V1_STR)

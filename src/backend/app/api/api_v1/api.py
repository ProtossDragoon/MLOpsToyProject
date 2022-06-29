# 서드파티
from fastapi import APIRouter

# 프로젝트
from src.backend.app.api.api_v1.endpoints import inference


api_router = APIRouter()
api_router.include_router(inference.router, prefix="/inference")

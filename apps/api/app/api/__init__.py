from fastapi import APIRouter

from app.api.analysis import router as analysis_router
from app.api.routes import router as projects_router

api_router = APIRouter()
api_router.include_router(projects_router)
api_router.include_router(analysis_router)

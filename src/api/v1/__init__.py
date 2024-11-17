from fastapi import APIRouter # type: ignore

from src.api.v1.routers.health import endpoints as health

v1_routers = APIRouter(prefix="/v1")

v1_routers.include_router(health, prefix="/health", tags=["health"])

from fastapi import APIRouter # type: ignore

from src.api.v1.routers.health import endpoints as health
from src.api.v1.routers.metaheuristica import endpoints as metaheuristic

v1_routers = APIRouter(prefix="/v1")

v1_routers.include_router(health, prefix="/health", tags=["health"])
v1_routers.include_router(metaheuristic, prefix="/metaheuristica", tags=["metaheuristica"])

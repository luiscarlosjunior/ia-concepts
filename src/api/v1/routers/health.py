from fastapi import APIRouter # type: ignore

endpoints = APIRouter()

@endpoints.get("/health")
async def read_health():
    return {"status": "ok"}

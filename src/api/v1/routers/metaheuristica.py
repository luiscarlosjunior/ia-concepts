from fastapi import APIRouter # type: ignore

endpoints = APIRouter()

@endpoints.get("/tabu_search")
async def get_tabu_search():
    return {"message": "Hello from metaheuristica endpoint"}
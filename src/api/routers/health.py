from fastapi import APIRouter

endpoints = APIRouter()

@endpoints.get("hill-climbing")
async def pegar_tour():
    pass

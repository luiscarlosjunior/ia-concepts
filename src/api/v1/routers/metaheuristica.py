from fastapi import APIRouter, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Tuple
from src.algorithms.single.tabusearch_tsm import TabuSearchTSP

endpoints = APIRouter()

class CityCoordinate(BaseModel):
    x: float
    y: float

class TabuSearchRequest(BaseModel):
    cities: List[CityCoordinate]
    tabu_list_size: int = 5
    max_iterations: int = 100

class TabuSearchResponse(BaseModel):
    best_solution: List[int]
    best_distance: float
    algorithm_info: dict
    iterations_executed: int
    
@endpoints.post("/tabu_search", response_model=TabuSearchResponse)
async def run_tabu_search(request: TabuSearchRequest):
    """
    Execute Tabu Search algorithm for Traveling Salesman Problem (TSP).
    
    Args:
        request: Contains city coordinates and algorithm parameters
        
    Returns:
        Best solution found, total distance, and execution details
    """
    try:
        # Convert city coordinates to the format expected by the algorithm
        cities = [(city.x, city.y) for city in request.cities]
        
        if len(cities) < 2:
            raise HTTPException(status_code=400, detail="At least 2 cities are required")
        
        # Create and execute Tabu Search
        tabu_search = TabuSearchTSP(
            cities=cities,
            tabu_list_size=request.tabu_list_size,
            max_iterations=request.max_iterations
        )
        
        # Run the algorithm (temporarily suppress print output)
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            best_solution = tabu_search.search()
            best_distance = tabu_search.avaliar_solucao(best_solution)
        finally:
            sys.stdout = old_stdout
        
        return TabuSearchResponse(
            best_solution=best_solution,
            best_distance=best_distance,
            algorithm_info=tabu_search.informacoes(),
            iterations_executed=request.max_iterations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing Tabu Search: {str(e)}")

@endpoints.get("/tabu_search")
async def get_tabu_search_info():
    """
    Get information about the Tabu Search algorithm.
    """
    return {
        "algorithm": "Tabu Search for TSP",
        "description": "Metaheuristic algorithm for solving Traveling Salesman Problem",
        "parameters": {
            "cities": "List of city coordinates (x, y)",
            "tabu_list_size": "Size of the tabu list (default: 5)",
            "max_iterations": "Maximum number of iterations (default: 100)"
        },
        "usage": "POST to /v1/metaheuristica/tabu_search with city coordinates"
    }
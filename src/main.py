from fastapi import FastAPI # type: ignore
import uvicorn # type: ignore

from src.api.v1 import v1_routers

# Função para registrar rotas
def register_routes(app: FastAPI):
    app.include_router(v1_routers)

# Função para criar a instância do aplicativo
def create_app() -> FastAPI:
    app = FastAPI(
        title="API de heuristica",
        version="0.1.0",
        description="Plataforma para teste de algoritmos"
    )
    register_routes(app)
    return app

# Criação da instância do app
app = create_app()

if __name__ == "__main__":
    # Executa o Uvicorn usando a string de importação
    uvicorn.run("main:app",  # Altere "main" para o nome do seu arquivo, se necessário
                 host="0.0.0.0", 
                 port=8000,
                 reload=True)
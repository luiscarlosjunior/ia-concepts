# Metaheuristics - Algoritmos Metaheurísticos

Este módulo contém implementações de algoritmos metaheurísticos para otimização.

## 📚 Algoritmos Implementados

### Simulated Annealing (Recozimento Simulado)
- **`simulated_annealing.py`**: Implementação base do algoritmo
- **`simulated_annealing_visualization.py`**: Versão com visualização gráfica

O Simulated Annealing é uma metaheurística inspirada no processo de recozimento de metais. 
É usado para encontrar soluções aproximadas em problemas de otimização.

### Tabu Search (Busca Tabu)

Implementações consolidadas e especializadas:

- **`tabu_search_base.py`**: Classe base genérica e configurável
  - Permite customização das funções de vizinhança e avaliação
  - Ideal para criar implementações específicas de domínio
  - Substituiu a antiga `tabu_search.py` e `tabu_search_generic.py`

- **`tabu_search_tsp.py`**: Implementação específica para TSP (Problema do Caixeiro Viajante)
  - Usa coordenadas cartesianas (x, y)
  - Calcula distâncias Euclidianas
  - Vizinhança por troca de cidades

- **`tabu_search_graph.py`**: Implementação para grafos com distâncias pré-definidas
  - Trabalha com estruturas de grafo
  - Valida conexões entre nós
  - Vizinhança respeitando arestas do grafo

## 🔧 Como Usar

### Simulated Annealing
```python
from src.algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing

# Criar instância e executar
sa = SimulatedAnnealing(...)
best_solution = sa.run()
```

### Tabu Search Base (Genérico)
```python
from src.algorithms.metaheuristics.tabu_search_base import TabuSearchBase

# Definir funções customizadas
def my_neighbor_func(solution):
    # Gerar vizinhos
    pass

def my_eval_func(solution):
    # Avaliar solução
    pass

# Usar classe base
tabu = TabuSearchBase(
    initial_solution=initial,
    tabu_list_size=5,
    max_iterations=100,
    neighbor_func=my_neighbor_func,
    eval_func=my_eval_func
)
best = tabu.search()
```

### Tabu Search TSP
```python
from src.algorithms.metaheuristics.tabu_search_tsp import TabuSearchTSP

cities = [(0, 0), (10, 20), (30, 15), ...]
tabu_tsp = TabuSearchTSP(cities, tabu_list_size=5, max_iterations=100)
best_route = tabu_tsp.search()
```

### Tabu Search Graph
```python
from src.algorithms.metaheuristics.tabu_search_graph import TabuSearchTSPGraph

graph = {
    "A": [("B", 10), ("C", 15)],
    "B": [("A", 10), ("D", 12)],
    ...
}
tabu_graph = TabuSearchTSPGraph(graph, tabu_list_size=5, max_iterations=100)
best_route = tabu_graph.search()
```

## 📝 Notas de Organização

### Consolidação Realizada
- **Removido**: `tabu_search.py` (exemplo genérico básico, funcionalidade incluída em `tabu_search_base.py`)
- **Renomeado**: `tabu_search_generic.py` → `tabu_search_base.py` (nome mais claro)
- **Mantidos**: Implementações especializadas (TSP, Graph) para problemas específicos

Esta organização melhora o aprendizado ao:
1. Separar implementação base de exemplos especializados
2. Evitar duplicação de código
3. Facilitar a criação de novas implementações especializadas

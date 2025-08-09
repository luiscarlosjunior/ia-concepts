# Tabu Search para Problema do Caixeiro Viajante (TSP)

## Descrição

O Tabu Search (Busca Tabu) é uma metaheurística desenvolvida por Fred Glover que utiliza uma lista tabu para evitar que o algoritmo fique preso em mínimos locais. É especialmente eficaz para resolver problemas de otimização combinatória como o Problema do Caixeiro Viajante (TSP).

## Como Funciona

### Conceitos Principais

1. **Lista Tabu**: Mantém um histórico de soluções recentemente visitadas para evitar ciclos
2. **Vizinhança**: Conjunto de soluções obtidas através de pequenas modificações na solução atual
3. **Aspiração**: Critério que permite aceitar soluções tabu se elas forem melhores que a melhor solução conhecida

### Algoritmo

1. **Inicialização**: 
   - Gera uma solução inicial aleatória
   - Inicializa a lista tabu vazia
   - Define a melhor solução como a solução inicial

2. **Iteração Principal**:
   - Gera todos os vizinhos da solução atual (permutações de duas cidades)
   - Seleciona o melhor vizinho que não esteja na lista tabu
   - Atualiza a solução atual
   - Adiciona a nova solução à lista tabu
   - Remove soluções antigas da lista tabu se exceder o tamanho máximo
   - Atualiza a melhor solução se necessário

3. **Critério de Parada**: Executa por um número máximo de iterações

### Parâmetros

- **cities**: Lista de coordenadas das cidades (x, y)
- **tabu_list_size**: Tamanho máximo da lista tabu (padrão: 5)
- **max_iterations**: Número máximo de iterações (padrão: 100)

## Vantagens

- Evita mínimos locais através da lista tabu
- Explora sistematicamente o espaço de soluções
- Funciona bem para problemas de otimização combinatória
- Relativamente simples de implementar

## Desvantagens

- Pode ser computacionalmente custoso para problemas grandes
- A eficácia depende da escolha adequada dos parâmetros
- Não garante encontrar a solução ótima global

## Exemplo de Uso via API

### Requisição POST

```json
{
  "cities": [
    {"x": 0, "y": 0},
    {"x": 1, "y": 1},
    {"x": 2, "y": 0},
    {"x": 1, "y": -1}
  ],
  "tabu_list_size": 5,
  "max_iterations": 100
}
```

### Resposta

```json
{
  "best_solution": [0, 2, 1, 3],
  "best_distance": 5.656854249492381,
  "algorithm_info": {
    "algoritmo": "tabu_search_tsp",
    "parametros": {
      "iteracoes": 100,
      "tabu_size": 5
    },
    "solucao_inicial": [1, 0, 3, 2],
    "melhor_solucao": [0, 2, 1, 3]
  },
  "iterations_executed": 100
}
```

## Implementação

A implementação está localizada em `src/algorithms/single/tabusearch_tsm.py` e utiliza:

- **Estruturas de dados**: Pandas DataFrame para lista tabu e histórico
- **Geração de vizinhos**: Troca de posições entre duas cidades
- **Avaliação**: Cálculo da distância euclidiana total do percurso
- **Persistência**: Salvamento do histórico em JSON

## Endpoint da API

- **GET** `/v1/metaheuristica/tabu_search` - Informações sobre o algoritmo
- **POST** `/v1/metaheuristica/tabu_search` - Execução do algoritmo com parâmetros personalizados

## Referências

- Glover, F. (1989). Tabu Search—Part I. ORSA Journal on Computing, 1(3), 190-206.
- Glover, F. (1990). Tabu Search—Part II. ORSA Journal on Computing, 2(1), 4-32.
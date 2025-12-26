import random
import pandas as pd
import math
from src.utils.save_df_to_json import SaveFile


class TabuSearchTSPGraph:
    def __init__(self, graph, tabu_list_size, max_iterations):
        """
        Args:
            graph (dict): Grafo onde as chaves são cidades e os valores são listas de tuplas
                          com (cidade_destino, distância).
            tabu_list_size (int): Tamanho máximo da lista tabu.
            max_iterations (int): Número de iterações para a busca.
        """
        self.graph = graph
        self.n_cities = len(graph)
        self.solucao_inicial = list(graph.keys())  # Ordem inicial com todas as cidades
        random.shuffle(self.solucao_inicial)  # Embaralha a solução inicial
        self.current_solution = self.solucao_inicial[:]
        self.best_solution = self.solucao_inicial[:]
        self.tabu_list = pd.DataFrame(columns=["solution", "value"])
        self.tabu_list_size = tabu_list_size
        self.max_iterations = max_iterations
        self.history = pd.DataFrame(columns=["iteration", "best_solution", "value"])

    def informacoes(self):
        return {
            "algoritmo": "tabu_search_tsp_graph",
            "parametros": {"iteracoes": self.max_iterations, "tabu_size": self.tabu_list_size},
            "solucao_inicial": self.solucao_inicial,
            "melhor_solucao": self.best_solution,
        }

    def calcular_distancia(self, city1, city2):
        """
        Obtém a distância entre duas cidades no grafo.
        """
        for neighbor, distance in self.graph[city1]:
            if neighbor == city2:
                return distance
        # Caso não haja conexão direta
        return float("inf")

    def avaliar_solucao(self, solution):
        """
        Avalia a solução no TSP, calculando a soma das distâncias.
        """
        total_distance = 0
        for i in range(self.n_cities - 1):
            total_distance += self.calcular_distancia(solution[i], solution[i + 1])
        # Adiciona a distância entre a última cidade e a primeira (volta ao início)
        total_distance += self.calcular_distancia(solution[-1], solution[0])
        return total_distance

    def generate_neighbors(self, solution):
        """
        Gera vizinhos permutando duas cidades do caminho atual,
        mas respeitando as conexões do grafo.
        """
        neighbors = []
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                # Apenas cria um vizinho se a troca gerar uma rota válida
                if (
                    solution[j] in [n[0] for n in self.graph[solution[i]]]
                    and solution[i] in [n[0] for n in self.graph[solution[j]]]
                ):
                    neighbor = solution[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append(neighbor)
        return neighbors

    def search(self):
        for iteration in range(self.max_iterations):
            neighbors = self.generate_neighbors(self.current_solution)
            best_neighbor = None
            best_neighbor_value = float("inf")

            for neighbor in neighbors:
                # Verifica se o vizinho não está na lista tabu
                if not self.tabu_list["solution"].apply(lambda x: x == neighbor).any():
                    neighbor_value = self.avaliar_solucao(neighbor)
                    if neighbor_value < best_neighbor_value:
                        best_neighbor = neighbor
                        best_neighbor_value = neighbor_value

            # Atualiza a solução atual e a lista tabu
            if best_neighbor is not None:
                self.current_solution = best_neighbor
                if best_neighbor_value < self.avaliar_solucao(self.best_solution):
                    self.best_solution = best_neighbor

                # Adiciona à lista tabu e mantém o tamanho
                new_entry = pd.DataFrame(
                    {"solution": [best_neighbor], "value": [best_neighbor_value]}
                )
                self.tabu_list = pd.concat([self.tabu_list, new_entry], ignore_index=True)
                if len(self.tabu_list) > self.tabu_list_size:
                    self.tabu_list = self.tabu_list.iloc[1:]

            # Registra o histórico
            self.history = pd.concat(
                [
                    self.history,
                    pd.DataFrame(
                        {
                            "iteration": [iteration],
                            "best_solution": [self.best_solution],
                            "value": [self.avaliar_solucao(self.best_solution)],
                        }
                    ),
                ],
                ignore_index=True,
            )

            print(
                f"Iteration {iteration}: Best Solution = {self.best_solution}, "
                f"Value = {self.avaliar_solucao(self.best_solution)}"
            )

        SaveFile().save_json_to_file(df=self.history, path="tabu_search_tsp_graph", informacao=self.informacoes())
        return self.best_solution


if __name__ == "__main__":
    # Exemplo de uso com um grafo
    # Grafo definido como {cidade: [(cidade_destino, distância), ...]}
    graph = {
        "A": [("B", 10), ("C", 15)],
        "B": [("A", 10), ("D", 12)],
        "C": [("A", 15), ("D", 10)],
        "D": [("B", 12), ("C", 10)],
    }
    
    tabu_search_graph = TabuSearchTSPGraph(graph, tabu_list_size=5, max_iterations=100)
    best_solution = tabu_search_graph.search()
    
    print(f"\nBest solution found: {best_solution}")
    print("\nTabu List (Final):")
    print(tabu_search_graph.tabu_list)
    print("\nSearch History:")
    print(tabu_search_graph.history)

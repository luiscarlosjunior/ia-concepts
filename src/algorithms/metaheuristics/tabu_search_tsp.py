import random
import pandas as pd
import math
from src.utils.save_df_to_json import SaveFile

class TabuSearchTSP:
    def __init__(self, cities, tabu_list_size, max_iterations):
        """
        Args:
            cities (list of tuple): Lista de coordenadas das cidades (x, y).
            tabu_list_size (int): Tamanho máximo da lista tabu.
            max_iterations (int): Número de iterações para a busca.
        """
        self.cities = cities  # Coordenadas das cidades (x, y)
        self.n_cities = len(cities)
        self.solucao_inicial = list(range(self.n_cities))  # Ordem inicial aleatória
        random.shuffle(self.solucao_inicial)  # Embaralha a solução inicial
        self.current_solution = self.solucao_inicial[:]
        self.best_solution = self.solucao_inicial[:]
        self.tabu_list = pd.DataFrame(columns=["solution", "value"])
        self.tabu_list_size = tabu_list_size
        self.max_iterations = max_iterations
        self.history = pd.DataFrame(columns=["iteration", "best_solution", "value"])

    def informacoes(self):
        return {
            "algoritmo": "tabu_search_tsp", 
            "parametros": 
                {"iteracoes": 100, "tabu_size": 5},
            "solucao_inicial": self.solucao_inicial,
            "melhor_solucao": self.best_solution
        }

    def calcular_distancia(self, city1, city2):
        """
        Calcula a distância Euclidiana entre duas cidades.
        """
        x1, y1 = self.cities[city1]
        x2, y2 = self.cities[city2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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
        Gera vizinhos permutando duas cidades do caminho atual.
        """
        neighbors = []
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                neighbor = solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Troca duas cidades
                neighbors.append(neighbor)
        return neighbors

    def search(self):
        for iteration in range(self.max_iterations):
            neighbors = self.generate_neighbors(self.current_solution)
            best_neighbor = None
            best_neighbor_value = float('inf')

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

        SaveFile().save_json_to_file(df=self.history, path="tabu_search_tsp", informacao=self.informacoes())
        return self.best_solution

if __name__ == "__main__":
    # Exemplo de uso
    # Definindo as coordenadas das cidades (x, y)
    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
    
    tabu_search_tsp = TabuSearchTSP(cities, tabu_list_size=5, max_iterations=100)
    best_solution = tabu_search_tsp.search()
    
    print(f"\nBest solution found: {best_solution}")
    print("\nTabu List (Final):")
    print(tabu_search_tsp.tabu_list)
    print("\nSearch History:")
    print(tabu_search_tsp.history)

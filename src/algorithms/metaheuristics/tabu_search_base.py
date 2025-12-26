import random
import pandas as pd

class TabuSearchBase:
    def __init__(self, initial_solution, tabu_list_size, max_iterations, neighbor_func, eval_func, stop_func=None):
        """
        Args:
            initial_solution (list): A solução inicial fornecida pelo usuário.
            tabu_list_size (int): Tamanho da lista tabu.
            max_iterations (int): Número máximo de iterações.
            neighbor_func (function): Função que gera vizinhos a partir de uma solução.
            eval_func (function): Função que avalia a qualidade da solução.
            stop_func (function, optional): Função de parada personalizada. (default: None)
        """
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.tabu_list = pd.DataFrame(columns=["solution", "value"])
        self.tabu_list_size = tabu_list_size
        self.max_iterations = max_iterations
        self.neighbor_func = neighbor_func
        self.eval_func = eval_func
        self.stop_func = stop_func
        self.history = pd.DataFrame(columns=["iteration", "best_solution", "value"])

    def informacoes(self):
        """
        Função para retornar os parâmetros e soluções para análise externa.
        """
        return {
            "algoritmo": "tabu_search_base",
            "solucao_inicial": self.current_solution,
            "melhor_solucao": self.best_solution
        }

    def search(self):
        for iteration in range(self.max_iterations):
            # Gerar vizinhos usando a função fornecida
            neighbors = self.neighbor_func(self.current_solution)
            best_neighbor = None
            best_neighbor_value = float('inf')

            for neighbor in neighbors:
                # Avaliar a solução do vizinho usando a função fornecida
                neighbor_value = self.eval_func(neighbor)
                
                # Verificar se o vizinho não está na lista tabu
                if not self.tabu_list["solution"].apply(lambda x: x == neighbor).any():
                    if neighbor_value < best_neighbor_value:
                        best_neighbor = neighbor
                        best_neighbor_value = neighbor_value

            # Atualiza a solução atual e a melhor solução encontrada
            if best_neighbor is not None:
                self.current_solution = best_neighbor
                if best_neighbor_value < self.eval_func(self.best_solution):
                    self.best_solution = best_neighbor

                # Atualiza a lista tabu
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
                            "value": [self.eval_func(self.best_solution)],
                        }
                    ),
                ],
                ignore_index=True,
            )

            # Critério de parada
            if self.stop_func and self.stop_func(self.best_solution):
                print(f"Parando a busca na iteração {iteration} devido ao critério de parada.")
                break

            print(
                f"Iteração {iteration}: Melhor Solução = {self.best_solution}, "
                f"Valor = {self.eval_func(self.best_solution)}"
            )

        return self.best_solution

if __name__ == "__main__":
    import math
    
    # Exemplo de uso com um problema de Caixeiro Viajante
    
    # Função para gerar vizinhos (trocar duas cidades de lugar)
    def tsp_neighbors(solution):
        neighbors = []
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                neighbor = solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Troca de cidades
                neighbors.append(neighbor)
        return neighbors
    
    # Função para avaliar a solução (distância total)
    def tsp_evaluate(solution, cities):
        total_distance = 0
        for i in range(len(solution) - 1):
            total_distance += math.sqrt((cities[solution[i + 1]][0] - cities[solution[i]][0]) ** 2 + (cities[solution[i + 1]][1] - cities[solution[i]][1]) ** 2)
        # Volta à cidade inicial
        total_distance += math.sqrt((cities[solution[-1]][0] - cities[solution[0]][0]) ** 2 + (cities[solution[-1]][1] - cities[solution[0]][1]) ** 2)
        return total_distance
    
    # Critério de parada simples (por exemplo, quando a solução não melhora em 10 iterações consecutivas)
    def stop_criteria(solution):
        return False  # Apenas exemplo, pode ser configurado para parar quando necessário
    
    # Exemplo de execução
    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
    initial_solution = list(range(len(cities)))
    random.shuffle(initial_solution)
    
    tabu_search = TabuSearchBase(
        initial_solution=initial_solution,
        tabu_list_size=5,
        max_iterations=100,
        neighbor_func=tsp_neighbors,
        eval_func=lambda solution: tsp_evaluate(solution, cities),
        stop_func=stop_criteria
    )
    
    best_solution = tabu_search.search()
    print(f"\nBest solution found: {best_solution}")

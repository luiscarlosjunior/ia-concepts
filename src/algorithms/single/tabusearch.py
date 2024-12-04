import random
import pandas as pd
from src.utils.save_df_to_json import SaveFile

class TabuSearch:
    def __init__(self, initial_solution, tabu_list_size, max_iterations):
        """
        Args:
            initial_solution (list): The starting solution.
            tabu_list_size (int): Maximum size of the tabu list.
            max_iterations (int): Number of iterations for the search.
        """
        self.solucao_inicial = initial_solution
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.tabu_list = pd.DataFrame(columns=["solution", "value"])
        self.tabu_list_size = tabu_list_size
        self.max_iterations = max_iterations
        self.history = pd.DataFrame(columns=["iteration", "best_solution", "value"])

    def informacoes(self):
        return {
            "algoritmo": "tabu_search", 
            "parametros": 
                {"iteracoes": 100, "tabu_size": 5},
            "solucao_inicial": self.solucao_inicial,
            "melhor_solucao": self.best_solution
        }
        
    
    def generate_neighbors(self, solution):
        """
        Generate neighboring solutions by making small changes to the current solution.
        """
        neighbors = []
        for i in range(len(solution)):
            neighbor = solution[:]
            neighbor[i] = random.randint(0, 100)  # Example change
            neighbors.append(neighbor)
        return neighbors

    def evaluate_solution(self, solution):
        """
        Evaluate the solution. Lower value is considered better.
        """
        return sum(solution)

    def search(self):
        for iteration in range(self.max_iterations):
            neighbors = self.generate_neighbors(self.current_solution)
            best_neighbor = None
            best_neighbor_value = float('inf')

            for neighbor in neighbors:
                neighbor_value = self.evaluate_solution(neighbor)
                # Check if the neighbor is not in the tabu list
                if not self.tabu_list["solution"].apply(lambda x: x == neighbor).any():
                    if neighbor_value < best_neighbor_value:
                        best_neighbor = neighbor
                        best_neighbor_value = neighbor_value

            # Update current solution and tabu list
            if best_neighbor is not None:
                self.current_solution = best_neighbor
                if best_neighbor_value < self.evaluate_solution(self.best_solution):
                    self.best_solution = best_neighbor

                # Add to tabu list and maintain its size
                new_entry = pd.DataFrame(
                    {"solution": [best_neighbor], "value": [best_neighbor_value]}
                )
                self.tabu_list = pd.concat([self.tabu_list, new_entry], ignore_index=True)
                if len(self.tabu_list) > self.tabu_list_size:
                    self.tabu_list = self.tabu_list.iloc[1:]

            # Record history for analysis
            self.history = pd.concat(
                [
                    self.history,
                    pd.DataFrame(
                        {
                            "iteration": [iteration],
                            "best_solution": [self.best_solution],
                            "value": [self.evaluate_solution(self.best_solution)],
                        }
                    ),
                ],
                ignore_index=True,
            )

            print(
                f"Iteration {iteration}: Best Solution = {self.best_solution}, "
                f"Value = {self.evaluate_solution(self.best_solution)}"
            )
        
        SaveFile().save_json_to_file(df=self.history, path="tabu_search", informacao=self.informacoes())
        return self.best_solution

# Example usage
initial_solution = [random.randint(0, 100) for _ in range(10)]
tabu_search = TabuSearch(initial_solution, tabu_list_size=5, max_iterations=50)
best_solution = tabu_search.search()

print(f"\nBest solution found: {best_solution}")
print("\nTabu List (Final):")
print(tabu_search.tabu_list)
print("\nSearch History:")
print(tabu_search.history)

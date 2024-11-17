import matplotlib
matplotlib.use('TkAgg')  # Força o backend interativo TkAgg

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class HillClimbingVisualization():
        
    # Função para calcular a distância total de um caminho
    def calculate_total_distance(self, path, distance_matrix):
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += distance_matrix[path[i], path[i + 1]]
        total_distance += distance_matrix[path[-1], path[0]]  # Retorno ao ponto inicial
        return total_distance

    # Função para gerar uma matriz de distâncias aleatória
    def generate_distance_matrix(self, num_cities):
        coordinates = np.random.rand(num_cities, 2)
        distance_matrix = np.sqrt(((coordinates[:, np.newaxis] - coordinates[np.newaxis, :]) ** 2).sum(axis=2))
        return distance_matrix, coordinates

    # Função de Hill Climbing para o problema do caixeiro viajante
    def hill_climbing(self, distance_matrix):
        num_cities = distance_matrix.shape[0]
        current_path = np.arange(num_cities)
        np.random.shuffle(current_path)
        current_distance = self.calculate_total_distance(current_path, distance_matrix)
        
        iteration = 0
        while True:
            iteration += 1
            best_distance = current_distance
            best_path = current_path.copy()
            
            for i in range(num_cities):
                for j in range(i + 1, num_cities):
                    new_path = current_path.copy()
                    new_path[i:j] = new_path[i:j][::-1]  # Reversão de segmento
                    new_distance = self.calculate_total_distance(new_path, distance_matrix)
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_path = new_path
            
            if best_distance >= current_distance:
                break
            
            current_path = best_path
            current_distance = best_distance
            yield current_path, current_distance, iteration

    # Função para plotar as cidades e o caminho
    def plot_path(self, coordinates, path, iteration):
        plt.clf()
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red')
        for i in range(len(path)):
            plt.annotate(str(i), (coordinates[i, 0], coordinates[i, 1]))
        path_coordinates = coordinates[path]
        plt.plot(path_coordinates[:, 0], path_coordinates[:, 1], 'b-')
        plt.plot([path_coordinates[-1, 0], path_coordinates[0, 0]], [path_coordinates[-1, 1], path_coordinates[0, 1]], 'b-')
        plt.title(f'Iteration: {iteration}')

    # Função para adaptar o gerador ao FuncAnimation
    def update_frame(self, frame, hill_climbing_gen, coordinates):
        try:
            path, _, iteration = next(hill_climbing_gen)
            self.plot_path(coordinates, path, iteration)
        except StopIteration:
            pass  # O gerador terminou; nenhuma ação necessária

    # Função principal
    def main(self):
        num_cities = 10
        distance_matrix, coordinates = self.generate_distance_matrix(num_cities)
        
        fig = plt.figure()
        hill_climbing_gen = self.hill_climbing(distance_matrix)  # Cria o gerador
        ani = animation.FuncAnimation(
            fig,
            self.update_frame,
            fargs=(hill_climbing_gen, coordinates),
            frames=100,
            repeat=False
        )
        plt.show()
        return ani

# if __name__ == "__main__":
#     ani = main()

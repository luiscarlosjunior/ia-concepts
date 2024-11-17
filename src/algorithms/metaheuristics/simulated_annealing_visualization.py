import math
import random

import pandas as pd # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib.animation import FuncAnimation # type: ignore

from src.utils.calc_distance import CalcDistancias as calc_dist
from src.utils.save_df_to_json import SaveFile

class SimulatedAnneling():
    
    def __init__(self, cities, inicial_temp, cooling_rate, stopping_temp):
        """
        1. **cities**: 
        - **Descrição**: Uma lista de cidades que o algoritmo irá percorrer.
        - **Contexto**: No problema do caixeiro-viajante, por exemplo, essa lista representa os pontos que precisam ser visitados. Cada cidade pode ser representada por coordenadas (x, y).

        2. **initial_temp**:
        - **Descrição**: A temperatura inicial do sistema.
        - **Contexto**: No Simulated Annealing, a temperatura controla a probabilidade de aceitar soluções piores no início do processo. Uma temperatura inicial alta permite explorar mais o espaço de soluções, ajudando a evitar mínimos locais.

        3. **cooling_rate**:
        - **Descrição**: A taxa de resfriamento da temperatura.
        - **Contexto**: Este parâmetro determina o quanto a temperatura diminui a cada iteração. Um valor de resfriamento mais lento (próximo de 1) permite uma exploração mais detalhada, enquanto um resfriamento mais rápido pode levar a uma convergência mais rápida, mas com maior risco de ficar preso em mínimos locais.

        4. **stopping_temp**:
        - **Descrição**: A temperatura de parada do algoritmo.
        - **Contexto**: Quando a temperatura atinge esse valor, o algoritmo para. Isso significa que o sistema está "frio" o suficiente e não faz mais sentido continuar a busca, pois as mudanças nas soluções serão mínimas.

        Aqui está o código com os comentários explicativos:

        Args:
            cities (_type_): Lista de cidades a serem percorridas
            inicial_temp (_type_): Temperatura inicial do sistema
            cooling_rate (_type_): Taxa de resfriamento da temperatura
            stopping_temp (_type_): Temperatura de parada do algoritmo
        """
        self.cities = cities
        self.initial_temp = inicial_temp
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temp
        
        self.steps = []  # Lista para armazenar o passo a passo do algoritmo
        self.iteration = 0
        
        self.best_solution = None  # Melhor caminho
        self.best_distance = float('inf')  # Menor distância

    def total_distance(self, cities, tour):
        return sum(calc_dist.euclidean_distance(cities[tour[i]], cities[tour[i + 1]]) for i in range(len(tour) - 1)) + calc_dist.euclidean_distance(cities[tour[-1]], cities[tour[0]])
    
    def simulated_annealing(self):
        cities = self.cities
        current_temp = self.initial_temp
        stopping_temp = self.stopping_temp
        cooling_rate = self.cooling_rate
        
        current_solution = list(range(len(cities)))
        random.shuffle(current_solution)
        self.best_solution = list(current_solution)
        self.best_distance = self.total_distance(cities, self.best_solution)

        # Registra o passo inicial
        self._record_step(self.iteration, current_temp, self.best_solution, self.best_distance)

        while current_temp > stopping_temp:
            self.iteration += 1

            # Gera uma nova solução por inversão de segmento
            new_solution = list(current_solution)
            l = random.randint(2, len(cities) - 1)
            i = random.randint(0, len(cities) - l)
            new_solution[i:i + l] = reversed(new_solution[i:i + l])

            current_distance = self.total_distance(cities, current_solution)
            new_distance = self.total_distance(cities, new_solution)

            # Decide se aceita a nova solução
            if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / current_temp):
                current_solution = list(new_solution)
                current_distance = new_distance

            # Atualiza a melhor solução se necessário
            if current_distance < self.best_distance:
                self.best_solution = list(current_solution)
                self.best_distance = current_distance

            # Registra o passo atual
            self._record_step(self.iteration, current_temp, current_solution, current_distance)

            # Resfria a temperatura
            current_temp *= cooling_rate

        # Converte os passos registrados para DataFrame e salva como JSON
        df_steps = pd.DataFrame(self.steps)
        SaveFile().save_json_to_file(df=df_steps, path="simulated_annealing/tsm")
        result_json = df_steps.to_json(orient='records', lines=True)
        
        # Exibe o melhor resultado
        print(f"Melhor solução: {self.best_solution}")
        print(f"Menor distância: {self.best_distance}")
        
        return result_json, self.best_solution, self.best_distance

    def _record_step(self, iteration, current_temp, current_tour, current_distance):
        """Registra passo a passo do algoritmo

        Args:
            iteration (_type_): _description_
            current_temp (_type_): _description_
            current_tour (_type_): _description_
            current_distance (_type_): _description_
        """
        self.steps.append({
            'iteration': iteration,
            'temperature': current_temp,
            'tour': current_tour,
            'distance': current_distance
        })
    
    def visualize_progress(self):
        """
        Gera visualizações do progresso do algoritmo.
        """
        # Extraindo dados das etapas
        df_steps = pd.DataFrame(self.steps)
        
        # Visualizando a evolução da distância ao longo das iterações
        plt.figure(figsize=(12, 6))
        plt.plot(df_steps['iteration'], df_steps['distance'], label='Distância', color='blue')
        plt.xlabel('Iterações')
        plt.ylabel('Distância')
        plt.title('Evolução da Distância ao Longo das Iterações')
        plt.grid(True)
        plt.legend()
        plt.show()

        """
        Plota o melhor caminho encontrado com cores para as cidades e rótulos de letras.
        """
        best_tour = self.best_solution
        best_cities = [self.cities[i] for i in best_tour] + [self.cities[best_tour[0]]]  # Fechando o ciclo
        x, y = zip(*best_cities)

        # Criando uma lista de letras para as cidades
        labels = [chr(65 + i) for i in range(len(self.cities))]

        # Criando a figura
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', linestyle='-', color='red', label='Melhor Caminho')

        # Adicionando as cidades como pontos coloridos
        for i, (xi, yi) in enumerate(zip(x, y)):
            # Verificando se o índice está dentro do limite de rótulos
            label_index = best_tour[i] if i < len(best_tour) else best_tour[0]  # Índice correto da cidade original
            plt.scatter(xi, yi, color='blue', zorder=5)  # Cidades coloridas em azul
            plt.text(xi, yi, f'{labels[label_index]}', color='black', fontsize=12, ha='right', va='bottom')

        # Adicionando título e rótulos
        plt.title('Melhor Caminho Encontrado com Cidades Rotuladas')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def animate_progress(self):
        """
        Cria uma animação mostrando os caminhos gerados ao longo das iterações.
        """
        # Configurando o estilo do seaborn
        sns.set_theme(style="whitegrid")

        # Criando a figura
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Evolução dos Caminhos Encontrados')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Extraindo os passos e preparando os dados
        steps = self.steps
        paths = [step['tour'] for step in steps]

        # Função de atualização para cada frame
        def update(frame):
            ax.clear()
            ax.set_title(f'Iteração {steps[frame]["iteration"]}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # Obter o caminho atual e adicionar o ciclo fechado
            current_tour = paths[frame]
            current_cities = [self.cities[i] for i in current_tour] + [self.cities[current_tour[0]]]
            x, y = zip(*current_cities)

            sns.lineplot(x=x, y=y, marker='o', ax=ax, color="blue", label="Caminho Atual")
            ax.legend()

        # Criando a animação
        anim = FuncAnimation(fig, update, frames=len(steps), interval=300, repeat=False)

        # Exibir ou salvar animação
        plt.show()
        
# Example usage
if __name__ == "__main__":
    cities = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 0), (2, 2)]
    initial_temp = 10000
    cooling_rate = 0.995
    stopping_temp = 1e-8

    sa = SimulatedAnneling(cities=cities, inicial_temp=initial_temp, cooling_rate=cooling_rate, stopping_temp=stopping_temp)
    result = sa.simulated_annealing()
    sa.visualize_progress()
    #print(result)
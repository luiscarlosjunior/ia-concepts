import math
import random

import pandas as pd # type: ignore

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
        """
        O algoritmo de recozimento simulado (SA) foi apresentado independentemente por 
        Kirkpatrick et al. (O algoritmo de recozimento simulado (SA) foi apresentado independentemente 
        por Kirkpatrick et al. () e Černý et al. (Černý, 1982, ) e Černý et al. (). 
        Alguns métodos semelhantes também podem ser encontrados em outros estudos 
        (Khachaturyan et al., 1981; Pincus, 1970). 
        A ideia básica do SA é ocasionalmente aceitar soluções que não melhorem, 
        o que significa que o SA nem sempre mudará para uma solução melhor.

        Returns:
            _type_: _description_
        """
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
        
# Example usage
if __name__ == "__main__":
    cities = [(0, 0), (1, 3), (4, 3), (6, 1), (3, 0), (2, 2)]
    initial_temp = 10000
    cooling_rate = 0.995
    stopping_temp = 1e-8

    sa = SimulatedAnneling(cities=cities, inicial_temp=initial_temp, cooling_rate=cooling_rate, stopping_temp=stopping_temp)
    result = sa.simulated_annealing()
    #print(result)
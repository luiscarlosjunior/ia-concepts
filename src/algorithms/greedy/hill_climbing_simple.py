import random
import pandas as pd # type: ignore
import os

from datetime import datetime

from src.utils.calc_distance import CalcDistancias as calc_dist
from src.utils.save_df_to_json import SaveFile

class HillClimbing():
    def __init__(self):
        self.steps = []  # Lista para armazenar o passo a passo do algoritmo
        self.iteration = 0
    
    def total_distance(self, tour, cities):
        """
        tour é uma lista que contém a ordem das cidades a serem visitadas.
        cities é uma lista de coordenadas de todas as cidades.
        distance(cities[tour[i]], cities[tour[i - 1]]) calcula a distância entre duas cidades consecutivas no tour, usando a função distance que foi explicada anteriormente.
        sum(...) soma as distâncias de todas as cidades no tour.

        Args:
            tour (_type_): é uma lista que contém a ordem das cidades a serem visitadas.
            cities (_type_): lista de coordenadas de todas as cidades

        Returns:
            _type_: _description_
        """
        # Esta função calcula a distância total de um "tour" (um percurso) que passa por todas as cidades.
        return sum(calc_dist.euclidean_distance(cities[tour[i]], cities[tour[i - 1]]) for i in range(len(tour)))
    
    def record_step(self, current_tour, current_distance):
        """
        Registra o passo atual no algoritmo no DataFrame.
        
        Args:
            current_tour (_type_): a ordem das cidades no tour atual.
            current_distance (_type_): a distância total do tour atual.
        """
        self.iteration += 1
        self.steps.append({
            'iteration': self.iteration,
            'tour': current_tour,
            'distance': current_distance
        })
    
    def hill_climbing(self, cities):
        """
        Objetivo: Esta é a função principal do algoritmo de Hill Climbing, que tenta encontrar o melhor tour (menor distância) para visitar todas as cidades.
        Como funciona:
        list(range(len(cities))) cria uma lista de índices representando todas as cidades. Se houver 10 cidades, essa lista será [0, 1, 2, ..., 9].
        random.shuffle(current_tour) embaralha essa lista, criando uma ordem aleatória das cidades, o que simula o ponto de partida para o algoritmo de Hill Climbing.
        total_distance(current_tour, cities) calcula a distância total do tour gerado.
        print(f"Initial tour: {current_tour} with distance: {current_distance}") imprime o tour inicial e sua distância.

        Args:
            cities (_type_): _description_

        Returns:
            _type_: _description_
        """
        current_tour = list(range(len(cities)))
        random.shuffle(current_tour)
        current_distance = self.total_distance(current_tour, cities)
        
        print(f"Initial tour: {current_tour} with distance: {current_distance}")
        
        # Registra o passo inicial
        self.record_step(current_tour, current_distance)
        
        while True:
            neighbors = []
            for i in range(len(current_tour)):
                for j in range(i + 1, len(current_tour)):
                    neighbor = current_tour[:]
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append(neighbor)
            
            next_tour = min(neighbors, key=lambda tour: self.total_distance(tour, cities))
            next_distance = self.total_distance(next_tour, cities)
            
            if next_distance >= current_distance:
                break
            
            # Atualiza o tour atual e registra o novo passo
            current_tour, current_distance = next_tour, next_distance
            print(f"New tour: {current_tour} with distance: {current_distance}")
            
            self.record_step(current_tour, current_distance)
        
        # Converte os passos registrados para DataFrame e depois para JSON
        df_steps = pd.DataFrame(self.steps)
        SaveFile().save_json_to_file(df=df_steps, path="hill_climbing\\tsm")
        result_json = df_steps.to_json(orient='records', lines=True)
        return result_json
    
# if __name__ == "__main__":
#     cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
#     best_tour_json = HillClimbing().hill_climbing(cities)
#     print(f"Best tour history in JSON format: {best_tour_json}")

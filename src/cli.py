import random
from src.algorithms.greedy.hill_climbing_visualization import HillClimbingVisualization
from src.algorithms.greedy.hill_climbing_simple import HillClimbing
from src.services.pegar_coordenada import PegarCoordenadaJanela


def hill_climbing(cidades):
    print("Executando Hill Climbing...")
    print(cidades)
    best_tour_json = HillClimbing().hill_climbing(cidades)
    print(f"Best tour history in JSON format: {best_tour_json}")
    #HillClimbingVisualization().main()

def simulated_annealing():
    print("Executando Simulated Annealing...")

def genetic_algorithm():
    print("Executando Algoritmo Genético...")

def main():    
    algorithms = {
        0: hill_climbing,
        1: simulated_annealing,
        2: genetic_algorithm
    }
        
    cidades = get_city_coordinates()
    
    print("Escolha um algoritmo para executar:")
    for key, value in algorithms.items():
        print(f"[{key}] -> {value.__name__}")

    choice = int(input("Digite o número do algoritmo: "))

    if choice in algorithms:
        algorithms[choice](cidades)
    else:
        print("Escolha inválida.")

def get_city_coordinates():
    enter_coordinates = input("Entrar com as coordenadas da cidade? (s/n): ").strip().lower() == 's'
    cidades = []
    if enter_coordinates:
        enter_coordinates_visual = input("Entrar usando janela visual? (s/n): ").strip().lower() == 's'
        if enter_coordinates_visual:
            capturador = PegarCoordenadaJanela()
            cidades = capturador.janela()
        else:
            num_cities = int(input("Digite o número de cidades: "))
            for i in range(num_cities):
                x = int(input(f"Digite a coordenada x da cidade {i+1}: "))
                y = int(input(f"Digite a coordenada y da cidade {i+1}: "))
                cidades.append((x, y))
    else:
        print("Gerando coordenadas aleatórios")
        cidades = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(10)]
    
    print("Coordenadas cidades")
    print(cidades)
    return cidades

if __name__ == "__main__":
    main()
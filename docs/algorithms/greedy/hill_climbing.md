# Hill Climbing: Uma Abordagem Heurística para Otimização

Hill Climbing é um algoritmo heurístico usado em problemas de otimização, onde o objetivo é encontrar a solução ideal (ou quase ideal) em um espaço de busca grande. Ele é amplamente aplicado em problemas como a otimização de funções, o problema do caixeiro viajante (TSP), agendamento e muitos outros. Apesar de sua simplicidade, Hill Climbing é um conceito poderoso e um ponto de partida para entender técnicas mais avançadas de otimização.

---

## **1. O Conceito de Hill Climbing**

O nome "Hill Climbing" (escalada de colinas) reflete a ideia central do algoritmo: dado um ponto inicial em um espaço de busca, o algoritmo tenta se mover para um ponto adjacente que ofereça uma melhoria, como subir uma colina. O processo continua até que nenhuma melhoria seja possível, indicando que o algoritmo alcançou um **pico local** ou global.

O objetivo pode variar:
- **Maximização:** Encontrar o pico mais alto.
- **Minimização:** Encontrar o vale mais profundo.

Em ambos os casos, o princípio é o mesmo: realizar passos incrementais em direção a uma solução melhor.

---

## **2. Funcionamento do Algoritmo**

### **2.1 Passos Básicos**

1. **Ponto de partida:** Escolha uma solução inicial (geralmente de forma aleatória).
2. **Vizinhaça:** Encontre soluções vizinhas que estão "próximas" da solução atual.
3. **Avaliação:** Calcule a qualidade (ou "fitness") de cada solução vizinha.
4. **Movimento:** Mova-se para a melhor solução vizinha, se ela for melhor do que a solução atual.
5. **Convergência:** Pare quando nenhuma melhoria for possível (um ótimo local foi alcançado).

### **2.2 Representação Gráfica**

Imaginemos que o espaço de busca é uma paisagem com colinas e vales. Cada ponto representa uma solução possível, e sua altura representa a qualidade da solução. O algoritmo tenta "subir" essa paisagem para encontrar o ponto mais alto.

---

## **3. Tipos de Hill Climbing**

### **3.1 Simple Hill Climbing**
O algoritmo avalia uma solução vizinha e se move para ela apenas se for melhor que a atual. Este método é vulnerável a ficar preso em ótimos locais.

### **3.2 Steepest-Ascent Hill Climbing**
Entre todas as soluções vizinhas, o algoritmo escolhe a que oferece a maior melhoria. Isso reduz a probabilidade de decisões subótimas, mas aumenta o custo computacional.

### **3.3 Stochastic Hill Climbing**
Escolhe aleatoriamente uma solução vizinha melhor, sem avaliar todas as opções. Este método é mais rápido, mas pode ser menos preciso.

### **3.4 First-Choice Hill Climbing**
Uma variação do Stochastic Hill Climbing que aceita imediatamente a primeira melhoria encontrada. É útil em espaços de busca grandes.

---

## **4. Vantagens e Desvantagens**

### **4.1 Vantagens**
- **Simplicidade:** Fácil de implementar e compreender.
- **Eficiência:** Pode ser rápido para encontrar uma solução aceitável.
- **Sem necessidade de memória:** Requer pouca ou nenhuma memória adicional.

### **4.2 Desvantagens**
- **Ótimos locais:** Pode ficar preso em ótimos locais em vez de encontrar o ótimo global.
- **Falta de visão global:** Não considera o espaço de busca como um todo.
- **Sensibilidade à solução inicial:** O resultado pode variar dependendo do ponto de partida.

---

## **5. Estratégias para Superar Ótimos Locais**

Para mitigar o problema de ótimos locais, várias estratégias podem ser empregadas:
- **Random Restarts:** Executar o algoritmo várias vezes com diferentes pontos de partida.
- **Simulated Annealing:** Adicionar uma probabilidade de aceitar soluções piores para escapar de ótimos locais.
- **Algoritmos Genéticos:** Combinar Hill Climbing com técnicas evolutivas.
- **Tabu Search:** Registrar soluções visitadas para evitar ciclos.

---

## **6. Aplicações do Hill Climbing**

### **6.1 Problema do Caixeiro Viajante (TSP)**
Dado um conjunto de cidades e suas distâncias, o objetivo é encontrar o menor percurso que visite todas as cidades uma vez e retorne ao ponto inicial. O Hill Climbing pode ser usado para gerar uma solução aproximada, especialmente com variações como Steepest-Ascent.

### **6.2 Ajuste de Parâmetros**
É usado para ajustar parâmetros em aprendizado de máquina e sistemas de controle.

### **6.3 Planejamento e Agendamento**
Hill Climbing é útil em problemas de alocação de recursos e agendamento em tempo real.

## **7.Pseudocódigo do Hill Climbing**

```plaintext
FUNÇÃO HillClimbing(Problema):
    1. Inicialize uma solução inicial S
       S ← Gerar uma solução aleatória ou uma inicial definida
    
    2. Avalie a solução inicial
       CustoAtual ← CalcularCusto(S)
    
    3. Enquanto houver melhorias possíveis:
       a. Gere os vizinhos da solução atual
          Vizinhos ← GerarVizinhos(S)
       
       b. Escolha o melhor vizinho
          MelhorVizinho ← Selecionar vizinho com menor custo em Vizinhos
          CustoVizinho ← CalcularCusto(MelhorVizinho)
       
       c. Se o custo do melhor vizinho for melhor que o custo atual:
          i. Atualize a solução atual
             S ← MelhorVizinho
             CustoAtual ← CustoVizinho
       d. Caso contrário:
          i. Termine o algoritmo (nenhum vizinho é melhor que a solução atual)
    
    4. Retorne a melhor solução encontrada e seu custo
       Retornar S, CustoAtual
```

---

### **7.1 - Explicação dos Passos**
1. **Inicialização**:
   - O algoritmo começa com uma solução inicial \( S \), que pode ser gerada aleatoriamente ou definida manualmente.
   - O custo da solução inicial é calculado usando uma função objetivo.

2. **Gerar vizinhos**:
   - Para cada solução \( S \), um conjunto de vizinhos é gerado. Um vizinho é uma solução semelhante a \( S \), mas ligeiramente modificada.
   - Exemplos de modificações incluem trocar elementos, alterar valores ou fazer permutações (no caso do TSP, trocar duas cidades no tour).

3. **Selecionar o melhor vizinho**:
   - Entre todos os vizinhos gerados, escolhe-se aquele que apresenta o menor custo.
   - O custo é calculado usando a mesma função objetivo usada para a solução inicial.

4. **Atualização ou parada**:
   - Se o custo do melhor vizinho for menor que o custo da solução atual, a solução atual é substituída pelo vizinho.
   - Caso contrário, o algoritmo para, pois não há mais melhorias possíveis.

5. **Retorno**:
   - O algoritmo retorna a solução final \( S \) e o custo correspondente.

---

### **7.2 - Exemplo de Aplicação no Problema do Caixeiro Viajante (TSP)**
```plaintext
FUNÇÃO HillClimbingTSP(Cidades):
    1. S ← Tour aleatório das cidades
    2. CustoAtual ← CalcularDistânciaTotal(S, Cidades)

    3. Enquanto verdadeiro:
       a. Vizinhos ← Lista de tours gerados trocando pares de cidades em S
       b. MelhorVizinho ← Vizinho com menor distância total
       c. CustoVizinho ← CalcularDistânciaTotal(MelhorVizinho, Cidades)

       d. Se CustoVizinho < CustoAtual:
             S ← MelhorVizinho
             CustoAtual ← CustoVizinho
          Caso contrário:
             Interrompa o loop
    
    4. Retornar S, CustoAtual
```

---

## **8. Exemplo de Implementação em Python**

```python
import random

def hill_climbing(function, start_point, step_size=0.1, max_iterations=1000):
    current_point = start_point
    current_value = function(current_point)
    
    for _ in range(max_iterations):
        next_point = current_point + random.uniform(-step_size, step_size)
        next_value = function(next_point)
        
        if next_value > current_value:
            current_point, current_value = next_point, next_value
        else:
            break  # Parar se nenhuma melhoria for encontrada
    
    return current_point, current_value

# Função exemplo: f(x) = -(x^2) + 4
result = hill_climbing(lambda x: -(x**2) + 4, start_point=0)
print("Melhor ponto encontrado:", result)
```

---

## **9. Referências Bibliográficas**

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. Pearson.
2. Pearl, J. (1984). *Heuristics: Intelligent Search Strategies for Computer Problem Solving*. Addison-Wesley.
3. Aarts, E., & Korst, J. (1989). *Simulated Annealing and Boltzmann Machines*. Wiley.
4. Yuret, D. (1997). "Hill Climbing and Optimization." *Journal of Artificial Intelligence Research*.

---

## **10. Conclusão**

Hill Climbing é uma técnica simples, gananciosa, mas fundamental, no campo de otimização heurística. Apesar de suas limitações, ele fornece uma base sólida para entender algoritmos mais complexos. A combinação de Hill Climbing com outras abordagens, como algoritmos genéticos e Simulated Annealing, permite explorar soluções ainda mais robustas para problemas desafiadores.
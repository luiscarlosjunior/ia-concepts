# Algoritmos Evolucionários

Os **Algoritmos Evolucionários** (Evolutionary Algorithms - EA) são uma família de algoritmos de otimização e busca inspirados nos mecanismos da evolução biológica. Estes algoritmos utilizam conceitos como seleção natural, reprodução, mutação e recombinação para encontrar soluções para problemas complexos.

![Evolutionary Algorithms Concept](../../images/evolutionary_algorithms_concept.png)

## 🧬 Fundamentos dos Algoritmos Evolucionários

### **Princípios Básicos**

Os algoritmos evolucionários compartilham princípios fundamentais baseados na teoria da evolução de Darwin:

1. **População:** Trabalham com um conjunto de soluções candidatas
2. **Seleção:** Soluções melhores têm maior probabilidade de serem escolhidas
3. **Variação:** Novas soluções são criadas através de operadores genéticos
4. **Hereditariedade:** Características das soluções pais são transmitidas aos filhos
5. **Evolução:** A população melhora ao longo das gerações

### **Ciclo Evolutivo Geral**

```
🌱 1. INICIALIZAÇÃO
   └── Criar população inicial aleatória

🔄 2. LOOP EVOLUTIVO (até critério de parada):
   ├── 📊 AVALIAÇÃO
   │   └── Calcular fitness de cada indivíduo
   │
   ├── 🎯 SELEÇÃO
   │   └── Escolher indivíduos para reprodução
   │
   ├── 🧬 REPRODUÇÃO
   │   ├── Crossover (recombinação)
   │   └── Mutação
   │
   └── 🔄 SUBSTITUIÇÃO
       └── Formar nova geração

🏆 3. RETORNAR melhor solução encontrada
```

---

## 📚 Algoritmos Disponíveis

### 1. [**Algoritmos Genéticos (Genetic Algorithms - GA)**](genetic_algorithms.md)

Os Algoritmos Genéticos são a forma mais clássica de algoritmos evolucionários, utilizando representação binária ou de valores reais e operadores inspirados na genética.

**Principais Características:**
- 🧬 Codificação em cromossomos (binária, real, permutação)
- 🔀 Crossover de um ou múltiplos pontos
- 🎲 Mutação bit-a-bit ou gaussiana
- 🎯 Seleção por roleta, torneio ou ranking

**Quando Usar:**
- Otimização combinatória (scheduling, routing)
- Busca em espaços discretos
- Problemas com múltiplos objetivos
- Feature selection em ML

**Aplicações:**
- Design de circuitos
- Otimização de portfólio
- Planejamento de rotas
- Calibração de modelos

---

### 2. [**Evolução Diferencial (Differential Evolution - DE)**](differential_evolution.md)

Evolução Diferencial é um método de otimização poderoso e simples para espaços contínuos, usando diferenças entre vetores para gerar mutações.

**Principais Características:**
- 📐 Especializado em otimização contínua
- ➗ Mutação baseada em diferenças vetoriais
- 🎛️ Poucos parâmetros: F (escala) e CR (crossover)
- 🚀 Convergência rápida e robusta

**Quando Usar:**
- Funções multimodais complexas
- Otimização numérica de alta dimensão
- Calibração de parâmetros
- Problemas sem derivadas disponíveis

**Aplicações:**
- Treinamento de redes neurais
- Ajuste de hiperparâmetros
- Engenharia (design, controle)
- Problemas de benchmark

---

### 3. [**Estratégias de Evolução (Evolution Strategies - ES)**](evolution_strategies.md)

Estratégias de Evolução focam na evolução de parâmetros contínuos e na auto-adaptação de parâmetros de mutação.

**Principais Características:**
- 📊 Notação: (μ + λ)-ES ou (μ, λ)-ES
- 🔧 Auto-adaptação de step-sizes
- 📈 Matriz de covariância adaptativa (CMA-ES)
- 🎯 Operadores especializados para contínuos

**Quando Usar:**
- Otimização de funções contínuas complexas
- Quando precisar de auto-adaptação
- Problemas ruidosos
- Otimização de alta dimensão

**Aplicações:**
- Reinforcement Learning (RL)
- Otimização de políticas neurais
- Robótica (controle, locomoção)
- Engenharia (aeronaves, estruturas)

---

### 4. [**Programação Genética (Genetic Programming - GP)**](genetic_programming.md)

Programação Genética evolui programas de computador e expressões matemáticas, representados como estruturas de árvore.

**Principais Características:**
- 🌳 Representação em árvores de expressões
- 🔀 Crossover de subárvores
- 🎲 Mutação de nós e subárvores
- 📝 Evolui código e fórmulas

**Quando Usar:**
- Descoberta de fórmulas e modelos
- Regressão simbólica
- Evolução de estratégias
- Geração automática de programas

**Aplicações:**
- Descoberta científica (equações)
- Trading algorithms
- Classificação e regressão
- Design de circuitos

---

### 5. [**Programação de Expressão Gênica (Gene Expression Programming - GEP)**](gene_expression_programming.md)

GEP combina a simplicidade de representação dos GAs com o poder expressivo da GP, usando cromossomos lineares que codificam estruturas de árvore.

**Principais Características:**
- 🧬 Genótipo linear + fenótipo em árvore
- 🔄 Separação genótipo-fenótipo
- 🎯 Operadores genéticos mais simples que GP
- 📐 Head e tail em cada gene

**Quando Usar:**
- Regressão simbólica
- Classificação baseada em funções
- Modelagem de séries temporais
- Descoberta de conhecimento

**Aplicações:**
- Predição financeira
- Bioinformática
- Modelagem de sistemas complexos
- Data mining

---

## 🔍 Comparação entre Algoritmos Evolucionários

| Aspecto | GA | DE | ES | GP | GEP |
|---------|----|----|----|----|-----|
| **Representação** | Binária/Real | Real | Real | Árvore | Linear→Árvore |
| **Espaço** | Discreto/Contínuo | Contínuo | Contínuo | Simbólico | Simbólico |
| **Crossover** | ✅ Importante | ✅ Essencial | ⚪ Opcional | ✅ Importante | ✅ Importante |
| **Mutação** | ⚪ Secundário | ✅ Principal | ✅ Principal | ⚪ Secundário | ⚪ Secundário |
| **Auto-adaptação** | ❌ Não | ⚪ Limitada | ✅ Sim | ❌ Não | ❌ Não |
| **Complexidade** | Baixa | Baixa | Média | Alta | Média |
| **Aplicação Principal** | Combinatória | Numérica | Numérica | Simbólica | Simbólica |

### **Legendas:**
- ✅ Característica central
- ⚪ Característica secundária
- ❌ Não aplicável ou raro

---

## 🎯 Quando Usar Cada Algoritmo

### **Escolha GA quando:**
- ✅ Problema combinatório (TSP, scheduling)
- ✅ Variáveis discretas ou mistas
- ✅ Múltiplos objetivos
- ✅ Necessita interpretabilidade

### **Escolha DE quando:**
- ✅ Otimização contínua
- ✅ Função multimodal
- ✅ Precisa de convergência rápida
- ✅ Quer simplicidade de implementação

### **Escolha ES quando:**
- ✅ Otimização numérica difícil
- ✅ Função ruidosa
- ✅ Alta dimensionalidade
- ✅ Precisa auto-adaptação (CMA-ES)

### **Escolha GP quando:**
- ✅ Quer descobrir fórmulas/modelos
- ✅ Regressão simbólica
- ✅ Evolução de programas
- ✅ Interpretabilidade é crucial

### **Escolha GEP quando:**
- ✅ Quer benefícios de GP
- ✅ Prefere operadores mais simples
- ✅ Regressão simbólica
- ✅ Modelagem de dados

---

## 🔧 Componentes Comuns

### **1. Representação (Encoding)**

Como codificar soluções do problema:

```python
# Binária (GA)
cromossomo = [1, 0, 1, 1, 0, 1, 0, 0]

# Real (DE, ES)
individuo = [2.5, -1.3, 0.8, 4.2]

# Permutação (GA para TSP)
tour = [0, 3, 1, 4, 2]

# Árvore (GP)
# tree = Add(Mul(X, 2), Div(Y, 3))
```

### **2. Função de Fitness**

Avalia a qualidade da solução:

```python
def fitness(individuo):
    """
    Retorna valor numérico indicando qualidade
    Maior = melhor (maximização)
    Menor = melhor (minimização)
    """
    return avaliar_solucao(individuo)
```

### **3. Seleção**

Métodos para escolher pais:

- **Roleta (Roulette Wheel):** Probabilidade proporcional ao fitness
- **Torneio (Tournament):** Compete k indivíduos, escolhe o melhor
- **Ranking:** Baseado em posição ordenada
- **Elitismo:** Preserva os melhores

### **4. Operadores de Variação**

**Crossover (Recombinação):**
```python
# Um ponto
pai1 = [1, 0, 1, 1, 0, 1, 0, 0]
pai2 = [0, 1, 0, 0, 1, 1, 1, 0]
#             ↓ ponto de corte
filho1 = [1, 0, 1, 0, 1, 1, 1, 0]
filho2 = [0, 1, 0, 1, 0, 1, 0, 0]
```

**Mutação:**
```python
# Flip de bit
antes  = [1, 0, 1, 1, 0, 1, 0, 0]
depois = [1, 0, 0, 1, 0, 1, 0, 0]  # bit 2 mudou

# Gaussiana (valores reais)
antes  = [2.5, -1.3, 0.8]
depois = [2.5, -1.1, 0.8]  # -1.3 + N(0, σ)
```

---

## 📊 Análise de Convergência

### **Métricas Importantes**

```python
# 1. Melhor fitness ao longo das gerações
best_fitness_history = [f1, f2, f3, ..., fn]

# 2. Fitness médio da população
avg_fitness_history = [avg1, avg2, ..., avgn]

# 3. Diversidade da população
diversity = std(population_fitness)

# 4. Taxa de sucesso
success_rate = num_runs_found_optimum / total_runs
```

### **Sinais de Problemas**

```
❌ Convergência prematura:
   - População perde diversidade muito rápido
   - Estagna em ótimo local
   
❌ Convergência lenta:
   - Fitness não melhora após muitas gerações
   - População muito diversa
   
❌ Estagnação:
   - Fitness melhor parou de melhorar
   - Fitness médio não converge para o melhor
```

---

## 🎓 Conceitos Avançados

### **1. Algoritmos Híbridos**

Combinam EAs com outras técnicas:

```python
# EA + Busca Local
def hybrid_ea():
    population = initialize()
    
    for gen in range(max_generations):
        # Parte evolutiva
        offspring = evolve(population)
        
        # Refinamento local
        for individual in offspring:
            individual = local_search(individual)
        
        population = select_survivors(population, offspring)
    
    return best_individual(population)
```

### **2. Multi-objetivo (NSGA-II, SPEA2)**

Otimização com múltiplos objetivos conflitantes:

```python
# Exemplo: Minimizar custo E maximizar qualidade
fitness1 = custo(solucao)          # minimizar
fitness2 = qualidade(solucao)       # maximizar

# Usa dominância de Pareto para seleção
```

### **3. Co-evolução**

Múltiplas populações evoluem simultaneamente:

```python
# Predadores e presas
populacao_predadores = evolve(predadores, presas)
populacao_presas = evolve(presas, predadores)
```

### **4. Paralelização**

```python
# Modelo ilha
# Múltiplas populações evoluem em paralelo
# Migração periódica entre ilhas

ilhas = [Population() for _ in range(n_islands)]

for gen in range(generations):
    # Evolui cada ilha em paralelo
    with ThreadPoolExecutor() as executor:
        ilhas = list(executor.map(evolve_population, ilhas))
    
    # Migração entre ilhas
    if gen % migration_interval == 0:
        migrate(ilhas)
```

---

## 💡 Boas Práticas

### **✅ Faça:**

1. **Ajuste o tamanho da população**
   - Pequena: convergência rápida, risco de ótimo local
   - Grande: mais exploração, custo computacional maior

2. **Balance exploração vs explotação**
   - Início: alta mutação, exploração
   - Final: baixa mutação, refinamento

3. **Use elitismo**
   - Preserve as melhores soluções

4. **Monitore diversidade**
   - Diversidade zero = convergência prematura

5. **Teste múltiplas execuções**
   - EAs são estocásticos, reporte média e desvio

### **❌ Evite:**

1. **População muito pequena**
   - Falta diversidade genética

2. **Taxa de mutação muito alta**
   - Comportamento aleatório, não evolutivo

3. **Ignorar restrições do problema**
   - Use repair operators ou penalidades

4. **Executar por gerações fixas sem critério**
   - Use convergência ou tempo como critério

---

## 🔬 Exemplo Unificado

Estrutura geral de um EA:

```python
import random
import numpy as np

class EvolutionaryAlgorithm:
    """Template geral de algoritmo evolucionário"""
    
    def __init__(self, pop_size, generations, mutation_rate=0.01):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def initialize_population(self):
        """Criar população inicial - IMPLEMENTAR"""
        raise NotImplementedError
    
    def evaluate_fitness(self, individual):
        """Avaliar fitness de um indivíduo - IMPLEMENTAR"""
        raise NotImplementedError
    
    def select_parents(self, population, fitnesses):
        """Seleção de pais - IMPLEMENTAR"""
        raise NotImplementedError
    
    def crossover(self, parent1, parent2):
        """Recombinação - IMPLEMENTAR"""
        raise NotImplementedError
    
    def mutate(self, individual):
        """Mutação - IMPLEMENTAR"""
        raise NotImplementedError
    
    def evolve(self):
        """Algoritmo evolutivo geral"""
        # 1. Inicialização
        population = self.initialize_population()
        best_individual = None
        best_fitness = float('-inf')
        history = []
        
        # 2. Loop evolutivo
        for gen in range(self.generations):
            # Avaliar população
            fitnesses = [self.evaluate_fitness(ind) for ind in population]
            
            # Atualizar melhor
            gen_best_idx = np.argmax(fitnesses)
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = population[gen_best_idx].copy()
            
            # Registrar histórico
            history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'diversity': np.std(fitnesses)
            })
            
            # Criar nova geração
            new_population = []
            
            # Elitismo: preservar melhor
            new_population.append(population[gen_best_idx])
            
            # Gerar resto da população
            while len(new_population) < self.pop_size:
                # Seleção
                parent1, parent2 = self.select_parents(population, fitnesses)
                
                # Crossover
                if random.random() < 0.8:  # Taxa de crossover
                    child = self.crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2]).copy()
                
                # Mutação
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_individual, best_fitness, history

# Exemplo de uso
"""
class MeuProblema(EvolutionaryAlgorithm):
    def initialize_population(self):
        # Implementar inicialização específica
        pass
    
    # Implementar outros métodos...

ea = MeuProblema(pop_size=100, generations=100)
melhor, fitness, historico = ea.evolve()
"""
```

---

## 📚 Recursos Adicionais

### **Livros Recomendados**
- "Introduction to Evolutionary Computing" - Eiben & Smith
- "Genetic Algorithms in Search, Optimization, and Machine Learning" - Goldberg
- "Differential Evolution: A Practical Approach" - Price, Storn & Lampinen

### **Bibliotecas Python**
- **DEAP** (Distributed Evolutionary Algorithms in Python)
- **PyGAD** (Python Genetic Algorithm)
- **pymoo** (Multi-objective Optimization)
- **gplearn** (Genetic Programming)

### **Frameworks**
- **NEAT** (NeuroEvolution of Augmenting Topologies)
- **CMA-ES** (Covariance Matrix Adaptation)
- **OpenAI Evolution Strategies**

---

## 🎯 Próximos Passos

1. **Escolha um algoritmo** específico nos links acima
2. **Leia a documentação completa** com teoria e exemplos
3. **Execute os exemplos** em Python
4. **Adapte para seu problema** específico
5. **Experimente** variantes e otimizações

---

## 🔗 Algoritmos Relacionados

- [**Hill Climbing**](../greedy/hill_climbing.md) - Busca local, ponto de partida
- [**Simulated Annealing**](../metaheuristics/simulated_annealing.md) - Metaheurística com aceitação probabilística
- [**Cross-Entropy Method**](../optimization/cross_entropy_method.md) - Otimização baseada em amostragem

---

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

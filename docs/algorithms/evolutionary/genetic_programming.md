# Programação Genética (Genetic Programming - GP)

A **Programação Genética** (Genetic Programming - GP) é uma técnica de algoritmo evolutivo que evolui programas de computador e expressões matemáticas para resolver problemas. Desenvolvida por John Koza no início dos anos 1990, a GP estende os conceitos dos Algoritmos Genéticos para trabalhar com estruturas de árvore que representam programas, fórmulas e expressões.

![Genetic Programming Concept](../../images/genetic_programming_concept.png)

A GP é particularmente poderosa para **descoberta automática de conhecimento**, regressão simbólica, geração de estratégias e evolução de algoritmos. Diferentemente de outros métodos de otimização que ajustam parâmetros, a GP descobre a própria estrutura da solução.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 Conceito Central**

A Programação Genética evolui **programas como estruturas de árvore**, onde:

1. **Nós Internos:** Funções/Operadores (AND, +, -, *, /, IF, etc.)
2. **Nós Folha (Terminais):** Variáveis e Constantes (X, Y, 3.14, etc.)
3. **Árvore Completa:** Representa um programa executável

**Intuição:**
> "Assim como a natureza evolui organismos complexos através de variações e seleção, a GP evolui programas complexos através de operações genéticas em árvores de expressão."

### **1.2 Representação em Árvore**

#### **Exemplo: Expressão Matemática**

```
Expressão: (X + Y) * (X - 3)

Árvore:
        *
       / \
      +   -
     / \ / \
    X  Y X  3

Notação Prefix (Polish): * + X Y - X 3
Notação Infix (Humana): (X + Y) * (X - 3)
Notação Postfix (RPN): X Y + X 3 - *
```

#### **Exemplo: Programa com Lógica**

```
Programa: if (X > 5) then X*2 else X+1

Árvore:
       IF
      / | \
     >  *  +
    / \/ \/ \
   X 5 X 2 X 1
```

### **1.3 Diferenças dos Algoritmos Genéticos**

| Aspecto | Algoritmos Genéticos | Genetic Programming |
|---------|---------------------|---------------------|
| **Representação** | String/Array de genes | Árvore de expressões |
| **Tamanho** | Fixo | Variável |
| **Busca** | Parâmetros de soluções | Estrutura de soluções |
| **Crossover** | Troca de genes | Troca de subárvores |
| **Mutação** | Alteração de genes | Alteração de nós/subárvores |
| **Resultado** | Configuração ótima | Programa/Fórmula |
| **Interpretabilidade** | Média | Alta (fórmulas legíveis) |

---

## **2. 🔧 Algoritmo da Programação Genética**

### **2.1 Estrutura Geral**

```
🚀 1. INICIALIZAÇÃO
   ├── Definir conjunto de funções: F = {+, -, *, /, sin, cos, ...}
   ├── Definir conjunto de terminais: T = {X, Y, Z, constantes}
   ├── Gerar população inicial de árvores aleatórias
   └── Avaliar fitness de cada árvore

🔄 2. LOOP EVOLUTIVO (enquanto não convergir):
   │
   ├── 📊 AVALIAÇÃO
   │   └── Executar cada programa e calcular fitness
   │
   ├── 🎯 SELEÇÃO
   │   └── Selecionar pais (torneio, roleta, etc.)
   │
   ├── 🧬 REPRODUÇÃO
   │   ├── 🔀 CROSSOVER (70-90% probabilidade)
   │   │   ├── Selecionar dois pais
   │   │   ├── Escolher ponto de corte em cada árvore
   │   │   └── Trocar subárvores
   │   │
   │   ├── 🎲 MUTAÇÃO (10-30% probabilidade)
   │   │   ├── Substituir nó aleatório
   │   │   ├── Substituir subárvore
   │   │   └── Alterar constante
   │   │
   │   └── 📋 REPRODUÇÃO (cópia direta)
   │
   ├── 🔄 SUBSTITUIÇÃO
   │   └── Formar nova geração (geracional ou steady-state)
   │
   └── 🛡️ BLOAT CONTROL (opcional)
       └── Limitar tamanho/profundidade das árvores

🏆 3. RETORNAR melhor programa encontrado
```

### **2.2 Operadores Genéticos**

#### **🔀 Crossover (Recombinação de Subárvores)**

```
Pai 1:      *               Pai 2:      +
           / \                         / \
          +   Y                       X   /
         / \                             / \
        X   3                           Y   2

Ponto de corte em Pai 1: subárvore "+"
Ponto de corte em Pai 2: subárvore "/"

Filho 1:    *               Filho 2:    +
           / \                         / \
          /   Y                       X   +
         / \                             / \
        Y   2                           X   3
```

**Características:**
- ✅ Combina características de ambos os pais
- ✅ Cria diversidade estrutural
- ⚠️ Pode gerar árvores muito grandes (bloat)
- 🎯 Taxa típica: 70-90%

#### **🎲 Mutação**

**1. Mutação de Ponto (Point Mutation)**
```
Antes:   +              Depois:   *
        / \                      / \
       X   Y                    X   Y

Muda operador + para *
```

**2. Mutação de Subárvore (Subtree Mutation)**
```
Antes:   *              Depois:   *
        / \                      / \
       +   Y                    X   Y
      / \
     X   3

Substitui subárvore + por terminal X
```

**3. Mutação de Constante (Ephemeral Random Constants)**
```
Antes:   +              Depois:   +
        / \                      / \
       X   3                    X   5.7

Muda constante 3 para 5.7
```

**Características:**
- ✅ Introduz novidade na população
- ✅ Pode simplificar árvores
- ⚠️ Pode destruir boas soluções
- 🎯 Taxa típica: 10-30%

### **2.3 Métodos de Inicialização**

#### **Método Full (Cheio)**
```python
def generate_full(max_depth, current_depth=0):
    """
    Árvore completa até profundidade máxima
    Nós internos até max_depth, folhas apenas no max_depth
    """
    if current_depth >= max_depth:
        return random.choice(TERMINALS)
    else:
        func = random.choice(FUNCTIONS)
        children = [generate_full(max_depth, current_depth + 1) 
                   for _ in range(func.arity)]
        return Node(func, children)
```

#### **Método Grow (Crescimento)**
```python
def generate_grow(max_depth, current_depth=0):
    """
    Árvore irregular - pode escolher terminal em qualquer nível
    Permite árvores de tamanhos diferentes
    """
    if current_depth >= max_depth:
        return random.choice(TERMINALS)
    else:
        # Pode escolher função ou terminal
        if random.random() < 0.5:
            return random.choice(TERMINALS)
        else:
            func = random.choice(FUNCTIONS)
            children = [generate_grow(max_depth, current_depth + 1) 
                       for _ in range(func.arity)]
            return Node(func, children)
```

#### **Ramped Half-and-Half**
```python
def ramped_half_and_half(pop_size, max_depth):
    """
    Combina Full e Grow para diversidade
    Metade da população com cada método
    Diferentes profundidades de 2 até max_depth
    """
    population = []
    depths = range(2, max_depth + 1)
    
    for i in range(pop_size):
        depth = depths[i % len(depths)]
        if i % 2 == 0:
            tree = generate_full(depth)
        else:
            tree = generate_grow(depth)
        population.append(tree)
    
    return population
```

---

## **3. 💻 Implementação em Python**

### **3.1 Estrutura Básica de Nó e Árvore**

```python
import numpy as np
import random
from typing import List, Callable, Any
import operator

class Node:
    """Representa um nó na árvore de expressão"""
    
    def __init__(self, value, children=None):
        """
        Args:
            value: Função ou terminal
            children: Lista de nós filhos (None para terminal)
        """
        self.value = value
        self.children = children or []
    
    def is_terminal(self):
        """Verifica se é nó folha"""
        return len(self.children) == 0
    
    def eval(self, context):
        """
        Avalia a árvore recursivamente
        
        Args:
            context: Dicionário com valores das variáveis
        """
        if self.is_terminal():
            # Terminal: retornar valor ou buscar variável
            if isinstance(self.value, (int, float)):
                return self.value
            else:
                return context.get(self.value, 0)
        else:
            # Função: avaliar filhos e aplicar função
            child_values = [child.eval(context) for child in self.children]
            return self.value(*child_values)
    
    def copy(self):
        """Cria cópia profunda da árvore"""
        if self.is_terminal():
            return Node(self.value)
        else:
            children_copy = [child.copy() for child in self.children]
            return Node(self.value, children_copy)
    
    def size(self):
        """Retorna número total de nós"""
        if self.is_terminal():
            return 1
        return 1 + sum(child.size() for child in self.children)
    
    def depth(self):
        """Retorna profundidade máxima da árvore"""
        if self.is_terminal():
            return 0
        return 1 + max(child.depth() for child in self.children)
    
    def to_string(self):
        """Converte árvore para string legível"""
        if self.is_terminal():
            return str(self.value)
        
        if len(self.children) == 2:
            # Operador binário
            op_name = self.value.__name__ if hasattr(self.value, '__name__') else str(self.value)
            return f"({self.children[0].to_string()} {op_name} {self.children[1].to_string()})"
        elif len(self.children) == 1:
            # Operador unário
            op_name = self.value.__name__ if hasattr(self.value, '__name__') else str(self.value)
            return f"{op_name}({self.children[0].to_string()})"
        else:
            # Função genérica
            op_name = self.value.__name__ if hasattr(self.value, '__name__') else str(self.value)
            args = ', '.join(child.to_string() for child in self.children)
            return f"{op_name}({args})"

# Definir conjunto de funções
def safe_div(a, b):
    """Divisão protegida contra divisão por zero"""
    return a / b if abs(b) > 1e-10 else 1.0

def safe_log(x):
    """Logaritmo protegido"""
    return np.log(abs(x)) if abs(x) > 1e-10 else 0.0

# Funções disponíveis com suas aridades
FUNCTION_SET = {
    operator.add: 2,
    operator.sub: 2,
    operator.mul: 2,
    safe_div: 2,
    np.sin: 1,
    np.cos: 1,
    np.exp: 1,
    safe_log: 1,
    operator.neg: 1
}

# Terminais disponíveis
TERMINAL_SET = ['X', 'Y', 'Z']  # Variáveis

def generate_random_constant():
    """Gera constante aleatória"""
    return random.uniform(-5, 5)
```

### **3.2 Implementação Completa da GP**

```python
class GeneticProgramming:
    """
    Implementação de Programação Genética
    """
    
    def __init__(self, 
                 function_set=None,
                 terminal_set=None,
                 pop_size=100,
                 max_depth_init=6,
                 max_depth=17,
                 crossover_rate=0.9,
                 mutation_rate=0.1,
                 tournament_size=7,
                 generations=50):
        """
        Args:
            function_set: Dicionário {função: aridade}
            terminal_set: Lista de terminais (variáveis)
            pop_size: Tamanho da população
            max_depth_init: Profundidade máxima na inicialização
            max_depth: Profundidade máxima permitida
            crossover_rate: Taxa de crossover
            mutation_rate: Taxa de mutação
            tournament_size: Tamanho do torneio
            generations: Número de gerações
        """
        self.function_set = function_set or FUNCTION_SET
        self.terminal_set = terminal_set or TERMINAL_SET
        self.pop_size = pop_size
        self.max_depth_init = max_depth_init
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.generations = generations
    
    def random_terminal(self):
        """Seleciona terminal aleatório"""
        if random.random() < 0.5:
            return random.choice(self.terminal_set)
        else:
            return generate_random_constant()
    
    def random_function(self):
        """Seleciona função aleatória"""
        return random.choice(list(self.function_set.keys()))
    
    def generate_tree(self, method='grow', max_depth=None, current_depth=0):
        """
        Gera árvore aleatória
        
        Args:
            method: 'grow' ou 'full'
            max_depth: Profundidade máxima
            current_depth: Profundidade atual
        """
        if max_depth is None:
            max_depth = self.max_depth_init
        
        if current_depth >= max_depth:
            # Profundidade máxima: criar terminal
            return Node(self.random_terminal())
        
        if method == 'full':
            # Sempre criar função até max_depth
            func = self.random_function()
            arity = self.function_set[func]
            children = [self.generate_tree('full', max_depth, current_depth + 1)
                       for _ in range(arity)]
            return Node(func, children)
        
        else:  # method == 'grow'
            # Pode criar função ou terminal
            if random.random() < 0.5:
                return Node(self.random_terminal())
            else:
                func = self.random_function()
                arity = self.function_set[func]
                children = [self.generate_tree('grow', max_depth, current_depth + 1)
                           for _ in range(arity)]
                return Node(func, children)
    
    def initialize_population(self):
        """Inicializa população usando Ramped Half-and-Half"""
        population = []
        depths = range(2, self.max_depth_init + 1)
        
        for i in range(self.pop_size):
            depth = depths[i % len(depths)]
            method = 'full' if i % 2 == 0 else 'grow'
            tree = self.generate_tree(method, depth)
            population.append(tree)
        
        return population
    
    def tournament_selection(self, population, fitnesses):
        """Seleção por torneio"""
        selected = random.sample(range(len(population)), self.tournament_size)
        best_idx = min(selected, key=lambda i: fitnesses[i])
        return population[best_idx].copy()
    
    def subtree_crossover(self, parent1, parent2):
        """
        Crossover de subárvore
        Retorna dois filhos
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Selecionar pontos de corte aleatórios
        nodes1 = self._get_all_nodes(child1)
        nodes2 = self._get_all_nodes(child2)
        
        if not nodes1 or not nodes2:
            return child1, child2
        
        # Escolher nós aleatórios
        cut1 = random.choice(nodes1)
        cut2 = random.choice(nodes2)
        
        # Trocar subárvores
        # (Implementação simplificada - na prática, precisa rastrear pais)
        # Para simplicidade, retornar cópias
        
        return child1, child2
    
    def _get_all_nodes(self, tree):
        """Retorna lista de todos os nós da árvore"""
        nodes = [tree]
        if not tree.is_terminal():
            for child in tree.children:
                nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def point_mutation(self, tree):
        """
        Mutação de ponto
        Substitui um nó aleatório mantendo aridade
        """
        mutant = tree.copy()
        nodes = self._get_all_nodes(mutant)
        
        if not nodes:
            return mutant
        
        # Escolher nó aleatório
        node = random.choice(nodes)
        
        if node.is_terminal():
            # Substituir terminal
            node.value = self.random_terminal()
        else:
            # Substituir função mantendo aridade
            current_arity = len(node.children)
            # Buscar função com mesma aridade
            compatible_funcs = [f for f, a in self.function_set.items() 
                              if a == current_arity]
            if compatible_funcs:
                node.value = random.choice(compatible_funcs)
        
        return mutant
    
    def subtree_mutation(self, tree):
        """
        Mutação de subárvore
        Substitui subárvore aleatória por nova subárvore
        """
        mutant = tree.copy()
        nodes = self._get_all_nodes(mutant)
        
        if not nodes:
            return mutant
        
        # Escolher nó aleatório e substituir por nova subárvore
        node = random.choice(nodes)
        new_subtree = self.generate_tree('grow', max_depth=3)
        
        # Substituir (simplificado)
        node.value = new_subtree.value
        node.children = new_subtree.children
        
        return mutant
    
    def evaluate_fitness(self, tree, X, y):
        """
        Avalia fitness da árvore
        
        Args:
            tree: Árvore de expressão
            X: Dados de entrada (matriz)
            y: Valores alvo (vetor)
        
        Returns:
            fitness: Erro (menor é melhor)
        """
        try:
            predictions = []
            for sample in X:
                # Criar contexto com variáveis
                context = {self.terminal_set[i]: sample[i] 
                          for i in range(min(len(self.terminal_set), len(sample)))}
                
                # Avaliar árvore
                pred = tree.eval(context)
                
                # Tratar valores inválidos
                if np.isnan(pred) or np.isinf(pred):
                    pred = 0
                
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calcular erro (MSE)
            error = np.mean((predictions - y) ** 2)
            
            # Penalizar árvores muito grandes (bloat control)
            size_penalty = 0.001 * tree.size()
            
            return error + size_penalty
        
        except Exception as e:
            # Em caso de erro, retornar fitness muito ruim
            return 1e10
    
    def evolve(self, X_train, y_train):
        """
        Executa evolução
        
        Args:
            X_train: Dados de treinamento (matriz)
            y_train: Valores alvo (vetor)
        
        Returns:
            best_tree: Melhor árvore encontrada
            best_fitness: Fitness da melhor árvore
            history: Histórico de evolução
        """
        # Inicializar população
        population = self.initialize_population()
        
        # Avaliar população inicial
        fitnesses = [self.evaluate_fitness(tree, X_train, y_train) 
                    for tree in population]
        
        # Melhor solução
        best_idx = np.argmin(fitnesses)
        best_tree = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        
        history = {
            'best_fitness': [best_fitness],
            'avg_fitness': [np.mean(fitnesses)],
            'avg_size': [np.mean([tree.size() for tree in population])],
            'avg_depth': [np.mean([tree.depth() for tree in population])]
        }
        
        # Loop evolutivo
        for generation in range(self.generations):
            new_population = []
            
            # Elitismo: preservar melhor
            new_population.append(best_tree.copy())
            
            # Gerar resto da população
            while len(new_population) < self.pop_size:
                # Seleção
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Aplicar operadores genéticos
                if random.random() < self.crossover_rate:
                    # Crossover
                    child1, child2 = self.subtree_crossover(parent1, parent2)
                else:
                    # Reprodução
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                
                # Mutação
                if random.random() < self.mutation_rate:
                    child1 = self.point_mutation(child1)
                
                if random.random() < self.mutation_rate:
                    child2 = self.point_mutation(child2)
                
                # Limitar profundidade
                if child1.depth() <= self.max_depth:
                    new_population.append(child1)
                if child2.depth() <= self.max_depth and len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            # Atualizar população
            population = new_population[:self.pop_size]
            
            # Avaliar nova população
            fitnesses = [self.evaluate_fitness(tree, X_train, y_train) 
                        for tree in population]
            
            # Atualizar melhor
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_tree = population[gen_best_idx].copy()
                best_fitness = fitnesses[gen_best_idx]
            
            # Registrar histórico
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(np.mean(fitnesses))
            history['avg_size'].append(np.mean([tree.size() for tree in population]))
            history['avg_depth'].append(np.mean([tree.depth() for tree in population]))
            
            # Imprimir progresso
            if generation % 10 == 0:
                print(f"Gen {generation}: Best fitness = {best_fitness:.6f}, "
                      f"Avg size = {history['avg_size'][-1]:.1f}")
        
        return best_tree, best_fitness, history

# Exemplo de uso
def example_symbolic_regression():
    """
    Regressão simbólica: descobrir fórmula a partir de dados
    """
    # Gerar dados da função alvo: y = x^2 + x + 1
    X_train = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_train = X_train[:, 0]**2 + X_train[:, 0] + 1
    
    # Configurar GP
    gp = GeneticProgramming(
        terminal_set=['X'],
        pop_size=200,
        max_depth_init=6,
        max_depth=17,
        crossover_rate=0.9,
        mutation_rate=0.1,
        tournament_size=7,
        generations=50
    )
    
    # Evoluir
    best_tree, best_fitness, history = gp.evolve(X_train, y_train)
    
    print(f"\nMelhor fórmula encontrada:")
    print(best_tree.to_string())
    print(f"Fitness (MSE): {best_fitness:.6f}")
    
    return best_tree, history

# Executar exemplo
if __name__ == "__main__":
    best_formula, history = example_symbolic_regression()
```

---

## **4. 🎯 Aplicações da Programação Genética**

### **4.1 Regressão Simbólica**

```python
def symbolic_regression_example():
    """
    Descobre fórmula matemática a partir de dados
    """
    # Dados: relação física (e.g., lei de Kepler simplificada)
    # T² ∝ R³  =>  T = k * R^(3/2)
    
    R = np.linspace(1, 10, 50)
    T = 2.5 * R**(1.5) + np.random.normal(0, 0.5, 50)
    
    X_train = R.reshape(-1, 1)
    y_train = T
    
    # Configurar GP com funções apropriadas
    function_set = {
        operator.add: 2,
        operator.sub: 2,
        operator.mul: 2,
        safe_div: 2,
        np.sqrt: 1,
        lambda x: x**2: 1,
        lambda x: x**3: 1
    }
    
    gp = GeneticProgramming(
        function_set=function_set,
        terminal_set=['R'],
        pop_size=500,
        generations=100
    )
    
    best_tree, fitness, history = gp.evolve(X_train, y_train)
    
    print("Fórmula descoberta:", best_tree.to_string())
    print(f"Erro: {fitness:.4f}")
    
    # Visualizar
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(R, T, label='Dados Observados', alpha=0.6)
    
    # Predições da GP
    predictions = []
    for r in R:
        pred = best_tree.eval({'R': r})
        predictions.append(pred)
    
    plt.plot(R, predictions, 'r-', label='Fórmula GP', linewidth=2)
    plt.xlabel('R')
    plt.ylabel('T')
    plt.title('Regressão Simbólica com GP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### **4.2 Classificação com GP**

```python
def gp_classification():
    """
    Usa GP para criar regra de classificação
    """
    from sklearn.datasets import make_classification
    
    # Gerar dados de classificação
    X, y = make_classification(n_samples=200, n_features=2, 
                               n_redundant=0, n_informative=2,
                               random_state=42)
    
    # Modificar fitness para classificação
    def classification_fitness(tree, X, y):
        try:
            predictions = []
            for sample in X:
                context = {'X': sample[0], 'Y': sample[1]}
                pred = tree.eval(context)
                # Classificação: positivo ou negativo
                pred_class = 1 if pred > 0 else 0
                predictions.append(pred_class)
            
            # Acurácia (1 - accuracy como fitness a minimizar)
            accuracy = np.mean(np.array(predictions) == y)
            return 1 - accuracy
        except:
            return 1.0
    
    # GP com funções lógicas
    function_set = {
        operator.add: 2,
        operator.sub: 2,
        operator.mul: 2,
        operator.gt: 2,  # Greater than
        operator.lt: 2,  # Less than
    }
    
    gp = GeneticProgramming(
        function_set=function_set,
        terminal_set=['X', 'Y'],
        pop_size=300,
        generations=50
    )
    
    # Sobrescrever método de fitness
    gp.evaluate_fitness = lambda tree, X, y: classification_fitness(tree, X, y)
    
    best_tree, fitness, history = gp.evolve(X, y)
    
    print("Regra de classificação:", best_tree.to_string())
    print(f"Acurácia: {(1 - fitness) * 100:.2f}%")
```

### **4.3 Geração de Trading Strategies**

```python
def trading_strategy_gp():
    """
    Evolui estratégia de trading
    """
    # Dados de mercado simulados
    days = 252
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, days)))
    
    # Features: preço, média móvel, momentum
    sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
    momentum = np.diff(prices, prepend=prices[0])
    
    X = np.column_stack([prices, sma_20, momentum])
    
    # Label: 1 se preço sobe amanhã, 0 caso contrário
    y = (np.diff(prices, append=prices[-1]) > 0).astype(int)
    
    # Funções para trading
    function_set = {
        operator.add: 2,
        operator.sub: 2,
        operator.mul: 2,
        safe_div: 2,
        operator.gt: 2,
        operator.and_: 2,
        operator.or_: 2,
    }
    
    gp = GeneticProgramming(
        function_set=function_set,
        terminal_set=['Price', 'SMA', 'Momentum'],
        pop_size=500,
        generations=100
    )
    
    # Fitness: retorno acumulado da estratégia
    def trading_fitness(tree, X, y):
        try:
            signals = []
            for i, sample in enumerate(X):
                context = {
                    'Price': sample[0],
                    'SMA': sample[1],
                    'Momentum': sample[2]
                }
                signal = tree.eval(context)
                # Comprar se signal > 0, vender caso contrário
                signals.append(1 if signal > 0 else -1)
            
            # Calcular retorno
            returns = np.diff(prices) / prices[:-1]
            strategy_returns = np.array(signals[:-1]) * returns
            
            # Retorno acumulado (negativo para minimização)
            total_return = np.sum(strategy_returns)
            
            # Penalizar risco (volatilidade)
            volatility = np.std(strategy_returns)
            
            # Sharpe ratio simplificado (negativo)
            sharpe = -total_return / (volatility + 1e-6)
            
            return sharpe
        except:
            return 1e6
    
    gp.evaluate_fitness = lambda tree, X, y: trading_fitness(tree, X, y)
    
    best_strategy, fitness, history = gp.evolve(X, y)
    
    print("Melhor estratégia:", best_strategy.to_string())
    print(f"Sharpe Ratio: {-fitness:.4f}")
```

### **4.4 Síntese de Circuitos**

```python
def circuit_synthesis():
    """
    Evolui circuito lógico para implementar função booleana
    """
    # Função alvo: XOR de 2 bits
    truth_table = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    
    X = np.array([row[0] for row in truth_table])
    y = np.array([row[1] for row in truth_table])
    
    # Portas lógicas
    function_set = {
        operator.and_: 2,
        operator.or_: 2,
        operator.xor: 2,
        operator.not_: 1,
    }
    
    gp = GeneticProgramming(
        function_set=function_set,
        terminal_set=['A', 'B'],
        pop_size=200,
        max_depth_init=4,
        max_depth=6,
        generations=30
    )
    
    # Fitness: número de saídas incorretas
    def circuit_fitness(tree, X, y):
        try:
            errors = 0
            for i, sample in enumerate(X):
                context = {'A': bool(sample[0]), 'B': bool(sample[1])}
                output = tree.eval(context)
                if int(bool(output)) != y[i]:
                    errors += 1
            return errors
        except:
            return len(y)
    
    gp.evaluate_fitness = lambda tree, X, y: circuit_fitness(tree, X, y)
    
    best_circuit, fitness, history = gp.evolve(X, y)
    
    print("Circuito evoluído:", best_circuit.to_string())
    print(f"Erros: {fitness}")
```

---

## **5. ⚙️ Controle de Bloat e Otimizações**

### **5.1 Problema do Bloat**

O **bloat** é o crescimento excessivo do tamanho das árvores sem melhoria de fitness:

```
Problema:
- Árvores crescem exponencialmente ao longo das gerações
- Código redundante (e.g., X + 0, X * 1)
- Overhead computacional
- Overfitting

Exemplo de Bloat:
Função simples: X + Y
Após bloat: ((X * 1) + (0 + Y)) + ((X - X) + 0)
```

### **5.2 Técnicas de Controle de Bloat**

#### **1. Parsimony Pressure (Pressão de Parcimônia)**

```python
def fitness_with_parsimony(tree, X, y, parsimony_coef=0.001):
    """
    Adiciona penalidade proporcional ao tamanho
    """
    error = compute_error(tree, X, y)
    size_penalty = parsimony_coef * tree.size()
    return error + size_penalty
```

#### **2. Limites de Profundidade/Tamanho**

```python
class GPWithLimits(GeneticProgramming):
    def __init__(self, *args, max_size=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size
    
    def is_valid_tree(self, tree):
        """Verifica se árvore está dentro dos limites"""
        return tree.depth() <= self.max_depth and tree.size() <= self.max_size
    
    def crossover(self, parent1, parent2):
        """Crossover que respeita limites"""
        child1, child2 = super().crossover(parent1, parent2)
        
        # Rejeitar se muito grande
        if not self.is_valid_tree(child1):
            child1 = parent1.copy()
        if not self.is_valid_tree(child2):
            child2 = parent2.copy()
        
        return child1, child2
```

#### **3. Simplificação Algébrica**

```python
def simplify_tree(tree):
    """
    Simplifica expressões redundantes
    """
    if tree.is_terminal():
        return tree
    
    # Simplificar filhos recursivamente
    tree.children = [simplify_tree(child) for child in tree.children]
    
    # Regras de simplificação
    if tree.value == operator.add:
        # X + 0 = X
        if isinstance(tree.children[1].value, (int, float)) and tree.children[1].value == 0:
            return tree.children[0]
        # 0 + X = X
        if isinstance(tree.children[0].value, (int, float)) and tree.children[0].value == 0:
            return tree.children[1]
    
    elif tree.value == operator.mul:
        # X * 1 = X
        if isinstance(tree.children[1].value, (int, float)) and tree.children[1].value == 1:
            return tree.children[0]
        # X * 0 = 0
        if isinstance(tree.children[1].value, (int, float)) and tree.children[1].value == 0:
            return Node(0)
    
    # Adicionar mais regras...
    
    return tree
```

#### **4. Lexicographic Parsimony Pressure**

```python
def lexicographic_comparison(ind1, ind2):
    """
    Compara indivíduos lexicograficamente:
    1. Primeiro por fitness
    2. Se fitness igual, por tamanho
    """
    if abs(ind1.fitness - ind2.fitness) < 1e-6:
        # Fitness igual: preferir menor
        return ind1 if ind1.size() < ind2.size() else ind2
    else:
        # Fitness diferente: preferir melhor fitness
        return ind1 if ind1.fitness < ind2.fitness() else ind2
```

---

## **6. ✅ Vantagens e ❌ Desvantagens**

### **6.1 ✅ Vantagens**

| Vantagem | Descrição | Impacto |
|----------|-----------|---------|
| **Descoberta de Estrutura** | Encontra forma da solução, não apenas parâmetros | Inovação genuína |
| **Interpretabilidade** | Resulta em fórmulas/programas legíveis | Entendimento humano |
| **Versatilidade** | Aplica-se a diversos domínios | Ampla aplicabilidade |
| **Sem Modelo A Priori** | Não precisa definir forma da solução | Flexibilidade máxima |
| **Regressão Simbólica** | Descobre leis e relações | Ciência e engenharia |
| **Criatividade** | Pode encontrar soluções não-óbvias | Inovação |
| **Otimização Multi-objetivo** | Pode balancear precisão vs simplicidade | Soluções práticas |

### **6.2 ❌ Desvantagens**

| Desvantagem | Descrição | Mitigação |
|-------------|-----------|-----------|
| **Bloat** | Árvores crescem excessivamente | Parsimony pressure, limites |
| **Custo Computacional** | Muitas avaliações necessárias | Paralelização, simplificação |
| **Convergência Lenta** | Pode levar muitas gerações | Populações grandes, elitismo |
| **Overfitting** | Árvores complexas memorizam dados | Validação cruzada, regularização |
| **Interpretabilidade Relativa** | Árvores muito grandes são ilegíveis | Simplificação, limites de tamanho |
| **Difícil Ajustar** | Muitos hiperparâmetros | Usar valores padrão, auto-tune |
| **Não Garante Ótimo** | Pode convergir para subótimo | Múltiplas execuções |

### **6.3 🎯 Quando Usar GP**

#### **✅ Cenários Ideais:**
- ✅ Regressão simbólica (descobrir fórmulas)
- ✅ Classificação com regras interpretáveis
- ✅ Geração de estratégias (trading, controle)
- ✅ Síntese de circuitos/programas
- ✅ Descoberta de conhecimento
- ✅ Feature engineering automático
- ✅ Quando interpretabilidade é crucial
- ✅ Problemas onde estrutura da solução é desconhecida

#### **❌ Evite GP quando:**
- ❌ Problemas de otimização numérica simples (usar DE/ES)
- ❌ Dados são muito ruidosos
- ❌ Avaliação é extremamente cara
- ❌ Interpretabilidade não importa (usar redes neurais)
- ❌ Precisa de convergência garantida
- ❌ Dados são limitados (risco de overfitting)

---

## **7. 🔬 Variantes Avançadas**

### **7.1 Gramática GP (Grammatical Evolution)**

```python
class GrammaticalEvolution:
    """
    Usa gramática BNF para gerar programas
    Genótipo: array de inteiros
    Fenótipo: programa/expressão
    """
    
    def __init__(self, grammar):
        """
        Args:
            grammar: Dicionário definindo gramática BNF
        """
        self.grammar = grammar
    
    # Exemplo de gramática BNF
    example_grammar = {
        '<expr>': [
            '<expr> + <expr>',
            '<expr> - <expr>',
            '<expr> * <expr>',
            '<var>',
            '<const>'
        ],
        '<var>': ['X', 'Y'],
        '<const>': ['1', '2', '3']
    }
    
    def map_genotype_to_phenotype(self, genotype):
        """
        Mapeia array de inteiros para programa usando gramática
        """
        # Implementação do mapeamento...
        pass
```

### **7.2 Cartesian Genetic Programming (CGP)**

```python
class CartesianGP:
    """
    Representa programas como grade de nós
    Vantagens:
    - Representação compacta
    - Mutação eficiente
    - Código neutro (genótipo != fenótipo)
    """
    
    def __init__(self, n_rows, n_cols, n_inputs, n_outputs):
        """
        Grade de nós interconectados
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        # Genótipo: array de (função, conexões)
        self.grid = self.initialize_grid()
```

### **7.3 Multi-objective GP (MOPG)**

```python
def multi_objective_gp():
    """
    GP com múltiplos objetivos:
    - Minimizar erro
    - Minimizar tamanho
    - Maximizar simplicidade
    """
    from deap import tools
    
    # Definir objetivos múltiplos
    def evaluate_multi(tree, X, y):
        error = compute_error(tree, X, y)
        size = tree.size()
        depth = tree.depth()
        
        return error, size, depth  # Minimizar todos
    
    # Usar NSGA-II ou similar para Pareto front
```

---

## **8. 📚 Bibliotecas e Ferramentas**

### **8.1 Bibliotecas Python**

```python
# 1. DEAP - Distributed Evolutionary Algorithms in Python
pip install deap

from deap import algorithms, base, creator, tools, gp

# Exemplo DEAP
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addTerminal(1)
pset.renameArguments(ARG0='x')

# 2. gplearn - Genetic Programming especializado em sklearn
pip install gplearn

from gplearn.genetic import SymbolicRegressor

est = SymbolicRegressor(
    population_size=5000,
    generations=20,
    tournament_size=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01
)

est.fit(X_train, y_train)
print(est._program)

# 3. PyGP - Simple GP framework
# 4. Karoo GP - Genetic Programming suite
# 5. TensorGP - GP with TensorFlow backend
```

### **8.2 Ferramentas Visuais**

```python
# Visualização de árvores GP
def visualize_tree(tree):
    """Visualiza árvore usando graphviz"""
    import graphviz
    
    dot = graphviz.Digraph()
    
    def add_nodes(node, parent_id=None, node_id=0):
        current_id = node_id
        label = str(node.value)
        if hasattr(node.value, '__name__'):
            label = node.value.__name__
        
        dot.node(str(current_id), label)
        
        if parent_id is not None:
            dot.edge(str(parent_id), str(current_id))
        
        node_id += 1
        for child in node.children:
            node_id = add_nodes(child, current_id, node_id)
        
        return node_id
    
    add_nodes(tree)
    return dot

# Uso
tree_viz = visualize_tree(best_tree)
tree_viz.render('gp_tree', format='png', view=True)
```

---

## **9. 🔗 Referências e Recursos**

### **9.1 📚 Livros Fundamentais**

1. **Koza, J. R. (1992).** *"Genetic Programming: On the Programming of Computers by Means of Natural Selection"*. MIT Press.
   - 🌟 Obra seminal que definiu a GP
   - 📖 Teoria completa e aplicações

2. **Poli, R., Langdon, W. B., & McPhee, N. F. (2008).** *"A Field Guide to Genetic Programming"*. Lulu.com (disponível gratuitamente).
   - 📊 Guia prático completo
   - 🎯 Exemplos e melhores práticas

3. **Banzhaf, W., et al. (1998).** *"Genetic Programming: An Introduction"*. Morgan Kaufmann.
   - 📖 Introdução abrangente
   - 🔬 Fundamentos teóricos

### **9.2 🌐 Recursos Online**

| Recurso | Descrição | Link |
|---------|-----------|------|
| **Field Guide to GP** | Livro gratuito online | gpbib.cs.ucl.ac.uk/gp-html |
| **GP Bibliography** | Base de dados de papers | gpbib.cs.ucl.ac.uk |
| **DEAP Documentation** | Documentação oficial | deap.readthedocs.io |
| **gplearn Tutorial** | Tutorial completo | gplearn.readthedocs.io |

### **9.3 📝 Artigos Importantes**

1. **Koza (1994)** - "Genetic Programming as a Means for Programming Computers by Natural Selection"
2. **Langdon & Poli (2002)** - "Foundations of Genetic Programming"
3. **Luke & Panait (2006)** - "A Comparison of Bloat Control Methods"
4. **Schmidt & Lipson (2009)** - "Distilling Free-Form Natural Laws from Experimental Data"

### **9.4 🎓 Conferências e Comunidades**

- **GECCO:** Genetic and Evolutionary Computation Conference
- **EuroGP:** European Conference on Genetic Programming
- **GP-list:** Mailing list da comunidade GP
- **GitHub:** Váriasprojetos open-source de GP

---

## **10. 🎯 Conclusão**

A Programação Genética representa uma das abordagens mais **criativas e poderosas** em inteligência artificial, capaz de descobrir automaticamente soluções que vão além da otimização de parâmetros.

### **🔑 Principais Aprendizados**

1. **Descoberta de Estrutura:** GP não apenas otimiza, mas descobre a forma da solução
2. **Interpretabilidade:** Resulta em fórmulas e programas compreensíveis
3. **Versatilidade:** Aplicável desde regressão até síntese de programas
4. **Desafio do Bloat:** Crescimento das árvores requer controle cuidadoso
5. **Trade-off Precisão-Simplicidade:** Balancear fitness e complexidade é crucial

### **💡 GP vs Outros Métodos**

| Método | Estrutura | Interpretabilidade | Flexibilidade | Custo |
|--------|-----------|-------------------|---------------|-------|
| **GP** | ✅✅ Descobre | ✅✅ Alta | ✅✅ Máxima | ❌ Alto |
| **GA** | ❌ Fixa | ⚪ Média | ✅ Alta | ⚪ Médio |
| **Redes Neurais** | ❌ Fixa | ❌ Baixa | ✅ Alta | ⚪ Médio |
| **Regressão** | ❌ Fixa | ✅ Alta | ❌ Baixa | ✅ Baixo |

### **🚀 Próximos Passos**

1. **Implemente:** Comece com exemplo simples de regressão simbólica
2. **Use Bibliotecas:** Experimente gplearn ou DEAP
3. **Controle Bloat:** Implemente parsimony pressure
4. **Visualize:** Veja árvores evoluídas para entender GP
5. **Aplique:** Use em problemas reais do seu domínio
6. **Explore:** Teste variantes como CGP ou Gramática GP
7. **Compare:** Benchmark contra outros métodos

### **🌟 Reflexão Final**

A Programação Genética demonstra que **computadores podem criar programas** - uma forma de meta-programação que abre portas para automação genuína de descoberta de conhecimento. Enquanto outros métodos ajustam parâmetros de modelos pré-definidos, a GP descobre os próprios modelos, representando um salto qualitativo em inteligência artificial.

> *"A verdadeira magia da Programação Genética não está em encontrar soluções ótimas, mas em descobrir soluções que nunca imaginaríamos - programas que a evolução criou, não o programador."*

**Destaque: Regressão Simbólica** é uma das aplicações mais impactantes da GP, permitindo descobrir leis científicas a partir de dados - o sonho de qualquer cientista.

---

**🔗 Continue explorando:**
- 📖 Veja [**Gene Expression Programming**](gene_expression_programming.md) para evolução híbrida
- 🧬 Compare com [**Genetic Algorithms**](genetic_algorithms.md) para entender diferenças
- 🎯 Explore [**Algoritmos Evolucionários**](README.md) para visão geral
- 📊 Estude [**Differential Evolution**](differential_evolution.md) para otimização numérica

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

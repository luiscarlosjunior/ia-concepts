# Programação de Expressão Gênica (Gene Expression Programming - GEP)

A **Programação de Expressão Gênica** (Gene Expression Programming - GEP) é um algoritmo evolutivo desenvolvido por Cândida Ferreira em 1999 que combina a simplicidade de representação dos Algoritmos Genéticos com o poder expressivo da Programação Genética. O GEP usa cromossomos lineares que codificam estruturas de árvore, separando **genótipo** (cromossomo) e **fenótipo** (árvore de expressão).

![Gene Expression Programming Concept](../../images/gene_expression_programming_concept.png)

O GEP é particularmente eficaz em regressão simbólica, classificação, modelagem de séries temporais e descoberta de conhecimento, oferecendo uma abordagem mais simples e eficiente que a GP tradicional.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 Conceito Central**

O GEP introduz uma distinção clara entre **genótipo** e **fenótipo**:

1. **Genótipo:** String linear de símbolos (simples de manipular)
2. **Fenótipo:** Árvore de expressão (poderosa para representar soluções)
3. **Mapeamento:** Tradução automática e não-ambígua do genótipo para fenótipo

**Intuição:**
> "Assim como no DNA biológico, onde genes lineares codificam proteínas tridimensionais, no GEP cromossomos lineares codificam árvores de expressão complexas."

### **1.2 Estrutura do Cromossomo GEP**

#### **Anatomia de um Gene**

Cada gene GEP tem duas partes:

```
Gene = HEAD + TAIL

HEAD (Cabeça):
- Pode conter FUNÇÕES e TERMINAIS
- Tamanho: h (definido pelo usuário)

TAIL (Cauda):
- Contém APENAS TERMINAIS
- Tamanho: t = h * (n_max - 1) + 1
  onde n_max = maior aridade nas funções

Exemplo:
HEAD: + - * X Y 2
TAIL: X Y 1
Gene Completo: + - * X Y 2 X Y 1
             (6 símbolos)  (3 símbolos)
```

#### **Mapeamento para Árvore de Expressão**

```
Gene: + * X Y - 3 | Y X 2
      \_____HEAD_____/  \_TAIL_/

Leitura em breadth-first (largura):

Nível 0:  +          (1 nó, aridade 2)
         / \
Nível 1: *   -       (2 nós, aridade 2 e 2)
        / \   / \
Nível 2: X Y 3  Y    (4 terminais da cauda)

Expressão: (X * Y) + (3 - Y)

Símbolos não utilizados na cauda são ignorados (código neutro)
```

### **1.3 Diferenças de GA e GP**

| Aspecto | GA | GP | GEP |
|---------|----|----|-----|
| **Genótipo** | Linear | Árvore | Linear |
| **Fenótipo** | Direto | Árvore | Árvore |
| **Separação G-F** | ❌ Não | ❌ Não | ✅ Sim |
| **Tamanho** | Fixo | Variável | Fixo |
| **Crossover** | Simples | Complexo | Simples |
| **Mutação** | Simples | Complexa | Simples |
| **Validade** | Sempre | Pode falhar | Sempre |
| **Complexidade Ops** | Baixa | Alta | Baixa |

**Vantagens do GEP:**
- ✅ Operadores genéticos simples como GA
- ✅ Poder expressivo como GP
- ✅ Sempre gera indivíduos válidos
- ✅ Código neutro permite exploração sem penalidade
- ✅ Multi-gênico permite modularidade

---

## **2. 🔧 Algoritmo do GEP**

### **2.1 Estrutura Geral**

```
🚀 1. INICIALIZAÇÃO
   ├── Definir funções: F = {+, -, *, /, sin, cos, ...}
   ├── Definir terminais: T = {X, Y, constantes}
   ├── Definir estrutura do gene:
   │   ├── h = tamanho da HEAD
   │   ├── t = h * (n_max - 1) + 1  (tamanho da TAIL)
   │   └── n_genes = número de genes por cromossomo
   ├── Gerar população de cromossomos aleatórios
   └── Avaliar fitness (mapear para árvore e executar)

🔄 2. LOOP EVOLUTIVO (enquanto não convergir):
   │
   ├── 📊 AVALIAÇÃO
   │   ├── Para cada cromossomo:
   │   │   ├── Decodificar gene → árvore de expressão
   │   │   └── Executar árvore e calcular fitness
   │
   ├── 🎯 SELEÇÃO
   │   └── Selecionar pais (roleta, torneio, etc.)
   │
   ├── 🧬 REPRODUÇÃO
   │   ├── 🔀 CROSSOVER
   │   │   ├── Um ponto (70% probabilidade)
   │   │   ├── Dois pontos
   │   │   └── Gene inteiro
   │   │
   │   ├── 🎲 MUTAÇÃO
   │   │   ├── Mutação de ponto (44 por 1000)
   │   │   ├── Inserção de sequência
   │   │   └── Inversão
   │   │
   │   ├── 🔄 TRANSPOSIÇÃO
   │   │   ├── IS transposição
   │   │   ├── RIS transposição
   │   │   └── Gene transposição
   │   │
   │   └── 🧬 RECOMBINAÇÃO GENE
   │       └── Troca de genes inteiros
   │
   ├── 🔄 SUBSTITUIÇÃO
   │   └── Formar nova geração
   │
   └── 📈 ELITISMO
       └── Preservar melhores indivíduos

🏆 3. RETORNAR melhor solução
```

### **2.2 Operadores Genéticos**

#### **🔀 Crossover (Um Ponto)**

```
Pai 1: + * X | Y 2 1   (HEAD|TAIL)
Pai 2: - / Y | X 3 2
           ↑ ponto de corte

Filho 1: + * X | X 3 2
Filho 2: - / Y | Y 2 1

Características:
- ✅ Simples como GA
- ✅ Sempre válido
- ✅ Preserva estrutura HEAD/TAIL
```

#### **🔀 Crossover de Gene**

```
Pai 1: [Gene1][Gene2][Gene3]
Pai 2: [GeneA][GeneB][GeneC]

Troca Gene2 ↔ GeneB

Filho 1: [Gene1][GeneB][Gene3]
Filho 2: [GeneA][Gene2][GeneC]
```

#### **🎲 Mutação de Ponto**

```
Antes: + * X | Y 2 1
              ↑ mutação
Depois: + * - | Y 2 1

Regras:
- Na HEAD: qualquer símbolo (função ou terminal)
- Na TAIL: apenas terminal
```

#### **🔄 Transposição IS (Insertion Sequence)**

```
Cromossomo: + * X Y - 2 | X Y 1 2

Selecionar sequência: * X
Inserir no início da HEAD: * X + * X Y - 2 | X Y 1 2
                          \_novo_/ \__deslocado__/

Efeito: Move subárvores para raiz
```

#### **🔄 Transposição RIS (Root IS)**

```
Similar a IS, mas:
- Sequência deve começar com FUNÇÃO
- Garante mudança na raiz da árvore
- Maior impacto estrutural
```

#### **🔄 Gene Transposição**

```
Cromossomo: [Gene1][Gene2][Gene3]

Move Gene3 para o início:
Resultado: [Gene3][Gene1][Gene2]

Efeito: Muda gene dominante
```

---

## **3. 💻 Implementação em Python**

### **3.1 Classes Básicas**

```python
import numpy as np
import random
import operator
from typing import List, Callable, Union

class GEPGene:
    """
    Representa um gene GEP com HEAD e TAIL
    """
    
    def __init__(self, head_length, functions, terminals):
        """
        Args:
            head_length: Tamanho da HEAD
            functions: Dicionário {função: aridade}
            terminals: Lista de terminais
        """
        self.head_length = head_length
        self.functions = functions
        self.terminals = terminals
        
        # Calcular tamanho da TAIL
        max_arity = max(functions.values())
        self.tail_length = head_length * (max_arity - 1) + 1
        self.gene_length = self.head_length + self.tail_length
        
        # Gerar gene aleatório
        self.chromosome = self._generate_random_gene()
    
    def _generate_random_gene(self):
        """Gera gene aleatório válido"""
        gene = []
        
        # HEAD: funções e terminais
        all_symbols = list(self.functions.keys()) + self.terminals
        for _ in range(self.head_length):
            gene.append(random.choice(all_symbols))
        
        # TAIL: apenas terminais
        for _ in range(self.tail_length):
            gene.append(random.choice(self.terminals))
        
        return gene
    
    def decode_to_tree(self):
        """
        Decodifica gene para árvore de expressão
        Usa algoritmo breadth-first
        """
        if not self.chromosome:
            return None
        
        # Fila para processamento BFS
        queue = [0]  # Começar com índice 0
        tree = []
        
        idx = 0
        while queue and idx < len(self.chromosome):
            current = queue.pop(0)
            
            if current >= len(self.chromosome):
                break
            
            symbol = self.chromosome[current]
            tree.append(symbol)
            
            # Se é função, adicionar filhos à fila
            if symbol in self.functions:
                arity = self.functions[symbol]
                idx += 1
                for _ in range(arity):
                    if idx < len(self.chromosome):
                        queue.append(idx)
                        idx += 1
        
        return tree
    
    def evaluate(self, context):
        """
        Avalia gene dado contexto de variáveis
        
        Args:
            context: Dicionário {variável: valor}
        
        Returns:
            Resultado da avaliação
        """
        tree = self.decode_to_tree()
        if not tree:
            return 0
        
        # Avaliar árvore recursivamente
        return self._evaluate_tree(tree, 0, context)[0]
    
    def _evaluate_tree(self, tree, idx, context):
        """
        Avalia árvore recursivamente
        
        Returns:
            (resultado, próximo_índice)
        """
        if idx >= len(tree):
            return 0, idx
        
        symbol = tree[idx]
        
        # Terminal
        if symbol not in self.functions:
            if isinstance(symbol, (int, float)):
                return symbol, idx + 1
            else:
                return context.get(symbol, 0), idx + 1
        
        # Função
        arity = self.functions[symbol]
        args = []
        next_idx = idx + 1
        
        for _ in range(arity):
            arg, next_idx = self._evaluate_tree(tree, next_idx, context)
            args.append(arg)
        
        try:
            result = symbol(*args)
            # Tratar valores inválidos
            if np.isnan(result) or np.isinf(result):
                result = 0
        except:
            result = 0
        
        return result, next_idx
    
    def to_string(self):
        """Converte gene para string legível"""
        return ''.join([str(s)[:3] for s in self.chromosome])
    
    def copy(self):
        """Cria cópia do gene"""
        new_gene = GEPGene(self.head_length, self.functions, self.terminals)
        new_gene.chromosome = self.chromosome.copy()
        return new_gene

# Funções seguras
def safe_div(a, b):
    """Divisão protegida"""
    return a / b if abs(b) > 1e-10 else 1.0

def safe_sqrt(x):
    """Raiz quadrada protegida"""
    return np.sqrt(abs(x))

def safe_log(x):
    """Logaritmo protegido"""
    return np.log(abs(x)) if abs(x) > 1e-10 else 0.0

# Conjunto de funções padrão
DEFAULT_FUNCTIONS = {
    operator.add: 2,
    operator.sub: 2,
    operator.mul: 2,
    safe_div: 2,
}

# Conjunto de terminais padrão
DEFAULT_TERMINALS = ['X', 'Y']
```

### **3.2 Classe GEP Completa**

```python
class GeneExpressionProgramming:
    """
    Implementação completa de Gene Expression Programming
    """
    
    def __init__(self,
                 functions=None,
                 terminals=None,
                 head_length=7,
                 n_genes=3,
                 pop_size=100,
                 generations=100,
                 mutation_rate=0.044,  # 44 por 1000
                 crossover_rate=0.7,
                 gene_crossover_rate=0.3,
                 transposition_rate=0.1,
                 tournament_size=7):
        """
        Args:
            functions: Dicionário {função: aridade}
            terminals: Lista de terminais
            head_length: Tamanho da HEAD de cada gene
            n_genes: Número de genes por cromossomo
            pop_size: Tamanho da população
            generations: Número de gerações
            mutation_rate: Taxa de mutação por gene
            crossover_rate: Taxa de crossover de ponto
            gene_crossover_rate: Taxa de crossover de gene
            transposition_rate: Taxa de transposição
            tournament_size: Tamanho do torneio
        """
        self.functions = functions or DEFAULT_FUNCTIONS
        self.terminals = terminals or DEFAULT_TERMINALS
        self.head_length = head_length
        self.n_genes = n_genes
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.gene_crossover_rate = gene_crossover_rate
        self.transposition_rate = transposition_rate
        self.tournament_size = tournament_size
    
    def create_individual(self):
        """Cria indivíduo (cromossomo multi-gênico)"""
        genes = []
        for _ in range(self.n_genes):
            gene = GEPGene(self.head_length, self.functions, self.terminals)
            genes.append(gene)
        return genes
    
    def initialize_population(self):
        """Inicializa população"""
        return [self.create_individual() for _ in range(self.pop_size)]
    
    def evaluate_individual(self, individual, X, y, linking='add'):
        """
        Avalia indivíduo multi-gênico
        
        Args:
            individual: Lista de genes
            X: Dados de entrada
            y: Valores alvo
            linking: Função para combinar genes ('add', 'mul', 'avg')
        
        Returns:
            fitness (menor é melhor)
        """
        try:
            predictions = []
            
            for sample in X:
                # Criar contexto
                context = {}
                for i, term in enumerate(self.terminals):
                    if i < len(sample):
                        context[term] = sample[i]
                
                # Avaliar cada gene
                gene_outputs = []
                for gene in individual:
                    output = gene.evaluate(context)
                    gene_outputs.append(output)
                
                # Combinar saídas dos genes
                if linking == 'add':
                    pred = sum(gene_outputs)
                elif linking == 'mul':
                    pred = np.prod(gene_outputs)
                elif linking == 'avg':
                    pred = np.mean(gene_outputs)
                else:
                    pred = gene_outputs[0]
                
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calcular erro MSE
            error = np.mean((predictions - y) ** 2)
            
            # Penalizar complexidade (opcional)
            complexity_penalty = 0.0001 * sum(g.gene_length for g in individual)
            
            return error + complexity_penalty
        
        except Exception as e:
            return 1e10
    
    def tournament_selection(self, population, fitnesses):
        """Seleção por torneio"""
        selected = random.sample(range(len(population)), self.tournament_size)
        best_idx = min(selected, key=lambda i: fitnesses[i])
        return [gene.copy() for gene in population[best_idx]]
    
    def one_point_crossover(self, parent1, parent2):
        """Crossover de um ponto"""
        child1 = [gene.copy() for gene in parent1]
        child2 = [gene.copy() for gene in parent2]
        
        # Para cada gene, aplicar crossover
        for i in range(len(child1)):
            if random.random() < self.crossover_rate:
                # Ponto de corte aleatório
                point = random.randint(1, child1[i].gene_length - 1)
                
                # Trocar segmentos
                child1[i].chromosome[point:], child2[i].chromosome[point:] = \
                    child2[i].chromosome[point:].copy(), child1[i].chromosome[point:].copy()
        
        return child1, child2
    
    def gene_crossover(self, parent1, parent2):
        """Crossover de gene inteiro"""
        child1 = [gene.copy() for gene in parent1]
        child2 = [gene.copy() for gene in parent2]
        
        if random.random() < self.gene_crossover_rate and len(child1) > 1:
            # Escolher gene aleatório para trocar
            gene_idx = random.randint(0, len(child1) - 1)
            child1[gene_idx], child2[gene_idx] = child2[gene_idx], child1[gene_idx]
        
        return child1, child2
    
    def mutate(self, individual):
        """Mutação de ponto"""
        mutant = [gene.copy() for gene in individual]
        
        for gene in mutant:
            for i in range(gene.gene_length):
                if random.random() < self.mutation_rate:
                    if i < gene.head_length:
                        # HEAD: qualquer símbolo
                        all_symbols = list(gene.functions.keys()) + gene.terminals
                        gene.chromosome[i] = random.choice(all_symbols)
                    else:
                        # TAIL: apenas terminal
                        gene.chromosome[i] = random.choice(gene.terminals)
        
        return mutant
    
    def is_transposition(self, individual):
        """Transposição IS (Insertion Sequence)"""
        mutant = [gene.copy() for gene in individual]
        
        if random.random() < self.transposition_rate:
            # Escolher gene aleatório
            gene = random.choice(mutant)
            
            # Escolher sequência para transpor (1-3 símbolos)
            seq_len = random.randint(1, 3)
            start = random.randint(0, gene.head_length - seq_len)
            sequence = gene.chromosome[start:start + seq_len]
            
            # Inserir no início
            gene.chromosome = sequence + gene.chromosome[:gene.head_length - seq_len] + \
                             gene.chromosome[gene.head_length:]
        
        return mutant
    
    def ris_transposition(self, individual):
        """Transposição RIS (Root IS)"""
        mutant = [gene.copy() for gene in individual]
        
        if random.random() < self.transposition_rate:
            gene = random.choice(mutant)
            
            # Buscar sequência que começa com função
            attempts = 0
            while attempts < 10:
                seq_len = random.randint(1, 3)
                start = random.randint(0, gene.head_length - seq_len)
                
                if gene.chromosome[start] in gene.functions:
                    sequence = gene.chromosome[start:start + seq_len]
                    gene.chromosome = sequence + gene.chromosome[:gene.head_length - seq_len] + \
                                     gene.chromosome[gene.head_length:]
                    break
                
                attempts += 1
        
        return mutant
    
    def gene_transposition(self, individual):
        """Gene transposição"""
        mutant = [gene.copy() for gene in individual]
        
        if random.random() < self.transposition_rate and len(mutant) > 1:
            # Escolher gene aleatório e mover para frente
            gene_idx = random.randint(1, len(mutant) - 1)
            gene = mutant.pop(gene_idx)
            mutant.insert(0, gene)
        
        return mutant
    
    def evolve(self, X_train, y_train, linking='add', verbose=True):
        """
        Executa evolução
        
        Args:
            X_train: Dados de treinamento
            y_train: Valores alvo
            linking: Função para combinar genes
            verbose: Imprimir progresso
        
        Returns:
            best_individual: Melhor indivíduo
            best_fitness: Fitness do melhor
            history: Histórico de evolução
        """
        # Inicializar população
        population = self.initialize_population()
        
        # Avaliar população inicial
        fitnesses = [self.evaluate_individual(ind, X_train, y_train, linking) 
                    for ind in population]
        
        # Melhor solução
        best_idx = np.argmin(fitnesses)
        best_individual = [gene.copy() for gene in population[best_idx]]
        best_fitness = fitnesses[best_idx]
        
        history = {
            'best_fitness': [best_fitness],
            'avg_fitness': [np.mean(fitnesses)],
            'std_fitness': [np.std(fitnesses)]
        }
        
        # Loop evolutivo
        for generation in range(self.generations):
            new_population = []
            
            # Elitismo
            new_population.append([gene.copy() for gene in best_individual])
            
            # Gerar resto da população
            while len(new_population) < self.pop_size:
                # Seleção
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover de ponto
                child1, child2 = self.one_point_crossover(parent1, parent2)
                
                # Crossover de gene
                child1, child2 = self.gene_crossover(child1, child2)
                
                # Mutação
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Transposição
                child1 = self.is_transposition(child1)
                child1 = self.ris_transposition(child1)
                child1 = self.gene_transposition(child1)
                
                child2 = self.is_transposition(child2)
                child2 = self.ris_transposition(child2)
                child2 = self.gene_transposition(child2)
                
                new_population.extend([child1, child2])
            
            # Atualizar população
            population = new_population[:self.pop_size]
            
            # Avaliar nova população
            fitnesses = [self.evaluate_individual(ind, X_train, y_train, linking) 
                        for ind in population]
            
            # Atualizar melhor
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_individual = [gene.copy() for gene in population[gen_best_idx]]
                best_fitness = fitnesses[gen_best_idx]
            
            # Registrar histórico
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(np.mean(fitnesses))
            history['std_fitness'].append(np.std(fitnesses))
            
            # Imprimir progresso
            if verbose and generation % 10 == 0:
                print(f"Gen {generation}: Best fitness = {best_fitness:.6f}, "
                      f"Avg = {history['avg_fitness'][-1]:.6f}")
        
        return best_individual, best_fitness, history

# Exemplo de uso
def example_gep_regression():
    """
    Exemplo de regressão simbólica com GEP
    """
    # Gerar dados: y = x^2 + 2*x + 1
    X_train = np.linspace(-5, 5, 100).reshape(-1, 1)
    y_train = X_train[:, 0]**2 + 2*X_train[:, 0] + 1
    
    # Configurar GEP
    gep = GeneExpressionProgramming(
        functions=DEFAULT_FUNCTIONS,
        terminals=['X'],
        head_length=7,
        n_genes=3,
        pop_size=100,
        generations=100,
        mutation_rate=0.044,
        crossover_rate=0.7,
        gene_crossover_rate=0.3,
        transposition_rate=0.1
    )
    
    # Evoluir
    best_ind, best_fitness, history = gep.evolve(
        X_train, y_train, 
        linking='add',
        verbose=True
    )
    
    print(f"\nMelhor fitness: {best_fitness:.6f}")
    print(f"\nGenes do melhor indivíduo:")
    for i, gene in enumerate(best_ind):
        print(f"  Gene {i+1}: {gene.to_string()}")
    
    return best_ind, history

# Executar exemplo
if __name__ == "__main__":
    best_solution, history = example_gep_regression()
```

---

## **4. 🎯 Aplicações do GEP**

### **4.1 Regressão Simbólica**

```python
def gep_symbolic_regression():
    """
    Descobre fórmula para relação física
    """
    # Lei de queda livre: h = h0 - (1/2)*g*t^2
    # Simplificado: h = 10 - 5*t^2
    
    t = np.linspace(0, 1.4, 50)
    h = 10 - 5 * t**2 + np.random.normal(0, 0.1, 50)
    
    X_train = t.reshape(-1, 1)
    y_train = h
    
    # Funções para física
    functions = {
        operator.add: 2,
        operator.sub: 2,
        operator.mul: 2,
        safe_div: 2,
        lambda x: x**2: 1,
        safe_sqrt: 1
    }
    
    gep = GeneExpressionProgramming(
        functions=functions,
        terminals=['t'],
        head_length=10,
        n_genes=2,
        pop_size=200,
        generations=150
    )
    
    best, fitness, history = gep.evolve(X_train, y_train, linking='add')
    
    print("Fórmula descoberta para h(t):")
    for i, gene in enumerate(best):
        print(f"  Componente {i+1}: {gene.to_string()}")
    print(f"Erro MSE: {fitness:.6f}")
```

### **4.2 Classificação**

```python
def gep_classification():
    """
    Classificação binária com GEP
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Dados de classificação
    X, y = make_classification(
        n_samples=200, 
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Fitness para classificação
    class GEPClassifier(GeneExpressionProgramming):
        def evaluate_individual(self, individual, X, y, linking='add'):
            try:
                predictions = []
                
                for sample in X:
                    context = {
                        'X': sample[0],
                        'Y': sample[1],
                        'Z': sample[2]
                    }
                    
                    # Avaliar genes
                    gene_outputs = [gene.evaluate(context) for gene in individual]
                    
                    # Combinar
                    if linking == 'add':
                        score = sum(gene_outputs)
                    else:
                        score = np.mean(gene_outputs)
                    
                    # Classificar: positivo ou negativo
                    pred_class = 1 if score > 0 else 0
                    predictions.append(pred_class)
                
                # Erro de classificação
                accuracy = np.mean(np.array(predictions) == y)
                return 1 - accuracy  # Minimizar erro
                
            except:
                return 1.0
    
    # Treinar
    gep = GEPClassifier(
        terminals=['X', 'Y', 'Z'],
        head_length=8,
        n_genes=3,
        pop_size=150,
        generations=100
    )
    
    best, fitness, history = gep.evolve(X_train, y_train)
    
    # Avaliar em teste
    test_fitness = gep.evaluate_individual(best, X_test, y_test)
    
    print(f"Acurácia treino: {(1 - fitness) * 100:.2f}%")
    print(f"Acurácia teste: {(1 - test_fitness) * 100:.2f}%")
```

### **4.3 Séries Temporais**

```python
def gep_time_series():
    """
    Previsão de séries temporais com GEP
    """
    # Série temporal: sin wave com tendência
    t = np.linspace(0, 10, 200)
    series = np.sin(2 * np.pi * 0.5 * t) + 0.1 * t + np.random.normal(0, 0.1, 200)
    
    # Criar features: janelas deslizantes
    window_size = 5
    X = []
    y = []
    
    for i in range(window_size, len(series)):
        X.append(series[i-window_size:i])
        y.append(series[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Terminais: valores anteriores
    terminals = [f'T{i}' for i in range(1, window_size + 1)]
    
    gep = GeneExpressionProgramming(
        terminals=terminals,
        head_length=10,
        n_genes=4,
        pop_size=200,
        generations=150
    )
    
    # Adaptar evaluate para usar nomes corretos
    best, fitness, history = gep.evolve(X_train, y_train)
    
    # Testar
    test_fitness = gep.evaluate_individual(best, X_test, y_test)
    
    print(f"MSE treino: {fitness:.6f}")
    print(f"MSE teste: {test_fitness:.6f}")
```

---

## **5. ✅ Vantagens e ❌ Desvantagens**

### **5.1 ✅ Vantagens**

| Vantagem | Descrição | vs GP | vs GA |
|----------|-----------|-------|-------|
| **Operadores Simples** | Crossover e mutação como GA | ✅✅ | ⚖️ |
| **Sempre Válido** | Nunca gera indivíduos inválidos | ✅✅ | ⚖️ |
| **Código Neutro** | Permite exploração sem penalidade | ✅ | ✅ |
| **Multi-gênico** | Modularidade natural | ✅ | ⚪ |
| **Poder Expressivo** | Árvores complexas como GP | ⚖️ | ✅✅ |
| **Controle de Bloat** | Tamanho fixo previne crescimento | ✅✅ | ⚖️ |
| **Eficiência** | Mais rápido que GP tradicional | ✅ | ⚪ |
| **Interpretabilidade** | Genes individuais são legíveis | ⚪ | ✅ |

### **5.2 ❌ Desvantagens**

| Desvantagem | Descrição | Mitigação |
|-------------|-----------|-----------|
| **Complexidade Conceitual** | Separação genótipo-fenótipo não é intuitiva | Estudo e exemplos |
| **Menos Conhecido** | Menos popular que GA/GP | Usar bibliotecas |
| **Parâmetros Adicionais** | HEAD length, linking function | Valores padrão |
| **Código Neutro** | Parte do gene pode ser inativa | É uma feature, não bug |
| **Linking Heurístico** | Combinar genes é ad-hoc | Testar diferentes funções |

### **5.3 🎯 Quando Usar GEP**

#### **✅ Cenários Ideais:**
- ✅ Regressão simbólica
- ✅ Classificação com regras
- ✅ Séries temporais
- ✅ Quando quer GP mas com operadores simples
- ✅ Problemas que beneficiam de modularidade (multi-genes)
- ✅ Quando bloat é problema em GP
- ✅ Feature engineering automático
- ✅ Descoberta de conhecimento

#### **❌ Evite GEP quando:**
- ❌ Problema é puramente numérico (usar DE/ES)
- ❌ GA resolve bem (não precisa de árvores)
- ❌ Interpretabilidade não importa (usar neural networks)
- ❌ Precisa de garantias teóricas
- ❌ Recursos computacionais são muito limitados

---

## **6. 🔬 Comparação GP vs GEP**

### **6.1 Diferenças Fundamentais**

| Aspecto | GP | GEP |
|---------|----|----|
| **Representação** | Árvore diretamente | String linear → Árvore |
| **Validade** | Pode gerar inválidos | Sempre válido |
| **Crossover** | Complexo (subárvores) | Simples (strings) |
| **Mutação** | Complexa | Simples |
| **Tamanho** | Variável (bloat) | Fixo (HEAD+TAIL) |
| **Modularidade** | Difícil | Natural (multi-genes) |
| **Eficiência** | Média | Alta |
| **Popularidade** | Alta | Média |

### **6.2 Exemplo Comparativo**

```python
# Mesmo problema: y = x^2 + x

# GP: Representa diretamente como árvore
#     +
#    / \
#   ^   X
#  / \
# X   2

# GEP: Representa como string
# Gene: + ^ * X 2 | X X X
#       (HEAD)     (TAIL)
# Decodifica para árvore similar
```

### **6.3 Quando Preferir Cada Um**

**Prefira GP quando:**
- ✅ Já domina GP
- ✅ Usa biblioteca madura (DEAP, gplearn)
- ✅ Problema beneficia de tamanho variável
- ✅ Comunidade e recursos são importantes

**Prefira GEP quando:**
- ✅ Quer simplicidade de operadores
- ✅ Bloat é problema
- ✅ Modularidade é importante
- ✅ Quer explorar código neutro
- ✅ Implementação própria

---

## **7. 📚 Recursos e Referências**

### **7.1 📖 Publicações Fundamentais**

1. **Ferreira, C. (2001).** *"Gene Expression Programming: A New Adaptive Algorithm for Solving Problems"*. Complex Systems, 13(2), 87-129.
   - 🌟 Artigo original que introduziu GEP
   - 📊 Teoria completa e comparações

2. **Ferreira, C. (2006).** *"Gene Expression Programming: Mathematical Modeling by an Artificial Intelligence"*. Springer.
   - 📖 Livro definitivo sobre GEP
   - 🎯 Teoria, implementação e aplicações

3. **Ferreira, C. (2002).** *"Gene Expression Programming in Problem Solving"*. Soft Computing and Industry, 635-653.
   - 📊 Aplicações práticas
   - 🔬 Estudos de caso

### **7.2 🛠️ Bibliotecas e Implementações**

```python
# 1. geppy - GEP em Python (mais completo)
pip install geppy

import geppy as gep
from geppy import SymbolicRegressor

# 2. Implementação em R
# install.packages("rgep")
library(rgep)

# 3. Gene Expression Programming  em C++
# Ver GitHub: GeneXproTools

# 4. Matlab GEP Toolbox
# Ver MathWorks File Exchange
```

### **7.3 🌐 Recursos Online**

| Recurso | Descrição | Link |
|---------|-----------|------|
| **GEP Official Site** | Site oficial de Cândida Ferreira | gene-expression-programming.com |
| **geppy Documentation** | Documentação da biblioteca Python | geppy.readthedocs.io |
| **GEP Book** | Livro completo online | gepsoft.com/gep-book |
| **Tutorials** | Tutoriais e exemplos | Vários blogs e YouTube |

### **7.4 📝 Papers Aplicados**

1. **Bioinformática:** Gene networks, protein structure prediction
2. **Finanças:** Stock market prediction, risk assessment
3. **Engenharia:** Design optimization, fault diagnosis
4. **Medicina:** Disease diagnosis, drug discovery
5. **Ambiental:** Climate modeling, pollution prediction

---

## **8. 🎯 Conclusão**

Gene Expression Programming representa uma **síntese elegante** entre Algoritmos Genéticos e Programação Genética, oferecendo o melhor dos dois mundos.

### **🔑 Principais Aprendizados**

1. **Separação Genótipo-Fenótipo:** Ideia central que permite operadores simples e resultados complexos
2. **Código Neutro:** Característica única que facilita exploração
3. **Multi-gênico:** Modularidade natural para problemas complexos
4. **Controle de Bloat:** Tamanho fixo previne crescimento excessivo
5. **Simplicidade Operacional:** Operadores como GA, poder como GP

### **💡 GEP no Contexto dos EAs**

```
Evolução dos Algoritmos Evolutivos:

GA (1970s)
├─ Representação: Linear, simples
├─ Aplicação: Otimização de parâmetros
└─ Limitação: Estrutura fixa

GP (1990)
├─ Representação: Árvore, complexa
├─ Aplicação: Evolução de programas
└─ Limitação: Operadores complexos, bloat

GEP (2001)
├─ Representação: Linear → Árvore
├─ Aplicação: Melhor dos dois mundos
└─ Inovação: Separação G-F, código neutro
```

### **🚀 Próximos Passos**

1. **Implemente:** Começar com exemplo simples de regressão
2. **Use geppy:** Experimente biblioteca pronta
3. **Compare:** Teste GEP vs GP vs regressão tradicional
4. **Explore:** Multi-genes, diferentes linking functions
5. **Aplique:** Use em problemas do seu domínio
6. **Otimize:** Ajuste HEAD length e operadores
7. **Visualize:** Veja genótipos e fenótipos

### **🌟 Reflexão Final**

Gene Expression Programming demonstra que **inovações conceituais simples** podem ter impacto profundo. Ao separar como soluções são representadas (genótipo) de como são expressas (fenótipo), GEP consegue combinar simplicidade operacional com poder expressivo - uma lição valiosa para design de algoritmos.

> *"A genialidade do GEP está em reconhecer que a forma como codificamos soluções (genótipo linear) não precisa ser a forma como as executamos (fenótipo em árvore). Esta separação libera ambos para serem ótimos em seus papéis."*

**Destaque Principal:** Para regressão simbólica e descoberta de fórmulas, GEP oferece uma alternativa superior à GP tradicional, com operadores mais simples e controle de bloat integrado.

---

**🔗 Continue explorando:**
- 📖 Compare com [**Genetic Programming**](genetic_programming.md) para entender diferenças
- 🧬 Veja [**Genetic Algorithms**](genetic_algorithms.md) para base conceitual
- 🎯 Explore [**Algoritmos Evolucionários**](README.md) para visão completa
- 📊 Estude [**Differential Evolution**](differential_evolution.md) para otimização numérica

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

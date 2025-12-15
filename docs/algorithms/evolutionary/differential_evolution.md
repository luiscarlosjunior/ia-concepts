# Evolução Diferencial (Differential Evolution - DE)

A **Evolução Diferencial** (Differential Evolution - DE) é um algoritmo de otimização evolutiva poderoso e eficiente, desenvolvido por Rainer Storn e Kenneth Price em 1995. É especialmente eficaz para otimização de funções contínuas, multimodais e não-diferenciáveis, sendo amplamente utilizado em engenharia, aprendizado de máquina e otimização numérica.

![Differential Evolution Concept](../../images/differential_evolution_concept.png)

O algoritmo se destaca pela sua simplicidade de implementação, poucos parâmetros de controle e excelente desempenho em problemas de alta dimensão. A ideia central é usar diferenças vetoriais entre membros da população para gerar mutações, criando um mecanismo de busca auto-adaptativo e robusto.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 Conceito Central**

A Evolução Diferencial opera em espaços de busca contínuos e se baseia em três operadores principais:

1. **Mutação Diferencial:** Usa diferenças entre vetores da população para criar novos candidatos
2. **Crossover (Recombinação):** Combina o vetor mutante com o vetor alvo
3. **Seleção:** Escolhe o melhor entre o candidato atual e o novo

**Intuição:**
> "Se a diferença entre dois bons vetores aponta em uma direção promissora, usá-la para criar novos candidatos pode levar a soluções ainda melhores."

### **1.2 Por Que DE Funciona?**

#### **🔍 Vantagens da Abordagem Diferencial**

```
✅ Auto-adaptação:
   - O tamanho do passo se ajusta automaticamente
   - Passos grandes em regiões de exploração
   - Passos pequenos próximo ao ótimo

✅ Balanceamento Exploração-Explotação:
   - Diversidade mantida pela população
   - Convergência através da seleção gulosa

✅ Simplicidade:
   - Poucos parâmetros: F (escala), CR (crossover), NP (população)
   - Não requer informações de gradiente
   - Implementação direta
```

#### **📊 Diferença de Outros EAs**

| Aspecto | Algoritmos Genéticos | Evolution Strategies | Differential Evolution |
|---------|---------------------|---------------------|------------------------|
| **Mutação** | Bit-flip ou Gaussiana | Gaussiana adaptativa | Diferenças vetoriais |
| **Adaptação** | Externa | Auto-adaptação | Implícita no operador |
| **Crossover** | Importante | Opcional | Essencial (binomial) |
| **Tipo de Variável** | Binária/Real | Real | Real |
| **Aplicação Principal** | Combinatória | Contínua | Contínua |

---

## **2. 🔧 Algoritmo da Evolução Diferencial**

### **2.1 Estrutura Geral**

```
🚀 1. INICIALIZAÇÃO
   ├── Gerar população inicial: X₀ = {x₁, x₂, ..., xₙₚ}
   ├── Definir parâmetros: F (escala), CR (crossover), NP (tamanho)
   └── Avaliar fitness de cada indivíduo

🔄 2. LOOP EVOLUTIVO (para cada geração g):
   │
   PARA cada indivíduo xᵢ na população:
   │
   ├── 🧬 MUTAÇÃO
   │   ├── Selecionar 3 vetores distintos: xᵣ₁, xᵣ₂, xᵣ₃ (r1≠r2≠r3≠i)
   │   └── Criar vetor mutante: vᵢ = xᵣ₁ + F·(xᵣ₂ - xᵣ₃)
   │
   ├── 🔀 CROSSOVER (RECOMBINAÇÃO)
   │   ├── PARA cada dimensão j:
   │   │   SE (rand() < CR) OU (j == jᵣₐₙ𝒹):
   │   │       uᵢⱼ = vᵢⱼ
   │   │   SENÃO:
   │   │       uᵢⱼ = xᵢⱼ
   │   └── Vetor trial: uᵢ = (uᵢ₁, uᵢ₂, ..., uᵢ𝒹)
   │
   └── 🎯 SELEÇÃO
       SE f(uᵢ) ≤ f(xᵢ):  # Para minimização
           xᵢ⁽ᵍ⁺¹⁾ = uᵢ
       SENÃO:
           xᵢ⁽ᵍ⁺¹⁾ = xᵢ

🏆 3. RETORNAR melhor solução encontrada
```

### **2.2 Variantes do Operador de Mutação**

A notação DE/x/y/z especifica:
- **x**: Vetor base (rand, best, current-to-best)
- **y**: Número de diferenças vetoriais
- **z**: Tipo de crossover (bin, exp)

#### **Estratégias Principais:**

**1. DE/rand/1/bin (Clássica)**
```
vᵢ = xᵣ₁ + F·(xᵣ₂ - xᵣ₃)
```
- ✅ Boa diversidade
- ⚪ Convergência moderada
- 🎯 Uso: Exploração ampla

**2. DE/best/1/bin**
```
vᵢ = xbest + F·(xᵣ₁ - xᵣ₂)
```
- ✅ Convergência rápida
- ❌ Pode ficar preso em ótimos locais
- 🎯 Uso: Funções unimodais

**3. DE/current-to-best/1/bin**
```
vᵢ = xᵢ + F·(xbest - xᵢ) + F·(xᵣ₁ - xᵣ₂)
```
- ✅ Balanceamento exploração-explotação
- ✅ Convergência estável
- 🎯 Uso: Problemas multimodais

**4. DE/best/2/bin**
```
vᵢ = xbest + F·(xᵣ₁ - xᵣ₂) + F·(xᵣ₃ - xᵣ₄)
```
- ✅ Busca mais agressiva
- ⚪ Requer população maior
- 🎯 Uso: Funções complexas

**5. DE/rand/2/bin**
```
vᵢ = xᵣ₁ + F·(xᵣ₂ - xᵣ₃) + F·(xᵣ₄ - xᵣ₅)
```
- ✅ Máxima diversidade
- ⚪ Convergência mais lenta
- 🎯 Uso: Alta dimensionalidade

---

## **3. 💻 Implementação em Python**

### **3.1 Implementação Básica**

```python
import numpy as np

class DifferentialEvolution:
    """
    Implementação da Evolução Diferencial (DE/rand/1/bin)
    """
    
    def __init__(self, objective_function, bounds, pop_size=50, 
                 F=0.8, CR=0.9, max_iter=1000):
        """
        Args:
            objective_function: Função a ser minimizada
            bounds: Lista de tuplas (min, max) para cada dimensão
            pop_size: Tamanho da população (NP)
            F: Fator de escala diferencial (0 < F ≤ 2)
            CR: Taxa de crossover (0 ≤ CR ≤ 1)
            max_iter: Número máximo de gerações
        """
        self.f = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.NP = pop_size
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
        
    def initialize_population(self):
        """Inicializa população aleatoriamente dentro dos limites"""
        pop = np.random.rand(self.NP, self.dim)
        for i in range(self.dim):
            pop[:, i] = self.bounds[i, 0] + pop[:, i] * (
                self.bounds[i, 1] - self.bounds[i, 0]
            )
        return pop
    
    def mutate(self, population, current_idx):
        """
        Operador de mutação DE/rand/1
        vᵢ = xᵣ₁ + F·(xᵣ₂ - xᵣ₃)
        """
        # Selecionar 3 índices distintos (diferentes de current_idx)
        candidates = [idx for idx in range(self.NP) if idx != current_idx]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        
        # Criar vetor mutante
        mutant = population[r1] + self.F * (population[r2] - population[r3])
        
        return mutant
    
    def crossover(self, target, mutant):
        """
        Operador de crossover binomial
        """
        trial = np.copy(target)
        
        # Garantir pelo menos uma dimensão do mutante
        j_rand = np.random.randint(0, self.dim)
        
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def clip_to_bounds(self, vector):
        """Garante que o vetor está dentro dos limites"""
        return np.clip(vector, self.bounds[:, 0], self.bounds[:, 1])
    
    def optimize(self):
        """Executa o algoritmo DE"""
        # Inicialização
        population = self.initialize_population()
        fitness = np.array([self.f(ind) for ind in population])
        
        # Melhor solução
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Histórico
        history = {
            'best_fitness': [best_fitness],
            'avg_fitness': [np.mean(fitness)],
            'std_fitness': [np.std(fitness)]
        }
        
        # Loop evolutivo
        for generation in range(self.max_iter):
            # Para cada indivíduo
            for i in range(self.NP):
                # Mutação
                mutant = self.mutate(population, i)
                
                # Garantir limites
                mutant = self.clip_to_bounds(mutant)
                
                # Crossover
                trial = self.crossover(population[i], mutant)
                
                # Seleção
                trial_fitness = self.f(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Atualizar melhor global
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            # Registrar histórico
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(np.mean(fitness))
            history['std_fitness'].append(np.std(fitness))
            
            # Critério de parada (opcional)
            if history['std_fitness'][-1] < 1e-8:
                print(f"Convergência atingida na geração {generation}")
                break
        
        return best_solution, best_fitness, history

# Exemplo de uso
def sphere_function(x):
    """Função esfera: f(x) = sum(xᵢ²)"""
    return np.sum(x**2)

# Configurar problema
bounds = [(-5.0, 5.0)] * 10  # 10 dimensões
de = DifferentialEvolution(
    objective_function=sphere_function,
    bounds=bounds,
    pop_size=50,
    F=0.8,
    CR=0.9,
    max_iter=500
)

# Executar otimização
best_solution, best_fitness, history = de.optimize()

print(f"Melhor solução encontrada: {best_solution}")
print(f"Melhor fitness: {best_fitness}")
```

### **3.2 Implementação com Estratégias Múltiplas**

```python
class AdaptiveDifferentialEvolution(DifferentialEvolution):
    """
    DE com múltiplas estratégias de mutação
    """
    
    def __init__(self, *args, strategy='rand/1', **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
    
    def mutate(self, population, current_idx, fitness=None):
        """Aplica estratégia de mutação selecionada"""
        candidates = [idx for idx in range(self.NP) if idx != current_idx]
        
        if self.strategy == 'rand/1':
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = population[r1] + self.F * (population[r2] - population[r3])
            
        elif self.strategy == 'best/1':
            best_idx = np.argmin(fitness)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            mutant = population[best_idx] + self.F * (population[r1] - population[r2])
            
        elif self.strategy == 'current-to-best/1':
            best_idx = np.argmin(fitness)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            mutant = (population[current_idx] + 
                     self.F * (population[best_idx] - population[current_idx]) +
                     self.F * (population[r1] - population[r2]))
            
        elif self.strategy == 'rand/2':
            r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
            mutant = (population[r1] + 
                     self.F * (population[r2] - population[r3]) +
                     self.F * (population[r4] - population[r5]))
        
        else:
            raise ValueError(f"Estratégia desconhecida: {self.strategy}")
        
        return mutant
```

### **3.3 Visualização da Convergência**

```python
import matplotlib.pyplot as plt

def plot_convergence(history):
    """Plota convergência do algoritmo DE"""
    plt.figure(figsize=(12, 4))
    
    # Fitness ao longo das gerações
    plt.subplot(1, 2, 1)
    plt.plot(history['best_fitness'], 'b-', label='Melhor Fitness', linewidth=2)
    plt.plot(history['avg_fitness'], 'r--', label='Fitness Médio', linewidth=1)
    plt.xlabel('Geração')
    plt.ylabel('Fitness')
    plt.title('Convergência do DE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Diversidade da população
    plt.subplot(1, 2, 2)
    plt.plot(history['std_fitness'], 'g-', linewidth=2)
    plt.xlabel('Geração')
    plt.ylabel('Desvio Padrão do Fitness')
    plt.title('Diversidade da População')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

# Uso
plot_convergence(history)
```

---

## **4. 🎯 Exemplos de Aplicação**

### **4.1 Otimização de Funções de Benchmark**

```python
# Função de Rastrigin (multimodal)
def rastrigin(x):
    """
    Função de Rastrigin: altamente multimodal
    Mínimo global: f(0,...,0) = 0
    """
    n = len(x)
    A = 10
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Função de Rosenbrock (vale)
def rosenbrock(x):
    """
    Função de Rosenbrock: vale estreito
    Mínimo global: f(1,...,1) = 0
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Função de Ackley
def ackley(x):
    """
    Função de Ackley: muitos ótimos locais
    Mínimo global: f(0,...,0) = 0
    """
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

# Otimizar com DE
bounds = [(-5.12, 5.12)] * 10
de = DifferentialEvolution(rastrigin, bounds, pop_size=100, F=0.8, CR=0.9)
solution, fitness, history = de.optimize()

print(f"Rastrigin - Melhor fitness: {fitness:.6f}")
```

### **4.2 Ajuste de Hiperparâmetros de ML**

```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

def optimize_svm_hyperparameters(X, y):
    """
    Otimiza hiperparâmetros de SVM usando DE
    """
    def objective(params):
        C, gamma = 10**params[0], 10**params[1]
        svm = SVC(C=C, gamma=gamma, kernel='rbf')
        # Minimizar erro (1 - accuracy)
        score = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
        return 1 - score.mean()
    
    # Limites: log10(C) e log10(gamma)
    bounds = [(-3, 3), (-3, 3)]
    
    de = DifferentialEvolution(
        objective_function=objective,
        bounds=bounds,
        pop_size=20,
        F=0.8,
        CR=0.7,
        max_iter=50
    )
    
    best_params, best_error, history = de.optimize()
    C_opt = 10**best_params[0]
    gamma_opt = 10**best_params[1]
    
    return C_opt, gamma_opt, 1 - best_error

# Exemplo
data = load_iris()
X, y = data.data, data.target

C_opt, gamma_opt, accuracy = optimize_svm_hyperparameters(X, y)
print(f"Melhores hiperparâmetros:")
print(f"  C = {C_opt:.4f}")
print(f"  gamma = {gamma_opt:.4f}")
print(f"  Acurácia = {accuracy:.4f}")
```

### **4.3 Treinamento de Redes Neurais**

```python
def train_neural_network_with_de():
    """
    Treina uma rede neural simples usando DE
    """
    import torch
    import torch.nn as nn
    
    # Dados de exemplo
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    
    class SimpleNet(nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)
            self.set_weights(weights)
        
        def set_weights(self, weights):
            # Dividir vetor de pesos em camadas
            idx = 0
            # fc1: 10*5 + 5 = 55 parâmetros
            w1_size = 10 * 5
            self.fc1.weight.data = torch.tensor(
                weights[idx:idx+w1_size].reshape(5, 10), dtype=torch.float32
            )
            idx += w1_size
            self.fc1.bias.data = torch.tensor(
                weights[idx:idx+5], dtype=torch.float32
            )
            idx += 5
            # fc2: 5*1 + 1 = 6 parâmetros
            w2_size = 5 * 1
            self.fc2.weight.data = torch.tensor(
                weights[idx:idx+w2_size].reshape(1, 5), dtype=torch.float32
            )
            idx += w2_size
            self.fc2.bias.data = torch.tensor(
                weights[idx:idx+1], dtype=torch.float32
            )
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    def objective(weights):
        model = SimpleNet(weights)
        y_pred = model(X_train)
        loss = nn.MSELoss()(y_pred, y_train)
        return loss.item()
    
    # Total de parâmetros: 55 + 5 + 6 = 66
    num_params = 10*5 + 5 + 5*1 + 1
    bounds = [(-1.0, 1.0)] * num_params
    
    de = DifferentialEvolution(
        objective_function=objective,
        bounds=bounds,
        pop_size=50,
        F=0.8,
        CR=0.9,
        max_iter=200
    )
    
    best_weights, best_loss, history = de.optimize()
    
    print(f"Melhor loss: {best_loss:.6f}")
    return best_weights, history
```

---

## **5. ⚙️ Configuração de Parâmetros**

### **5.1 Parâmetros Principais**

| Parâmetro | Símbolo | Faixa Típica | Descrição | Efeito |
|-----------|---------|--------------|-----------|--------|
| **Tamanho da População** | NP | 5D a 10D | Número de indivíduos | Maior = mais exploração |
| **Fator de Escala** | F | 0.5 a 1.0 | Controla magnitude da mutação | Maior = mais exploração |
| **Taxa de Crossover** | CR | 0.7 a 0.9 | Probabilidade de herdar gene do mutante | Maior = mais mudanças |

**Legenda:** D = dimensionalidade do problema

### **5.2 Guia de Configuração**

#### **🎯 Para Problemas Unimodais (Um Ótimo)**
```python
NP = 5 * D        # População pequena
F = 0.5           # Convergência rápida
CR = 0.9          # Alta recombinação
strategy = 'best/1'  # Exploração direcionada
```

#### **🌋 Para Problemas Multimodais (Múltiplos Ótimos)**
```python
NP = 10 * D       # População maior
F = 0.8           # Mais exploração
CR = 0.9          # Alta recombinação
strategy = 'rand/1' ou 'current-to-best/1'
```

#### **📈 Para Alta Dimensionalidade (D > 50)**
```python
NP = 10 * D       # População proporcional
F = 0.9           # Passos grandes
CR = 0.1 a 0.3    # Crossover baixo
strategy = 'rand/2'  # Mais diversidade
```

#### **🎲 Para Funções Ruidosas**
```python
NP = 15 * D       # População muito grande
F = 0.5           # Mutações moderadas
CR = 0.9          # Alta recombinação
# Usar múltiplas avaliações e média
```

### **5.3 Auto-adaptação de Parâmetros**

```python
class SelfAdaptiveDE(DifferentialEvolution):
    """DE com auto-adaptação de F e CR"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parâmetros individuais para cada membro da população
        self.F_values = np.random.uniform(0.5, 1.0, self.NP)
        self.CR_values = np.random.uniform(0.0, 1.0, self.NP)
    
    def adapt_parameters(self, i):
        """Auto-adapta F e CR com probabilidade τ"""
        tau1, tau2 = 0.1, 0.1
        
        if np.random.rand() < tau1:
            self.F_values[i] = 0.1 + 0.9 * np.random.rand()
        
        if np.random.rand() < tau2:
            self.CR_values[i] = np.random.rand()
    
    def optimize(self):
        """Otimização com parâmetros auto-adaptativos"""
        population = self.initialize_population()
        fitness = np.array([self.f(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = {'best_fitness': [best_fitness]}
        
        for generation in range(self.max_iter):
            for i in range(self.NP):
                # Adaptar parâmetros
                self.adapt_parameters(i)
                
                # Usar parâmetros específicos do indivíduo
                self.F = self.F_values[i]
                self.CR = self.CR_values[i]
                
                # Continuar com DE padrão
                mutant = self.mutate(population, i)
                mutant = self.clip_to_bounds(mutant)
                trial = self.crossover(population[i], mutant)
                
                trial_fitness = self.f(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            history['best_fitness'].append(best_fitness)
        
        return best_solution, best_fitness, history
```

---

## **6. ✅ Vantagens e ❌ Desvantagens**

### **6.1 ✅ Vantagens**

| Vantagem | Descrição | Impacto Prático |
|----------|-----------|-----------------|
| **Simplicidade** | Poucos parâmetros para ajustar | Fácil de implementar e usar |
| **Robustez** | Funciona bem em diversos problemas | Não requer conhecimento específico |
| **Sem Gradientes** | Não precisa de derivadas | Funciona com funções não-diferenciáveis |
| **Paralelizável** | Avaliações independentes | Escalável para sistemas distribuídos |
| **Auto-adaptação** | Tamanho de passo ajusta-se automaticamente | Menos ajuste manual |
| **Multimodal** | Lida bem com múltiplos ótimos | Evita ótimos locais |
| **Alta Dimensão** | Eficiente em espaços de alta dimensão | Escalável para problemas complexos |

### **6.2 ❌ Desvantagens**

| Desvantagem | Descrição | Mitigação |
|-------------|-----------|-----------|
| **Convergência Lenta** | Pode ser lento próximo ao ótimo | Usar hibridização com busca local |
| **Sensibilidade a Parâmetros** | Desempenho varia com F, CR, NP | Usar auto-adaptação ou valores padrão |
| **Apenas Contínuo** | Não funciona diretamente em discreto | Adaptar com mapeamento ou arredondamento |
| **Sem Garantias** | Não garante ótimo global | Executar múltiplas vezes |
| **Custo Computacional** | Muitas avaliações de função | Paralelizar ou usar surrogates |

### **6.3 🎯 Quando Usar DE**

#### **✅ Cenários Ideais:**
- ✅ Otimização contínua multidimensional
- ✅ Funções multimodais complexas
- ✅ Não há informação de gradiente
- ✅ Função objetivo é ruidosa
- ✅ Restrições podem ser tratadas por penalização
- ✅ Calibração de modelos e hiperparâmetros
- ✅ Problemas de engenharia e design

#### **❌ Evite DE quando:**
- ❌ Função é unimodal e suave (usar otimização baseada em gradiente)
- ❌ Dimensionalidade é muito baixa (< 3D)
- ❌ Avaliação da função é extremamente custosa
- ❌ Precisa de solução ótima provada
- ❌ Problema é puramente discreto (usar GA)

---

## **7. 🔬 Variantes Avançadas**

### **7.1 jDE (Self-Adaptive DE)**

```python
class jDE(DifferentialEvolution):
    """
    jDE: DE com auto-adaptação de F e CR
    Brest et al. (2006)
    """
    
    def __init__(self, *args, tau1=0.1, tau2=0.1, 
                 F_lower=0.1, F_upper=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau1 = tau1
        self.tau2 = tau2
        self.F_lower = F_lower
        self.F_upper = F_upper
        
        # Parâmetros por indivíduo
        self.F_i = np.full(self.NP, self.F)
        self.CR_i = np.full(self.NP, self.CR)
    
    def adapt_control_parameters(self, i):
        """Adapta F e CR para indivíduo i"""
        if np.random.rand() < self.tau1:
            self.F_i[i] = self.F_lower + np.random.rand() * (
                self.F_upper - self.F_lower
            )
        
        if np.random.rand() < self.tau2:
            self.CR_i[i] = np.random.rand()
        
        return self.F_i[i], self.CR_i[i]
```

### **7.2 SHADE (Success-History Adaptation)**

```python
class SHADE(DifferentialEvolution):
    """
    SHADE: Success-History based Adaptive DE
    Tanabe & Fukunaga (2013)
    """
    
    def __init__(self, *args, H=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.H = H  # Tamanho da memória
        
        # Memória de parâmetros bem-sucedidos
        self.M_F = [0.5] * H
        self.M_CR = [0.5] * H
        self.k = 0
    
    def get_parameters_from_memory(self):
        """Obtém F e CR da memória"""
        r = np.random.randint(0, self.H)
        
        # F usando distribuição Cauchy
        F = np.clip(np.random.standard_cauchy() * 0.1 + self.M_F[r], 0, 1)
        
        # CR usando distribuição Normal
        CR = np.clip(np.random.normal(self.M_CR[r], 0.1), 0, 1)
        
        return F, CR
    
    def update_memory(self, successful_F, successful_CR):
        """Atualiza memória com parâmetros bem-sucedidos"""
        if len(successful_F) > 0:
            # Média ponderada por melhoria de fitness
            mean_F = np.mean(successful_F)
            mean_CR = np.mean(successful_CR)
            
            self.M_F[self.k] = mean_F
            self.M_CR[self.k] = mean_CR
            
            self.k = (self.k + 1) % self.H
```

### **7.3 L-SHADE (Linear Population Size Reduction)**

Combina SHADE com redução linear do tamanho da população:

```python
class LSHADE(SHADE):
    """
    L-SHADE: SHADE com redução linear de população
    """
    
    def __init__(self, *args, N_init=None, N_min=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_init = N_init or self.NP
        self.N_min = N_min
    
    def update_population_size(self, generation):
        """Reduz tamanho da população linearmente"""
        new_size = round(
            self.N_min + (self.N_init - self.N_min) * 
            (1 - generation / self.max_iter)
        )
        return max(new_size, self.N_min)
```

### **7.4 JADE (Adaptive DE with Archive)**

```python
class JADE(DifferentialEvolution):
    """
    JADE: Adaptive DE with Optional External Archive
    Zhang & Sanderson (2009)
    """
    
    def __init__(self, *args, c=0.1, p=0.05, archive_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c  # Taxa de aprendizado
        self.p = p  # Top-p% para current-to-pbest
        self.archive_size = archive_size or self.NP
        self.archive = []
        
        # Parâmetros adaptativos
        self.mu_F = 0.5
        self.mu_CR = 0.5
    
    def mutate_jade(self, population, current_idx, fitness):
        """Mutação current-to-pbest/1 com arquivo"""
        # Selecionar um dos top-p% melhores
        p_best_size = max(1, int(self.p * self.NP))
        top_indices = np.argsort(fitness)[:p_best_size]
        p_best_idx = np.random.choice(top_indices)
        
        # Selecionar r1 da população e r2 da população + arquivo
        candidates = [idx for idx in range(self.NP) if idx != current_idx]
        r1 = np.random.choice(candidates)
        
        combined = list(range(self.NP)) + list(range(len(self.archive)))
        r2 = np.random.choice([idx for idx in combined if idx != current_idx])
        
        if r2 < self.NP:
            x_r2 = population[r2]
        else:
            x_r2 = self.archive[r2 - self.NP]
        
        # Mutação
        mutant = (population[current_idx] + 
                 self.F * (population[p_best_idx] - population[current_idx]) +
                 self.F * (population[r1] - x_r2))
        
        return mutant
    
    def update_archive(self, failed_individual):
        """Adiciona indivíduo substituído ao arquivo"""
        self.archive.append(failed_individual)
        if len(self.archive) > self.archive_size:
            # Remover aleatoriamente
            self.archive.pop(np.random.randint(len(self.archive)))
```

---

## **8. 🎓 Comparações e Benchmarks**

### **8.1 Comparação com Outros Algoritmos**

```python
import numpy as np
from scipy.optimize import minimize

def benchmark_algorithms(func, bounds, dim=10):
    """
    Compara DE com outros métodos de otimização
    """
    results = {}
    
    # 1. Differential Evolution
    de = DifferentialEvolution(func, bounds, pop_size=50, max_iter=100)
    x_de, f_de, _ = de.optimize()
    results['DE'] = {'solution': x_de, 'fitness': f_de}
    
    # 2. Scipy - Nelder-Mead
    x0 = np.random.uniform(bounds[0][0], bounds[0][1], dim)
    res_nm = minimize(func, x0, method='Nelder-Mead', 
                     options={'maxiter': 5000})
    results['Nelder-Mead'] = {'solution': res_nm.x, 'fitness': res_nm.fun}
    
    # 3. Scipy - L-BFGS-B (com gradiente)
    res_lbfgs = minimize(func, x0, method='L-BFGS-B', 
                        bounds=[bounds[0]]*dim, 
                        options={'maxiter': 5000})
    results['L-BFGS-B'] = {'solution': res_lbfgs.x, 'fitness': res_lbfgs.fun}
    
    # 4. Scipy - Differential Evolution
    from scipy.optimize import differential_evolution
    res_scipy_de = differential_evolution(func, [bounds[0]]*dim, 
                                         maxiter=100, popsize=5)
    results['Scipy-DE'] = {'solution': res_scipy_de.x, 'fitness': res_scipy_de.fun}
    
    return results

# Testar em função de Rastrigin
bounds = [(-5.12, 5.12)]
results = benchmark_algorithms(rastrigin, bounds, dim=10)

print("Comparação de Algoritmos na Função de Rastrigin (10D):")
print("-" * 60)
for method, data in results.items():
    print(f"{method:15s}: f = {data['fitness']:.6f}")
```

### **8.2 Análise de Desempenho**

```python
def performance_analysis(func, bounds, dim=10, runs=30):
    """
    Análise estatística de desempenho do DE
    """
    results = []
    
    for run in range(runs):
        de = DifferentialEvolution(
            func, bounds, 
            pop_size=50, 
            F=0.8, 
            CR=0.9, 
            max_iter=200
        )
        _, fitness, _ = de.optimize()
        results.append(fitness)
    
    results = np.array(results)
    
    stats = {
        'mean': np.mean(results),
        'std': np.std(results),
        'median': np.median(results),
        'min': np.min(results),
        'max': np.max(results),
        'q25': np.percentile(results, 25),
        'q75': np.percentile(results, 75)
    }
    
    return stats, results

# Executar análise
stats, results = performance_analysis(rastrigin, [(-5.12, 5.12)], dim=10, runs=30)

print("Estatísticas de Desempenho (30 execuções):")
print(f"Média:    {stats['mean']:.6f}")
print(f"Desvio:   {stats['std']:.6f}")
print(f"Mediana:  {stats['median']:.6f}")
print(f"Mínimo:   {stats['min']:.6f}")
print(f"Máximo:   {stats['max']:.6f}")
print(f"Q25-Q75:  {stats['q25']:.6f} - {stats['q75']:.6f}")
```

---

## **9. 📚 Funções de Benchmark**

### **9.1 Biblioteca de Funções de Teste**

```python
class BenchmarkFunctions:
    """Coleção de funções de benchmark para otimização"""
    
    @staticmethod
    def sphere(x):
        """
        Função Esfera
        Unimodal, separável, convexa
        Mínimo: f(0,...,0) = 0
        """
        return np.sum(x**2)
    
    @staticmethod
    def rastrigin(x):
        """
        Função Rastrigin
        Multimodal, separável
        Mínimo: f(0,...,0) = 0
        """
        n = len(x)
        return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
    @staticmethod
    def rosenbrock(x):
        """
        Função Rosenbrock
        Unimodal, não-separável, vale estreito
        Mínimo: f(1,...,1) = 0
        """
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    @staticmethod
    def ackley(x):
        """
        Função Ackley
        Multimodal, não-separável
        Mínimo: f(0,...,0) = 0
        """
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2*np.pi*x))
        return (-20*np.exp(-0.2*np.sqrt(sum1/n)) - 
                np.exp(sum2/n) + 20 + np.e)
    
    @staticmethod
    def schwefel(x):
        """
        Função Schwefel
        Multimodal, não-separável
        Mínimo: f(420.9687,...,420.9687) = 0
        """
        n = len(x)
        return 418.9829*n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def griewank(x):
        """
        Função Griewank
        Multimodal, não-separável
        Mínimo: f(0,...,0) = 0
        """
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
        return sum_sq - prod_cos + 1
    
    @staticmethod
    def levy(x):
        """
        Função Levy
        Multimodal, não-separável
        Mínimo: f(1,...,1) = 0
        """
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
        return term1 + term2 + term3

# Uso
bench = BenchmarkFunctions()
x = np.zeros(10)
print(f"Sphere(0) = {bench.sphere(x)}")  # Deve ser 0
```

---

## **10. 🔗 Referências e Recursos**

### **10.1 📚 Artigos Fundamentais**

1. **Storn, R., & Price, K. (1997).** *"Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces"*. Journal of Global Optimization, 11(4), 341-359.
   - 🌟 Artigo original que introduziu o DE
   - 📊 Descrição completa do algoritmo

2. **Price, K., Storn, R. M., & Lampinen, J. A. (2005).** *"Differential Evolution: A Practical Approach to Global Optimization"*. Springer.
   - 📖 Livro definitivo sobre DE
   - 🎯 Teoria e aplicações práticas

3. **Das, S., & Suganthan, P. N. (2011).** *"Differential Evolution: A Survey of the State-of-the-Art"*. IEEE Transactions on Evolutionary Computation, 15(1), 4-31.
   - 📊 Survey abrangente sobre variantes
   - 🔬 Análise teórica e experimental

### **10.2 🌐 Recursos Online**

| Recurso | Tipo | Descrição | URL |
|---------|------|-----------|-----|
| **DE Homepage** | Site Oficial | Site original dos criadores | www1.icsi.berkeley.edu/~storn/code.html |
| **scipy.optimize.differential_evolution** | Biblioteca | Implementação em SciPy | docs.scipy.org |
| **PyGMO** | Framework | Otimização global multi-objetivo | esa.github.io/pygmo2 |
| **DEAP** | Biblioteca | Framework de algoritmos evolutivos | deap.readthedocs.io |

### **10.3 🛠️ Implementações Disponíveis**

```python
# 1. SciPy (mais comum)
from scipy.optimize import differential_evolution

# 2. PyGMO (otimização espacial)
import pygmo as pg

# 3. DEAP (framework completo)
from deap import algorithms, base, creator, tools

# 4. Pymoo (multi-objetivo)
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize

# 5. NiaPy (natureza inspirada)
from niapy.algorithms.basic import DifferentialEvolution
```

### **10.4 📝 Artigos sobre Variantes Avançadas**

1. **jDE**: Brest et al. (2006) - "Self-Adapting Control Parameters in Differential Evolution"
2. **JADE**: Zhang & Sanderson (2009) - "JADE: Adaptive Differential Evolution with Optional External Archive"
3. **SHADE**: Tanabe & Fukunaga (2013) - "Success-History Based Parameter Adaptation for DE"
4. **L-SHADE**: Tanabe & Fukunaga (2014) - "Improving the Search Performance of SHADE"
5. **CoBiDE**: Wang et al. (2011) - "Composite Differential Evolution for Constrained Evolutionary Optimization"

### **10.5 🎓 Tutoriais e Cursos**

- **Coursera:** Evolutionary Computation
- **MIT OpenCourseWare:** Computational Evolutionary Biology
- **YouTube:** Lectures on Differential Evolution
- **Kaggle:** DE for Hyperparameter Tuning

---

## **11. 🎯 Conclusão**

A Evolução Diferencial é um dos algoritmos de otimização mais **versáteis e eficientes** disponíveis para problemas contínuos. Suas principais características são:

### **🔑 Principais Aprendizados**

1. **Simplicidade Elegante:** Poucos parâmetros, implementação direta, resultados sólidos
2. **Mutação Diferencial:** Uso inteligente de diferenças vetoriais para auto-adaptação
3. **Robustez:** Funciona bem em diversos tipos de problemas sem ajuste fino
4. **Flexibilidade:** Múltiplas variantes para diferentes cenários

### **💡 Quando Usar DE**

| ✅ **Use quando:** | ❌ **Evite quando:** |
|-------------------|---------------------|
| Otimização contínua multimodal | Função é unimodal e diferenciável |
| Não há gradiente disponível | Avaliação é extremamente custosa |
| Função é ruidosa ou não-suave | Precisa de garantias teóricas |
| Alta dimensionalidade (< 100D) | Problema é puramente discreto |
| Calibração de modelos | Baixa dimensionalidade (< 3D) |

### **🚀 Próximos Passos**

1. **Implemente** a versão básica do DE
2. **Experimente** diferentes estratégias de mutação
3. **Teste** em funções de benchmark
4. **Aplique** ao seu problema específico
5. **Explore** variantes avançadas (SHADE, L-SHADE)
6. **Compare** com outros métodos de otimização
7. **Considere** hibridização com busca local

### **🌟 Reflexão Final**

A Evolução Diferencial demonstra que **simplicidade e eficácia** podem andar juntas. Ao usar diferenças entre vetores da população, o algoritmo captura implicitamente a geometria do espaço de busca, criando um mecanismo de busca naturalmente adaptativo e robusto.

> *"A beleza da Evolução Diferencial está em usar a sabedoria coletiva da população - as diferenças entre indivíduos guiam a busca de forma inteligente e auto-organizante."*

---

**🔗 Continue sua jornada:**
- 📖 Explore [**Evolution Strategies**](evolution_strategies.md) para auto-adaptação avançada
- 🧬 Volte para [**Genetic Algorithms**](genetic_algorithms.md) para comparação
- 🎯 Veja [**Algoritmos Evolucionários**](README.md) para visão geral
- 🔄 Investigue hibridização com busca local para melhor performance

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

# Estratégias de Evolução (Evolution Strategies - ES)

As **Estratégias de Evolução** (Evolution Strategies - ES) são uma família de algoritmos evolutivos desenvolvidos na Alemanha nos anos 1960 por Ingo Rechenberg e Hans-Paul Schwefel. Diferentemente dos Algoritmos Genéticos, as ES foram projetadas especificamente para otimização de parâmetros contínuos e são conhecidas por sua capacidade de **auto-adaptação** de parâmetros de controle.

![Evolution Strategies Concept](../../images/evolution_strategies_concept.png)

As ES são particularmente eficazes em problemas de otimização numérica complexa, ruidosa e de alta dimensão, sendo amplamente utilizadas em robótica, engenharia, reinforcement learning e design de sistemas complexos.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 Conceito Central**

As Estratégias de Evolução se baseiam em princípios fundamentais:

1. **Representação Real:** Trabalha diretamente com valores reais (não binários)
2. **Auto-adaptação:** Parâmetros de mutação evoluem junto com a solução
3. **Seleção Determinística:** Baseada apenas em ranking de fitness
4. **Mutação Gaussiana:** Principal operador de variação

**Intuição:**
> "Assim como na natureza, não apenas as soluções evoluem, mas também a forma como elas se transformam - a estratégia de evolução em si evolui."

### **1.2 Notação das ES**

As ES são comumente descritas pela notação **(μ/ρ +/,  λ)-ES:**

- **μ (mi):** Número de pais selecionados para reprodução
- **ρ (rho):** Número de pais que contribuem para criar um filho
- **λ (lambda):** Número de filhos gerados
- **+:** Seleção geracional (pais + filhos competem)
- **,:** Seleção não-geracional (apenas filhos competem)

#### **Exemplos Comuns:**

```
(1+1)-ES: 1 pai, 1 filho, melhor sobrevive
(μ+λ)-ES: μ pais, λ filhos, melhores μ sobrevivem (elitista)
(μ,λ)-ES: μ pais, λ filhos, melhores μ entre filhos sobrevivem (não-elitista)
          Requer λ ≥ μ
```

### **1.3 Diferenças dos Algoritmos Genéticos**

| Aspecto | Algoritmos Genéticos | Evolution Strategies |
|---------|---------------------|---------------------|
| **Origem** | EUA (John Holland) | Alemanha (Rechenberg, Schwefel) |
| **Representação** | Binária/Inteira | Real/Contínua |
| **Mutação** | Secundária | Primária |
| **Crossover** | Primário | Secundário/Opcional |
| **Seleção** | Probabilística (roleta, torneio) | Determinística (ranking) |
| **Auto-adaptação** | Rara | Fundamental |
| **Aplicação** | Problemas combinatórios | Otimização numérica |

---

## **2. 🔧 Algoritmo das Estratégias de Evolução**

### **2.1 (1+1)-ES: A ES Mais Simples**

A forma mais básica com um pai e um filho:

```
🚀 1. INICIALIZAÇÃO
   ├── x ← solução inicial aleatória
   ├── σ ← step-size inicial (desvio padrão)
   └── Avaliar f(x)

🔄 2. LOOP EVOLUTIVO (para cada geração):
   │
   ├── 🧬 MUTAÇÃO
   │   ├── x' ← x + N(0, σ²I)    # I = matriz identidade
   │   └── Avaliar f(x')
   │
   ├── 🎯 SELEÇÃO
   │   SE f(x') ≤ f(x):  # Para minimização
   │       x ← x'        # Aceita filho
   │   SENÃO:
   │       manter x      # Mantém pai
   │
   └── 🔧 ADAPTAR STEP-SIZE (Regra 1/5)
       ├── Contar sucessos nas últimas n gerações
       ├── SE taxa_sucesso > 1/5:
       │       σ ← σ / c    # Aumentar σ (c < 1)
       ├── SE taxa_sucesso < 1/5:
       │       σ ← σ * c    # Diminuir σ
       └── c ≈ 0.82 (típico)

🏆 3. RETORNAR melhor solução
```

**Regra 1/5 de Rechenberg:**
> "Para convergência ótima, a taxa de mutações bem-sucedidas deve ser aproximadamente 1/5."

### **2.2 (μ+λ)-ES: Estratégia Geracional**

```
🚀 1. INICIALIZAÇÃO
   ├── Gerar população P de μ indivíduos
   │   Cada indivíduo: (x, σ) onde x = solução, σ = step-sizes
   └── Avaliar todos os indivíduos

🔄 2. LOOP EVOLUTIVO:
   │
   ├── 🧬 CRIAR λ FILHOS
   │   PARA i = 1 até λ:
   │       ├── Selecionar ρ pais aleatoriamente
   │       ├── Recombinação (opcional):
   │       │   x_filho ← recombinar(pais)
   │       │   σ_filho ← recombinar(σ_pais)
   │       ├── Mutação:
   │       │   σ'_filho ← σ_filho * exp(τ·N(0,1) + τ'·N_i(0,1))
   │       │   x'_filho ← x_filho + N(0, (σ'_filho)²I)
   │       └── Avaliar f(x'_filho)
   │
   ├── 🎯 SELEÇÃO (μ+λ)
   │   ├── Combinar pais P e filhos C
   │   ├── Ordenar por fitness
   │   └── Selecionar melhores μ para próxima geração
   │
   └── Incrementar geração

🏆 3. RETORNAR melhor indivíduo
```

### **2.3 (μ,λ)-ES: Estratégia Não-Geracional**

Diferença principal: apenas filhos competem (λ ≥ μ obrigatório)

```
🎯 SELEÇÃO (μ,λ)
   ├── Avaliar apenas os λ filhos
   ├── Ordenar filhos por fitness
   ├── Selecionar melhores μ filhos
   └── Pais são descartados
```

**Vantagens de (μ,λ):**
- ❌ Não-elitista: perde melhor solução temporariamente
- ✅ Melhor em ambientes ruidosos
- ✅ Evita convergência prematura
- ✅ Permite mudanças no landscape

---

## **3. 🔧 Auto-adaptação de Parâmetros**

### **3.1 Auto-adaptação de Step-sizes**

Um dos recursos mais poderosos das ES é a auto-adaptação:

```python
# Auto-adaptação de um único σ (isotropic)
σ' = σ * exp(τ * N(0, 1))
x' = x + N(0, (σ')²·I)

# Parâmetro de aprendizado típico
τ = 1 / sqrt(2·n)  # n = dimensionalidade
```

**Variantes:**

#### **1. Isotropic (1 step-size para todas as dimensões)**
```python
σ' = σ * exp(τ · N(0,1))
```
- ✅ Simples
- ❌ Não captura diferentes escalas

#### **2. Individual Step-sizes (σ por dimensão)**
```python
σ'_i = σ_i * exp(τ' · N(0,1) + τ · N_i(0,1))
x'_i = x_i + σ'_i · N_i(0,1)

τ' = 1 / sqrt(2·n)    # Aprendizado global
τ = 1 / sqrt(2·sqrt(n))  # Aprendizado local
```
- ✅ Captura escalas diferentes
- ⚪ Mais parâmetros

#### **3. Correlated Mutations (com rotações)**
```python
# Inclui correlações entre dimensões
# Usa matriz de covariância completa
# Complexo, mas mais poderoso
```

### **3.2 CMA-ES (Covariance Matrix Adaptation)**

A variante mais avançada, considerada estado-da-arte:

**Ideia:** Adaptar uma matriz de covariância completa para capturar:
- Escalas diferentes nas dimensões
- Correlações/rotações no espaço de busca
- Direção de busca promissora

```
C ← matriz de covariância (n×n)
σ ← step-size global
m ← média da distribuição (centróide)

Em cada geração:
1. Gerar λ amostras: x_i ~ N(m, σ²C)
2. Selecionar μ melhores
3. Atualizar m (média dos melhores)
4. Atualizar C (direções promissoras)
5. Atualizar σ (controle de convergência)
```

---

## **4. 💻 Implementação em Python**

### **4.1 (1+1)-ES Básica**

```python
import numpy as np

class OnePlusOneES:
    """
    Implementação de (1+1)-ES com regra 1/5
    """
    
    def __init__(self, objective_function, bounds, 
                 sigma_init=1.0, max_iter=1000):
        """
        Args:
            objective_function: Função a minimizar
            bounds: Lista de (min, max) para cada dimensão
            sigma_init: Step-size inicial
            max_iter: Número máximo de gerações
        """
        self.f = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.sigma = sigma_init
        self.max_iter = max_iter
        
        # Para regra 1/5
        self.n_window = int(self.dim * 10)  # Janela de observação
        self.success_history = []
        self.c = 0.82  # Fator de ajuste
    
    def initialize(self):
        """Inicializa solução aleatória"""
        x = np.random.rand(self.dim)
        for i in range(self.dim):
            x[i] = self.bounds[i, 0] + x[i] * (
                self.bounds[i, 1] - self.bounds[i, 0]
            )
        return x
    
    def mutate(self, x):
        """Aplica mutação gaussiana"""
        mutation = np.random.normal(0, self.sigma, self.dim)
        x_new = x + mutation
        # Garantir limites
        x_new = np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])
        return x_new
    
    def adapt_stepsize(self):
        """Adapta step-size usando regra 1/5"""
        if len(self.success_history) >= self.n_window:
            # Calcular taxa de sucesso
            success_rate = np.mean(self.success_history[-self.n_window:])
            
            if success_rate > 0.2:  # 1/5 = 0.2
                self.sigma = self.sigma / self.c  # Aumentar
            elif success_rate < 0.2:
                self.sigma = self.sigma * self.c  # Diminuir
    
    def optimize(self):
        """Executa otimização"""
        # Inicialização
        x = self.initialize()
        fitness = self.f(x)
        best_x = x.copy()
        best_fitness = fitness
        
        history = {
            'best_fitness': [best_fitness],
            'sigma': [self.sigma]
        }
        
        # Loop evolutivo
        for generation in range(self.max_iter):
            # Mutação
            x_new = self.mutate(x)
            fitness_new = self.f(x_new)
            
            # Seleção
            if fitness_new <= fitness:
                x = x_new
                fitness = fitness_new
                self.success_history.append(1)
                
                if fitness < best_fitness:
                    best_x = x.copy()
                    best_fitness = fitness
            else:
                self.success_history.append(0)
            
            # Adaptar step-size
            self.adapt_stepsize()
            
            # Registrar histórico
            history['best_fitness'].append(best_fitness)
            history['sigma'].append(self.sigma)
        
        return best_x, best_fitness, history

# Exemplo de uso
def sphere(x):
    return np.sum(x**2)

bounds = [(-5, 5)] * 10
es = OnePlusOneES(sphere, bounds, sigma_init=1.0, max_iter=500)
solution, fitness, history = es.optimize()

print(f"Solução: {solution}")
print(f"Fitness: {fitness:.8f}")
print(f"Sigma final: {history['sigma'][-1]:.6f}")
```

### **4.2 (μ+λ)-ES com Auto-adaptação**

```python
class MuPlusLambdaES:
    """
    (μ+λ)-ES com auto-adaptação individual de step-sizes
    """
    
    def __init__(self, objective_function, bounds, 
                 mu=15, lambda_=100, rho=None, max_iter=500):
        """
        Args:
            mu: Número de pais
            lambda_: Número de filhos (deve ser >= mu)
            rho: Número de pais para recombinação (None = mu)
        """
        self.f = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.mu = mu
        self.lambda_ = lambda_
        self.rho = rho or mu
        self.max_iter = max_iter
        
        # Parâmetros de auto-adaptação
        self.tau = 1.0 / np.sqrt(2 * self.dim)
        self.tau_prime = 1.0 / np.sqrt(2 * np.sqrt(self.dim))
    
    def initialize_population(self):
        """Inicializa população com (x, σ)"""
        population = []
        for _ in range(self.mu):
            # Solução
            x = np.random.rand(self.dim)
            for i in range(self.dim):
                x[i] = self.bounds[i, 0] + x[i] * (
                    self.bounds[i, 1] - self.bounds[i, 0]
                )
            
            # Step-sizes individuais
            sigma = np.ones(self.dim) * 0.5
            
            population.append({
                'x': x,
                'sigma': sigma,
                'fitness': self.f(x)
            })
        
        return population
    
    def recombination_intermediate(self, parents, key):
        """Recombinação intermediária (média)"""
        return np.mean([p[key] for p in parents], axis=0)
    
    def mutate(self, parent):
        """Mutação com auto-adaptação"""
        # Auto-adaptar step-sizes
        global_factor = self.tau_prime * np.random.normal()
        individual_factors = self.tau * np.random.normal(size=self.dim)
        
        sigma_new = parent['sigma'] * np.exp(
            global_factor + individual_factors
        )
        
        # Garantir σ mínimo
        sigma_new = np.maximum(sigma_new, 1e-10)
        
        # Mutar solução
        x_new = parent['x'] + sigma_new * np.random.normal(size=self.dim)
        
        # Garantir limites
        x_new = np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])
        
        return {
            'x': x_new,
            'sigma': sigma_new,
            'fitness': self.f(x_new)
        }
    
    def select_parents(self, population, n):
        """Seleciona n pais aleatoriamente"""
        return [population[i] for i in np.random.choice(
            len(population), n, replace=False
        )]
    
    def optimize(self):
        """Executa (μ+λ)-ES"""
        # Inicialização
        population = self.initialize_population()
        
        # Melhor solução
        population.sort(key=lambda ind: ind['fitness'])
        best = population[0].copy()
        
        history = {
            'best_fitness': [best['fitness']],
            'avg_fitness': [np.mean([ind['fitness'] for ind in population])],
            'avg_sigma': [np.mean([np.mean(ind['sigma']) for ind in population])]
        }
        
        # Loop evolutivo
        for generation in range(self.max_iter):
            offspring = []
            
            # Gerar λ filhos
            for _ in range(self.lambda_):
                # Selecionar ρ pais
                parents = self.select_parents(population, self.rho)
                
                # Recombinação
                child = {
                    'x': self.recombination_intermediate(parents, 'x'),
                    'sigma': self.recombination_intermediate(parents, 'sigma')
                }
                
                # Mutação
                child = self.mutate(child)
                offspring.append(child)
            
            # Seleção (μ+λ): combinar pais e filhos
            combined = population + offspring
            combined.sort(key=lambda ind: ind['fitness'])
            
            # Selecionar melhores μ
            population = combined[:self.mu]
            
            # Atualizar melhor
            if population[0]['fitness'] < best['fitness']:
                best = population[0].copy()
            
            # Registrar histórico
            history['best_fitness'].append(best['fitness'])
            history['avg_fitness'].append(
                np.mean([ind['fitness'] for ind in population])
            )
            history['avg_sigma'].append(
                np.mean([np.mean(ind['sigma']) for ind in population])
            )
        
        return best['x'], best['fitness'], history

# Exemplo de uso
def rastrigin(x):
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

bounds = [(-5.12, 5.12)] * 10
es = MuPlusLambdaES(
    rastrigin, 
    bounds, 
    mu=15, 
    lambda_=100, 
    rho=7,
    max_iter=200
)

solution, fitness, history = es.optimize()
print(f"Melhor fitness: {fitness:.6f}")
```

### **4.3 CMA-ES Simplificada**

```python
class SimpleCMAES:
    """
    Implementação simplificada de CMA-ES
    Para produção, use biblioteca pycma
    """
    
    def __init__(self, objective_function, bounds, 
                 pop_size=None, sigma_init=0.5, max_iter=500):
        self.f = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.sigma = sigma_init
        self.max_iter = max_iter
        
        # Tamanho da população
        self.lambda_ = pop_size or (4 + int(3 * np.log(self.dim)))
        self.mu = self.lambda_ // 2
        
        # Pesos para recombinação
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = self.weights / np.sum(self.weights)
        
        # Média inicial
        self.mean = np.random.rand(self.dim)
        for i in range(self.dim):
            self.mean[i] = self.bounds[i, 0] + self.mean[i] * (
                self.bounds[i, 1] - self.bounds[i, 0]
            )
        
        # Matriz de covariância
        self.C = np.eye(self.dim)
        
        # Caminhos de evolução
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        
        # Parâmetros de aprendizado
        self.cc = 4 / (self.dim + 4)
        self.cs = 4 / (self.dim + 4)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu)
        self.cmu = min(1 - self.c1, 2 * (self.mu - 2 + 1/self.mu) / 
                      ((self.dim + 2)**2 + self.mu))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu - 1)/(self.dim + 1)) - 1) + self.cs
    
    def optimize(self):
        """Executa CMA-ES"""
        history = {'best_fitness': []}
        best_fitness = np.inf
        best_solution = None
        
        for generation in range(self.max_iter):
            # Gerar população
            population = []
            for _ in range(self.lambda_):
                # Amostra da distribuição normal multivariada
                z = np.random.normal(0, 1, self.dim)
                y = np.dot(np.linalg.cholesky(self.C), z)
                x = self.mean + self.sigma * y
                
                # Garantir limites
                x = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
                
                fitness = self.f(x)
                population.append((fitness, x, y))
            
            # Ordenar por fitness
            population.sort(key=lambda item: item[0])
            
            # Atualizar melhor
            if population[0][0] < best_fitness:
                best_fitness = population[0][0]
                best_solution = population[0][1].copy()
            
            history['best_fitness'].append(best_fitness)
            
            # Selecionar μ melhores
            selected = population[:self.mu]
            
            # Atualizar média (recombinação ponderada)
            old_mean = self.mean.copy()
            self.mean = np.sum([
                self.weights[i] * selected[i][1] 
                for i in range(self.mu)
            ], axis=0)
            
            # Atualizar caminhos de evolução e covariância
            # (simplificado - versão completa é mais complexa)
            
        return best_solution, best_fitness, history

# Para uso em produção, recomenda-se:
# pip install cma
# import cma
# es = cma.CMAEvolutionStrategy(x0, sigma0)
# es.optimize(objective_function)
```

---

## **5. 🎯 Exemplos de Aplicação**

### **5.1 Controle de Robô**

```python
def robot_control_optimization():
    """
    Otimiza parâmetros de controlador de robô
    """
    def simulate_robot(params):
        """
        Simula robô com parâmetros de controle
        Retorna erro acumulado (a minimizar)
        """
        # params = [kp, ki, kd, ...] (parâmetros PID, etc.)
        
        # Simulação simplificada
        error_total = 0
        state = 0
        target = 10
        
        for t in range(100):
            error = target - state
            control = params[0] * error  # PID simplificado
            state += control * 0.1
            error_total += abs(error)
        
        return error_total
    
    # Otimizar com ES
    bounds = [(0, 10)] * 3  # kp, ki, kd
    es = MuPlusLambdaES(
        simulate_robot,
        bounds,
        mu=10,
        lambda_=70,
        max_iter=100
    )
    
    best_params, best_error, history = es.optimize()
    
    print(f"Melhores parâmetros: {best_params}")
    print(f"Erro final: {best_error:.4f}")
    
    return best_params

# Executar
robot_params = robot_control_optimization()
```

### **5.2 Evolution Strategies para Reinforcement Learning**

```python
def es_for_rl():
    """
    Usa ES para treinar política de RL
    Baseado em "Evolution Strategies as a Scalable Alternative to RL"
    (Salimans et al., OpenAI, 2017)
    """
    class NeuralPolicy:
        """Política neural simples"""
        def __init__(self, input_dim, hidden_dim, output_dim):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            
            # Número total de parâmetros
            self.n_params = (input_dim * hidden_dim + hidden_dim +
                           hidden_dim * output_dim + output_dim)
        
        def set_params(self, params):
            """Define pesos da rede"""
            self.params = params
        
        def forward(self, state):
            """Forward pass simplificado"""
            # Implementação simplificada
            # Na prática, seria uma rede neural completa
            idx = 0
            # W1: input_dim x hidden_dim
            W1_size = self.input_dim * self.hidden_dim
            W1 = params[idx:idx+W1_size].reshape(self.hidden_dim, self.input_dim)
            idx += W1_size
            # ... resto dos parâmetros
            
            # Computar ação
            action = np.tanh(W1 @ state)  # Simplificado
            return action
    
    def evaluate_policy(params, n_episodes=5):
        """
        Avalia política em ambiente
        Retorna recompensa total (negativa para minimização)
        """
        policy = NeuralPolicy(input_dim=4, hidden_dim=8, output_dim=2)
        policy.set_params(params)
        
        total_reward = 0
        for episode in range(n_episodes):
            # Simulação de episódio
            state = np.random.randn(4)
            episode_reward = 0
            
            for step in range(100):
                action = policy.forward(state)
                # Simular ambiente
                reward = -np.sum(state**2)  # Exemplo simplificado
                episode_reward += reward
                # Atualizar estado (simplificado)
                state = state + 0.1 * action[:len(state)]
            
            total_reward += episode_reward
        
        return -total_reward / n_episodes  # Negativo para minimização
    
    # Configurar ES
    policy = NeuralPolicy(input_dim=4, hidden_dim=8, output_dim=2)
    n_params = policy.n_params
    
    # Bounds amplos para pesos neurais
    bounds = [(-2, 2)] * n_params
    
    es = MuPlusLambdaES(
        evaluate_policy,
        bounds,
        mu=20,
        lambda_=100,
        max_iter=50
    )
    
    best_params, best_reward, history = es.optimize()
    
    print(f"Melhor recompensa: {-best_reward:.2f}")
    
    return best_params

# Nota: Para uso real em RL, veja:
# - OpenAI ES: https://github.com/openai/evolution-strategies-starter
# - PyTorch ES: estorch
```

---

## **6. ⚙️ Configuração e Diretrizes**

### **6.1 Escolha de μ e λ**

| Estratégia | μ | λ | Relação | Uso |
|-----------|---|---|---------|-----|
| **(1+1)-ES** | 1 | 1 | - | Problemas simples, rápido |
| **(1+λ)-ES** | 1 | 10-100 | λ >> 1 | Exploração ampla |
| **(μ+λ)-ES** | 10-30% de λ | ~5μ | λ ≥ μ | Balanceado, elitista |
| **(μ,λ)-ES** | ~1/7 de λ | 7μ | λ ≥ 7μ | Ambientes ruidosos |
| **CMA-ES** | λ/2 | 4+⌊3ln(n)⌋ | μ = λ/2 | Estado-da-arte |

**Regras Gerais:**
- **μ pequeno:** Convergência rápida, menos diversidade
- **μ grande:** Mais robusto, convergência lenta
- **λ grande:** Mais exploração, mais avaliações
- **Razão λ/μ:** Tipicamente 5-7

### **6.2 Step-size Inicial (σ)**

```python
# Regras de bolso
σ_init = (bounds_max - bounds_min) / 3  # Cobre ~99% da faixa
σ_init = 0.3  # Para problema normalizado [0,1]
σ_init = 1.0  # Para problema centrado em 0

# Ajuste adaptativo
# ES ajusta automaticamente, então valor inicial não é crítico
```

### **6.3 Quando Usar Cada Variante**

#### **(1+1)-ES**
```
✅ Usar quando:
- Problema simples/unimodal
- Recursos limitados
- Prototipagem rápida
- Baseline para comparação
```

#### **(μ+λ)-ES**
```
✅ Usar quando:
- Precisa de elitismo
- Convergência estável importante
- Ambiente determinístico
- Problemas multimodais
```

#### **(μ,λ)-ES**
```
✅ Usar quando:
- Função objetivo ruidosa
- Landscape dinâmico
- Evitar convergência prematura
- λ >> μ disponível
```

#### **CMA-ES**
```
✅ Usar quando:
- Problema complexo/multimodal
- Alta dimensionalidade
- Precisa do melhor desempenho
- Recursos computacionais disponíveis
- Função tem correlações entre variáveis
```

---

## **7. ✅ Vantagens e ❌ Desvantagens**

### **7.1 ✅ Vantagens**

| Vantagem | Descrição | Benefício |
|----------|-----------|-----------|
| **Auto-adaptação** | Parâmetros evoluem automaticamente | Menos ajuste manual |
| **Robustez a Ruído** | (μ,λ) lida bem com avaliações ruidosas | Funciona em ambientes reais |
| **Sem Gradiente** | Não precisa de derivadas | Funções black-box |
| **Espaço Contínuo** | Projetado para valores reais | Ideal para otimização numérica |
| **Teoria Sólida** | Base matemática forte | Entendimento profundo |
| **Paralelizável** | Avaliações independentes | Escalável |
| **CMA-ES** | Estado-da-arte em otimização | Melhor performance |

### **7.2 ❌ Desvantagens**

| Desvantagem | Descrição | Mitigação |
|-------------|-----------|-----------|
| **Custo Computacional** | Muitas avaliações necessárias | Paralelizar, usar surrogates |
| **Apenas Contínuo** | Não funciona em discreto | Usar GA para discreto |
| **Convergência Lenta** | Pode ser lento | Usar CMA-ES ou hibridizar |
| **Dimensionalidade** | CMA-ES escala O(n²) em memória | Usar variantes para alta dimensão |
| **Complexidade** | CMA-ES é complexo | Usar bibliotecas prontas |

### **7.3 🎯 Quando Usar ES**

#### **✅ Cenários Ideais:**
- ✅ Otimização numérica contínua
- ✅ Funções ruidosas ou estocásticas
- ✅ Não há gradiente disponível
- ✅ Problemas multimodais
- ✅ Reinforcement Learning (treinar políticas)
- ✅ Controle e robótica
- ✅ Engenharia (design, calibração)
- ✅ Alta dimensionalidade (CMA-ES até ~100D)

#### **❌ Evite ES quando:**
- ❌ Função é convexa e suave (usar otimização baseada em gradiente)
- ❌ Problema é discreto (usar GA)
- ❌ Avaliação é extremamente cara
- ❌ Dimensionalidade é muito alta (> 1000D)
- ❌ Precisa de garantias teóricas

---

## **8. 🔬 Variantes Avançadas**

### **8.1 Natural Evolution Strategies (NES)**

```python
"""
NES usa gradiente natural da distribuição de busca
Mais eficiente que ES padrão em alguns casos
"""

class NaturalES:
    """
    Implementação simplificada de NES
    """
    def __init__(self, objective_function, dim, pop_size=50):
        self.f = objective_function
        self.dim = dim
        self.pop_size = pop_size
        
        # Parâmetros da distribuição
        self.mu = np.zeros(dim)
        self.sigma = 1.0
        
        # Learning rates
        self.lr_mu = 0.1
        self.lr_sigma = 0.05
    
    def optimize(self, max_iter=500):
        history = {'best_fitness': []}
        
        for generation in range(max_iter):
            # Gerar população
            noise = np.random.randn(self.pop_size, self.dim)
            population = self.mu + self.sigma * noise
            
            # Avaliar
            fitness = np.array([self.f(ind) for ind in population])
            
            # Normalizar fitness (utilities)
            utilities = compute_centered_ranks(fitness)
            
            # Gradiente natural
            grad_mu = (1.0 / (self.pop_size * self.sigma)) * np.dot(
                noise.T, utilities
            )
            grad_sigma = (1.0 / (self.pop_size * self.sigma)) * np.dot(
                (noise**2 - 1).T, utilities
            ).mean()
            
            # Atualizar parâmetros
            self.mu += self.lr_mu * grad_mu
            self.sigma *= np.exp(0.5 * self.lr_sigma * grad_sigma)
            
            history['best_fitness'].append(np.min(fitness))
        
        return self.mu, history

def compute_centered_ranks(fitness):
    """Calcula utilities baseados em ranking"""
    ranks = np.argsort(np.argsort(fitness))
    utilities = ranks / (len(ranks) - 1) - 0.5
    return utilities
```

### **8.2 Separable CMA-ES (Sep-CMA-ES)**

Para alta dimensionalidade, usa apenas diagonal da covariância:

```python
"""
Sep-CMA-ES: Matriz diagonal ao invés de completa
Reduz complexidade de O(n²) para O(n)
Funciona bem em funções separáveis
"""

class SepCMAES:
    """CMA-ES com covariância diagonal"""
    def __init__(self, objective_function, dim, pop_size=None):
        self.f = objective_function
        self.dim = dim
        self.lambda_ = pop_size or (4 + int(3 * np.log(dim)))
        
        # Apenas diagonal
        self.sigma = np.ones(dim)  # Step-sizes por dimensão
        self.mean = np.zeros(dim)
        
    # Implementação similar a CMA-ES mas com diagonal
```

### **8.3 OpenAI Evolution Strategies**

Versão escalável para deep learning:

```python
"""
OpenAI ES: Paralelização massiva
Usa "virtual batch normalization" via seeds
Escala para milhares de cores
"""

class OpenAIES:
    """
    Versão simplificada do OpenAI ES
    """
    def __init__(self, objective_function, dim, 
                 pop_size=1000, learning_rate=0.01):
        self.f = objective_function
        self.dim = dim
        self.pop_size = pop_size
        self.lr = learning_rate
        self.theta = np.zeros(dim)  # Parâmetros
        self.sigma = 0.1
    
    def optimize_parallel(self, max_iter=100):
        """
        Otimização paralela
        Na prática, usar multiprocessing ou distributed computing
        """
        for generation in range(max_iter):
            # Gerar seeds
            seeds = np.random.randint(0, 2**32, self.pop_size)
            
            # Avaliar em paralelo (simplificado aqui)
            rewards = []
            for seed in seeds:
                np.random.seed(seed)
                epsilon = np.random.randn(self.dim)
                reward = self.f(self.theta + self.sigma * epsilon)
                rewards.append((reward, epsilon, seed))
            
            # Atualizar usando gradient estimator
            gradient = np.zeros(self.dim)
            for reward, epsilon, seed in rewards:
                gradient += reward * epsilon
            
            gradient /= (self.pop_size * self.sigma)
            
            # Gradient ascent (assumindo maximização)
            self.theta += self.lr * gradient
        
        return self.theta

# Para uso real:
# - Usar Ray ou MPI para paralelização
# - Ver: https://github.com/openai/evolution-strategies-starter
```

---

## **9. 📚 Aplicações Práticas**

### **9.1 Design de Aeronaves**

```python
def aircraft_design_optimization():
    """
    Otimiza parâmetros de design de aeronave
    """
    def evaluate_aircraft(params):
        """
        params: [wingspan, chord, thickness, sweep, ...]
        Retorna: custo (peso + drag + constraints)
        """
        # Simulação CFD ou modelo analítico
        weight = params[0] * params[1] * 10  # Simplificado
        drag = params[2]**2 + params[3]**2
        
        # Restrições (penalidades)
        penalty = 0
        if params[0] < 10:  # wingspan mínimo
            penalty += 1000
        
        return weight + drag + penalty
    
    # Bounds: [wingspan, chord, thickness, sweep]
    bounds = [(10, 30), (1, 5), (0.1, 0.5), (0, 45)]
    
    es = MuPlusLambdaES(
        evaluate_aircraft,
        bounds,
        mu=20,
        lambda_=140,
        max_iter=200
    )
    
    best_design, cost, history = es.optimize()
    return best_design, cost
```

### **9.2 Calibração de Modelos Climáticos**

```python
def climate_model_calibration():
    """
    Calibra parâmetros de modelo climático
    """
    def climate_model_error(params):
        """
        Compara simulação com dados observados
        params: parâmetros físicos do modelo
        """
        # Executar modelo climático
        simulated_temp = run_climate_model(params)  # Função externa
        
        # Comparar com observações
        observed_temp = load_observations()
        
        # Erro RMSE
        error = np.sqrt(np.mean((simulated_temp - observed_temp)**2))
        
        return error
    
    # Parâmetros: [albedo, cloud_feedback, ocean_heat, ...]
    bounds = [(0.2, 0.4), (-2, 2), (0, 100), ...]
    
    # Usar (μ,λ)-ES por ser robusto a ruído
    es = MuCommaLambdaES(  # Implementar variante comma
        climate_model_error,
        bounds,
        mu=30,
        lambda_=210,
        max_iter=100
    )
    
    best_params, error, history = es.optimize()
    return best_params
```

---

## **10. 🔗 Referências e Recursos**

### **10.1 📚 Publicações Fundamentais**

1. **Rechenberg, I. (1965).** *"Cybernetic Solution Path of an Experimental Problem"*. 
   - 🌟 Trabalho original que introduziu Evolution Strategies
   - Royal Aircraft Establishment, Library Translation

2. **Schwefel, H. P. (1981).** *"Numerical Optimization of Computer Models"*. 
   - 📖 Primeira formulação completa de ES
   - John Wiley & Sons

3. **Hansen, N., & Ostermeier, A. (2001).** *"Completely Derandomized Self-Adaptation in Evolution Strategies"*. Evolutionary Computation, 9(2), 159-195.
   - 🎯 Artigo definitivo sobre CMA-ES
   - Estado-da-arte em otimização

4. **Beyer, H. G., & Schwefel, H. P. (2002).** *"Evolution Strategies – A Comprehensive Introduction"*. Natural Computing, 1(1), 3-52.
   - 📊 Survey completo sobre ES
   - Teoria e prática

5. **Salimans, T., et al. (2017).** *"Evolution Strategies as a Scalable Alternative to Reinforcement Learning"*. arXiv:1703.03864.
   - 🚀 OpenAI ES para deep RL
   - Paralelização massiva

### **10.2 📖 Livros Recomendados**

- **"Introduction to Evolutionary Computing"** - Eiben & Smith (2015)
  - Capítulo completo sobre ES
  
- **"Evolutionary Algorithms in Theory and Practice"** - Bäck (1996)
  - Foco em ES e análise teórica

- **"Natural Computing Series: Theory of Evolutionary Algorithms"** - Beyer (2001)
  - Análise matemática profunda

### **10.3 🛠️ Bibliotecas e Ferramentas**

#### **Python**
```python
# 1. pycma - CMA-ES de referência
pip install cma
import cma
es = cma.CMAEvolutionStrategy(10 * [0], 0.5)
es.optimize(objective_function)

# 2. deap - Framework completo
pip install deap
from deap import algorithms, base, tools

# 3. evotorch - ES moderno com PyTorch
pip install evotorch

# 4. nevergrad - Facebook Research
pip install nevergrad
import nevergrad as ng
optimizer = ng.optimizers.CMA(parametrization=10)

# 5. estorch - ES para PyTorch/RL
pip install estorch
```

#### **Outras Linguagens**
- **Java:** JCLEC, ECJ
- **C++:** Shark ML Library
- **MATLAB:** Global Optimization Toolbox
- **R:** cmaes package

### **10.4 🌐 Recursos Online**

| Recurso | Descrição | Link |
|---------|-----------|------|
| **CMA-ES Tutorial** | Tutorial oficial de Hansen | cma.gforge.inria.fr |
| **OpenAI ES GitHub** | Implementação escalável | github.com/openai/evolution-strategies-starter |
| **Nikolaus Hansen's Page** | Papers, código, tutoriais | www.cmap.polytechnique.fr/~nikolaus.hansen |
| **Evolution Strategies Wiki** | Enciclopédia de ES | www.scholarpedia.org/article/Evolution_strategies |

### **10.5 🎓 Cursos e Tutoriais**

- **Coursera:** Evolutionary Algorithms
- **MIT OCW:** Computational Evolutionary Biology
- **YouTube:** Lectures by Nikolaus Hansen (CMA-ES)
- **Tutorial Papers:** Hansen (2016) "The CMA Evolution Strategy: A Tutorial"

---

## **11. 🎯 Conclusão**

As Estratégias de Evolução representam uma das abordagens mais **sofisticadas e eficazes** para otimização contínua. Suas características principais são:

### **🔑 Principais Aprendizados**

1. **Auto-adaptação:** O conceito revolucionário de evoluir a estratégia de busca junto com a solução
2. **Robustez:** Especialmente em ambientes ruidosos e dinâmicos
3. **Teoria Sólida:** Base matemática forte permite entendimento profundo
4. **CMA-ES:** Estado-da-arte em otimização contínua
5. **Escalabilidade:** OpenAI ES demonstrou aplicabilidade em deep learning

### **💡 Comparação com Outros Métodos**

| Método | Gradiente | Multimodal | Ruído | Dimensão Alta | Auto-adaptação |
|--------|-----------|------------|-------|---------------|----------------|
| **ES** | ❌ | ✅ | ✅✅ | ✅ | ✅✅ |
| **GA** | ❌ | ✅ | ⚪ | ⚪ | ❌ |
| **DE** | ❌ | ✅✅ | ⚪ | ✅ | ⚪ |
| **Gradient Descent** | ✅ | ❌ | ❌ | ✅✅ | ❌ |
| **Particle Swarm** | ❌ | ✅ | ⚪ | ⚪ | ❌ |

### **🚀 Próximos Passos**

1. **Comece Simples:** Implemente (1+1)-ES
2. **Entenda Auto-adaptação:** Experimente com diferentes τ
3. **Use CMA-ES:** Para problemas reais, use biblioteca `pycma`
4. **Explore Paralelização:** Teste OpenAI ES para problemas grandes
5. **Compare:** Benchmark contra DE e GA
6. **Aplique:** Use em seus problemas de otimização

### **🌟 Reflexão Final**

As Estratégias de Evolução demonstram um princípio profundo: **não apenas a solução deve evoluir, mas também a estratégia para encontrá-la**. Esta meta-evolução permite que o algoritmo se adapte automaticamente às características do problema, tornando-o excepcionalmente robusto e versátil.

> *"A essência das Estratégias de Evolução está em reconhecer que o caminho para a solução é tão importante quanto a solução em si - e ambos devem evoluir juntos."*

**Em destaque: CMA-ES** é considerado por muitos como o melhor algoritmo de otimização livre de gradiente para problemas contínuos de até ~100 dimensões. Se você precisa otimizar uma função black-box, CMA-ES deve ser sua primeira escolha.

---

**🔗 Continue explorando:**
- 📖 Compare com [**Differential Evolution**](differential_evolution.md) para entender diferenças
- 🧬 Veja [**Genetic Algorithms**](genetic_algorithms.md) para abordagem discreta
- 🎯 Explore [**Algoritmos Evolucionários**](README.md) para visão geral
- 🌳 Descubra [**Genetic Programming**](genetic_programming.md) para evolução de programas

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

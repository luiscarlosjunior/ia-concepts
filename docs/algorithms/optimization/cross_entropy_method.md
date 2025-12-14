# Método de Entropia Cruzada (Cross-Entropy Method - CE)

O **Método de Entropia Cruzada** (Cross-Entropy Method - CE) é um algoritmo de otimização estocástica e simulação de eventos raros baseado em amostragem adaptativa. Desenvolvido por Reuven Rubinstein em 1997, o método CE combina conceitos de teoria da informação, simulação de Monte Carlo e aprendizado adaptativo para resolver problemas complexos de otimização e estimação.

![Cross-Entropy Method Concept](../../images/ce_concept.png)

O algoritmo é particularmente eficaz em problemas de otimização combinatória, otimização contínua e estimação de probabilidades de eventos raros, sendo amplamente utilizado em áreas como aprendizado de máquina, engenharia de confiabilidade, telecomunicações e otimização de sistemas complexos.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 Conceito de Entropia Cruzada**

A **entropia cruzada** é uma medida da diferença entre duas distribuições de probabilidade. Dadas duas distribuições P (distribuição alvo) e Q (distribuição de amostragem), a entropia cruzada é definida como:

```
D(P||Q) = ∑ P(x) log(P(x)/Q(x))
```

Esta é também conhecida como **divergência de Kullback-Leibler (KL)**, que mede quão diferente Q é de P.

### **1.2 Princípio Fundamental do Método CE**

O método CE utiliza um processo iterativo para:

1. **Gerar amostras** de uma distribuição de probabilidade parametrizada
2. **Selecionar as melhores amostras** baseado em um critério de performance
3. **Atualizar os parâmetros** da distribuição para gerar melhores amostras na próxima iteração
4. **Minimizar a entropia cruzada** entre a distribuição atual e a distribuição ótima

**Intuição:**
> "Se queremos encontrar soluções ótimas, devemos aprender a gerar amostras cada vez melhores, concentrando nossa distribuição de amostragem nas regiões promissoras do espaço de busca."

### **1.3 Motivação: Por Que o Método CE Funciona?**

#### **🎲 Problema de Amostragem Naive**
```
Problema: Encontrar x que maximize f(x)
Abordagem ingênua: Amostrar uniformemente e escolher o melhor

❌ Problemas:
- Espaço de busca muito grande
- Soluções ótimas são raras
- Desperdício de recursos computacionais
```

#### **✅ Solução do Método CE**
```
1. Iniciar com distribuição ampla
2. Amostrar soluções
3. Identificar as top-k melhores
4. Ajustar distribuição para favorecer regiões promissoras
5. Repetir até convergência

✅ Vantagens:
- Foco adaptativo nas regiões boas
- Equilíbrio exploração-explotação
- Convergência eficiente
```

---

## **2. 🔧 Algoritmo do Método CE**

### **2.1 Estrutura Geral**

```
🚀 1. INICIALIZAÇÃO
   ├── Definir distribuição inicial f(·;θ₀)
   ├── Definir parâmetros: N (tamanho da amostra), ρ (percentil)
   └── t ← 0

🔄 2. LOOP PRINCIPAL (enquanto não convergir):
   ├── 📊 Gerar N amostras de f(·;θₜ)
   ├── 🎯 Avaliar função objetivo S(x) para cada amostra
   ├── 📈 Selecionar top-(ρN) melhores amostras (elite)
   ├── 🔄 Atualizar parâmetros θₜ₊₁ minimizando CE com elite
   ├── 🎛️ Aplicar suavização: θₜ₊₁ ← αθₜ₊₁ + (1-α)θₜ
   └── t ← t + 1

🏁 3. RETORNAR melhor solução encontrada
```

### **2.2 Pseudocódigo Detalhado**

```python
def cross_entropy_method(f_objective, theta_init, N, rho, max_iter, alpha=0.7):
    """
    Método de Entropia Cruzada para otimização
    
    Args:
        f_objective: Função objetivo a maximizar
        theta_init: Parâmetros iniciais da distribuição
        N: Número de amostras por iteração
        rho: Percentil para seleção de elite (0 < rho < 1)
        max_iter: Número máximo de iterações
        alpha: Parâmetro de suavização (0 < alpha < 1)
    
    Returns:
        melhor_solucao, melhor_valor, historico
    """
    
    theta = theta_init
    historico = []
    melhor_solucao_global = None
    melhor_valor_global = -inf
    
    for iteracao in range(max_iter):
        # 1. GERAÇÃO: Amostrar N soluções da distribuição atual
        amostras = gerar_amostras(theta, N)
        
        # 2. AVALIAÇÃO: Calcular valor objetivo de cada amostra
        valores = [f_objective(x) for x in amostras]
        
        # 3. SELEÇÃO: Escolher elite (top-rho amostras)
        num_elite = max(1, int(rho * N))
        indices_ordenados = argsort(valores)[::-1]  # Ordem decrescente
        indices_elite = indices_ordenados[:num_elite]
        elite = [amostras[i] for i in indices_elite]
        
        # 4. ATUALIZAÇÃO: Ajustar parâmetros baseado na elite
        theta_novo = estimar_parametros(elite)
        
        # 5. SUAVIZAÇÃO: Evitar mudanças muito bruscas
        theta = alpha * theta_novo + (1 - alpha) * theta
        
        # 6. REGISTRO: Armazenar melhor solução
        melhor_idx = indices_elite[0]
        if valores[melhor_idx] > melhor_valor_global:
            melhor_solucao_global = amostras[melhor_idx]
            melhor_valor_global = valores[melhor_idx]
        
        # 7. HISTÓRICO: Guardar progresso
        historico.append({
            'iteracao': iteracao,
            'melhor_valor': melhor_valor_global,
            'media_elite': mean([valores[i] for i in indices_elite]),
            'theta': theta.copy()
        })
        
        # 8. CRITÉRIO DE PARADA: Verificar convergência
        if verificar_convergencia(historico, criterio='variacao_theta'):
            break
    
    return melhor_solucao_global, melhor_valor_global, historico
```

### **2.3 Componentes Fundamentais**

#### **📊 1. Distribuição de Amostragem**

A escolha da distribuição de probabilidade depende do tipo do problema:

**Problemas Contínuos:**
```python
# Distribuição Gaussiana Multivariada
class GaussianSampling:
    def __init__(self, dim):
        self.mu = np.zeros(dim)          # Média inicial
        self.sigma = np.eye(dim)         # Covariância inicial
    
    def sample(self, N):
        """Gerar N amostras da distribuição"""
        return np.random.multivariate_normal(self.mu, self.sigma, N)
    
    def update(self, elite_samples):
        """Atualizar parâmetros baseado na elite"""
        self.mu = np.mean(elite_samples, axis=0)
        self.sigma = np.cov(elite_samples.T)
        
        # Regularização para evitar singularidade
        self.sigma += 1e-6 * np.eye(len(self.mu))
```

**Problemas Discretos/Combinatoriais:**
```python
# Distribuição Bernoulli para problemas binários
class BernoulliSampling:
    def __init__(self, dim):
        self.p = np.ones(dim) * 0.5  # Probabilidades iniciais
    
    def sample(self, N):
        """Gerar N amostras binárias"""
        return np.random.rand(N, len(self.p)) < self.p
    
    def update(self, elite_samples):
        """Atualizar probabilidades"""
        self.p = np.mean(elite_samples, axis=0)
        
        # Evitar probabilidades extremas
        self.p = np.clip(self.p, 0.01, 0.99)
```

#### **🎯 2. Seleção de Elite**

A seleção das melhores amostras é crucial:

```python
def selecionar_elite(amostras, valores, rho, metodo='percentil'):
    """
    Seleciona amostras elite baseado em diferentes critérios
    """
    N = len(amostras)
    
    if metodo == 'percentil':
        # Usar top-rho% das amostras
        num_elite = max(1, int(rho * N))
        indices = np.argsort(valores)[::-1][:num_elite]
    
    elif metodo == 'limiar_adaptativo':
        # Usar limiar que se adapta à performance
        limiar = np.percentile(valores, 100 * (1 - rho))
        indices = np.where(valores >= limiar)[0]
    
    elif metodo == 'ranking_ponderado':
        # Pesos maiores para melhores soluções
        ranks = np.argsort(np.argsort(valores)[::-1])
        pesos = np.exp(-ranks / (rho * N))
        # Amostragem ponderada da elite
        indices = np.random.choice(N, size=int(rho*N), 
                                  replace=False, p=pesos/pesos.sum())
    
    return [amostras[i] for i in indices], indices
```

#### **🎛️ 3. Suavização de Parâmetros**

Evita mudanças muito bruscas que podem causar convergência prematura:

```python
def suavizar_parametros(theta_novo, theta_antigo, alpha=0.7, metodo='linear'):
    """
    Suavização adaptativa dos parâmetros
    """
    if metodo == 'linear':
        # Suavização linear padrão
        theta = alpha * theta_novo + (1 - alpha) * theta_antigo
    
    elif metodo == 'adaptativo':
        # Ajustar alpha baseado na taxa de melhoria
        if melhorou_significativamente():
            alpha = 0.9  # Aceitar mudança maior
        else:
            alpha = 0.5  # Ser mais conservador
        
        theta = alpha * theta_novo + (1 - alpha) * theta_antigo
    
    elif metodo == 'momentum':
        # Incluir "momentum" das iterações anteriores
        if not hasattr(suavizar_parametros, 'velocidade'):
            suavizar_parametros.velocidade = theta_novo - theta_antigo
        
        beta = 0.9
        suavizar_parametros.velocidade = (beta * suavizar_parametros.velocidade + 
                                         (1 - beta) * (theta_novo - theta_antigo))
        theta = theta_antigo + suavizar_parametros.velocidade
    
    return theta
```

---

## **3. 📊 Aplicações do Método CE**

### **3.1 🎯 Otimização de Funções Contínuas**

**Problema:** Minimizar função de Rastrigin (muitos ótimos locais)

```python
import numpy as np
import matplotlib.pyplot as plt

class CEContinuousOptimization:
    """CE para otimização de funções contínuas"""
    
    def __init__(self, objective_func, dim, bounds):
        self.objective = objective_func
        self.dim = dim
        self.bounds = bounds  # [(min, max) para cada dimensão]
        
        # Inicializar distribuição gaussiana
        self.mu = np.array([(b[0] + b[1])/2 for b in bounds])
        range_vals = np.array([b[1] - b[0] for b in bounds])
        self.sigma = np.diag((range_vals / 4) ** 2)
    
    def optimize(self, N=100, rho=0.1, max_iter=100, alpha=0.7):
        """Executar otimização"""
        historico = []
        melhor_solucao = None
        melhor_valor = float('inf')
        
        for it in range(max_iter):
            # 1. Gerar amostras
            amostras = np.random.multivariate_normal(self.mu, self.sigma, N)
            
            # Aplicar bounds
            for i, (low, high) in enumerate(self.bounds):
                amostras[:, i] = np.clip(amostras[:, i], low, high)
            
            # 2. Avaliar
            valores = np.array([self.objective(x) for x in amostras])
            
            # 3. Selecionar elite
            num_elite = max(1, int(rho * N))
            indices_elite = np.argsort(valores)[:num_elite]
            elite = amostras[indices_elite]
            
            # 4. Atualizar distribuição
            mu_novo = np.mean(elite, axis=0)
            sigma_novo = np.cov(elite.T)
            
            # Regularização
            sigma_novo += 1e-6 * np.eye(self.dim)
            
            # 5. Suavização
            self.mu = alpha * mu_novo + (1 - alpha) * self.mu
            self.sigma = alpha * sigma_novo + (1 - alpha) * self.sigma
            
            # 6. Registrar melhor
            melhor_idx = indices_elite[0]
            if valores[melhor_idx] < melhor_valor:
                melhor_valor = valores[melhor_idx]
                melhor_solucao = amostras[melhor_idx].copy()
            
            historico.append({
                'iteracao': it,
                'melhor_valor': melhor_valor,
                'media_elite': np.mean(valores[indices_elite]),
                'variancia': np.mean(np.diag(self.sigma))
            })
            
            # Convergência
            if np.mean(np.diag(self.sigma)) < 1e-8:
                print(f"Convergiu na iteração {it}")
                break
        
        return melhor_solucao, melhor_valor, historico
    
    def plot_convergence(self, historico):
        """Visualizar convergência"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        iterations = [h['iteracao'] for h in historico]
        melhor = [h['melhor_valor'] for h in historico]
        media = [h['media_elite'] for h in historico]
        variancia = [h['variancia'] for h in historico]
        
        # Convergência do valor objetivo
        ax1.plot(iterations, melhor, 'b-', label='Melhor Valor', linewidth=2)
        ax1.plot(iterations, media, 'r--', label='Média Elite', alpha=0.7)
        ax1.set_xlabel('Iteração')
        ax1.set_ylabel('Valor Objetivo')
        ax1.set_title('Convergência do Método CE')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Evolução da variância
        ax2.plot(iterations, variancia, 'g-', linewidth=2)
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('Variância Média')
        ax2.set_title('Redução da Variância')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Exemplo: Otimizar função de Rastrigin
def rastrigin(x, A=10):
    """Função de Rastrigin - muitos ótimos locais"""
    n = len(x)
    return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)

if __name__ == "__main__":
    # Configurar problema
    dim = 5
    bounds = [(-5.12, 5.12)] * dim
    
    ce = CEContinuousOptimization(rastrigin, dim, bounds)
    
    # Executar otimização
    solucao, valor, hist = ce.optimize(N=200, rho=0.1, max_iter=100)
    
    print(f"Melhor solução: {solucao}")
    print(f"Valor objetivo: {valor:.6f}")
    print(f"Ótimo global: 0.0")
    
    ce.plot_convergence(hist)
```

### **3.2 🧩 Problema do Caixeiro Viajante (TSP)**

**Aplicação:** Usar CE para resolver TSP com distribuição sobre permutações

```python
class CETSP:
    """Cross-Entropy Method para TSP"""
    
    def __init__(self, cities):
        self.cities = np.array(cities)
        self.n_cities = len(cities)
        self.dist_matrix = self._compute_distances()
        
        # Matriz de probabilidade de transição
        # P[i][j] = probabilidade de ir da cidade i para j
        self.P = np.ones((self.n_cities, self.n_cities)) / (self.n_cities - 1)
        np.fill_diagonal(self.P, 0)  # Não pode ir para si mesmo
    
    def _compute_distances(self):
        """Pré-computar matriz de distâncias"""
        n = self.n_cities
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(self.cities[i] - self.cities[j])
                dist[i][j] = dist[j][i] = d
        return dist
    
    def sample_tour(self):
        """Amostrar tour baseado na matriz de probabilidade"""
        tour = [0]  # Começar da cidade 0
        remaining = list(range(1, self.n_cities))
        
        while remaining:
            current = tour[-1]
            # Probabilidades para próxima cidade
            probs = self.P[current, remaining]
            probs = probs / probs.sum()  # Normalizar
            
            # Escolher próxima cidade
            next_city = np.random.choice(remaining, p=probs)
            tour.append(next_city)
            remaining.remove(next_city)
        
        return tour
    
    def tour_length(self, tour):
        """Calcular comprimento total do tour"""
        length = 0
        for i in range(self.n_cities):
            length += self.dist_matrix[tour[i]][tour[(i+1) % self.n_cities]]
        return length
    
    def optimize(self, N=100, rho=0.1, max_iter=100, alpha=0.7):
        """Executar CE para TSP"""
        historico = []
        melhor_tour = None
        melhor_distancia = float('inf')
        
        for it in range(max_iter):
            # 1. Gerar tours
            tours = [self.sample_tour() for _ in range(N)]
            
            # 2. Avaliar
            distancias = np.array([self.tour_length(tour) for tour in tours])
            
            # 3. Selecionar elite
            num_elite = max(1, int(rho * N))
            indices_elite = np.argsort(distancias)[:num_elite]
            elite_tours = [tours[i] for i in indices_elite]
            
            # 4. Atualizar matriz de probabilidade
            P_novo = np.zeros((self.n_cities, self.n_cities))
            
            for tour in elite_tours:
                for i in range(self.n_cities):
                    cidade_atual = tour[i]
                    proxima_cidade = tour[(i+1) % self.n_cities]
                    P_novo[cidade_atual][proxima_cidade] += 1
            
            # Normalizar
            for i in range(self.n_cities):
                row_sum = P_novo[i].sum()
                if row_sum > 0:
                    P_novo[i] /= row_sum
                else:
                    P_novo[i] = np.ones(self.n_cities) / self.n_cities
            
            np.fill_diagonal(P_novo, 0)
            
            # 5. Suavização
            self.P = alpha * P_novo + (1 - alpha) * self.P
            
            # Garantir que soma das linhas = 1
            for i in range(self.n_cities):
                row_sum = self.P[i].sum()
                if row_sum > 0:
                    self.P[i] /= row_sum
            
            # 6. Registrar melhor
            melhor_idx = indices_elite[0]
            if distancias[melhor_idx] < melhor_distancia:
                melhor_distancia = distancias[melhor_idx]
                melhor_tour = tours[melhor_idx].copy()
            
            historico.append({
                'iteracao': it,
                'melhor_distancia': melhor_distancia,
                'media_elite': np.mean(distancias[indices_elite]),
                'pior_elite': np.max(distancias[indices_elite])
            })
            
            if it % 10 == 0:
                print(f"Iter {it}: Melhor = {melhor_distancia:.2f}")
        
        return melhor_tour, melhor_distancia, historico
    
    def plot_tour(self, tour, title="TSP Tour"):
        """Visualizar tour"""
        plt.figure(figsize=(8, 8))
        
        # Plotar cidades
        x = self.cities[:, 0]
        y = self.cities[:, 1]
        plt.scatter(x, y, c='red', s=200, zorder=5)
        
        # Plotar tour
        for i in range(self.n_cities):
            start = self.cities[tour[i]]
            end = self.cities[tour[(i + 1) % self.n_cities]]
            plt.plot([start[0], end[0]], [start[1], end[1]], 
                    'b-', linewidth=2, alpha=0.7)
        
        # Numerar cidades
        for i, (cx, cy) in enumerate(self.cities):
            plt.annotate(str(i), (cx, cy), fontsize=12, ha='center')
        
        plt.title(f"{title}\nDistância: {self.tour_length(tour):.2f}")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Gerar cidades aleatórias
    np.random.seed(42)
    n_cities = 20
    cities = np.random.rand(n_cities, 2) * 100
    
    # Resolver TSP com CE
    tsp = CETSP(cities)
    tour, dist, hist = tsp.optimize(N=200, rho=0.1, max_iter=100, alpha=0.7)
    
    print(f"\nMelhor tour: {tour}")
    print(f"Distância: {dist:.2f}")
    
    # Visualizar
    tsp.plot_tour(tour, "Melhor Tour Encontrado pelo CE")
```

### **3.3 🎰 Estimação de Eventos Raros**

**Problema:** Estimar probabilidade de eventos raros em simulação

```python
class CERareEventEstimation:
    """CE para estimação de probabilidades de eventos raros"""
    
    def __init__(self, system_model, rare_event_threshold):
        self.system = system_model
        self.threshold = rare_event_threshold
    
    def estimate_probability(self, N=1000, rho=0.1, max_iter=50):
        """
        Estimar P(S(X) >= threshold) onde S(X) é o score do sistema
        """
        # Parâmetros iniciais da distribuição (ex: Gaussiana)
        mu = np.zeros(self.system.dim)
        sigma = np.eye(self.system.dim)
        
        gamma = []  # Limiares adaptativos
        
        for it in range(max_iter):
            # 1. Gerar amostras
            samples = np.random.multivariate_normal(mu, sigma, N)
            
            # 2. Avaliar scores
            scores = np.array([self.system.evaluate(x) for x in samples])
            
            # 3. Determinar limiar adaptativo
            gamma_t = np.percentile(scores, 100 * (1 - rho))
            gamma.append(gamma_t)
            
            # Verificar se atingiu o evento raro
            if gamma_t >= self.threshold:
                # Estimar probabilidade final
                prob = np.mean(scores >= self.threshold)
                
                # Probabilidade acumulada
                prob_total = prob
                for g in gamma[:-1]:
                    prob_total *= rho
                
                return prob_total, gamma
            
            # 4. Selecionar elite (amostras que excederam limiar)
            elite_indices = np.where(scores >= gamma_t)[0]
            elite = samples[elite_indices]
            
            # 5. Atualizar distribuição
            mu = np.mean(elite, axis=0)
            sigma = np.cov(elite.T) + 1e-6 * np.eye(self.system.dim)
        
        # Se não convergiu
        prob = np.mean(scores >= self.threshold)
        return prob, gamma

# Exemplo: Sistema de confiabilidade
class ReliabilitySystem:
    """Modelo de sistema para análise de confiabilidade"""
    
    def __init__(self, dim=10):
        self.dim = dim
        self.weights = np.random.randn(dim)
    
    def evaluate(self, x):
        """Score do sistema (maior = pior)"""
        return np.dot(self.weights, x)

if __name__ == "__main__":
    # Definir sistema
    system = ReliabilitySystem(dim=5)
    threshold = 10.0  # Evento raro: score >= 10
    
    # Estimar probabilidade do evento raro
    ce = CERareEventEstimation(system, threshold)
    prob, limiares = ce.estimate_probability(N=1000, rho=0.1)
    
    print(f"Probabilidade estimada do evento raro: {prob:.6e}")
    print(f"Limiares adaptativos: {limiares}")
```

### **3.4 🤖 Treinamento de Redes Neurais**

**Aplicação:** Otimizar pesos de rede neural usando CE

```python
class CENeuralNetwork:
    """CE para treinamento de redes neurais"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Calcular número total de parâmetros
        self.n_params = (input_dim * hidden_dim + hidden_dim +  # Camada 1
                        hidden_dim * output_dim + output_dim)   # Camada 2
        
        # Distribuição inicial
        self.mu = np.zeros(self.n_params)
        self.sigma = np.eye(self.n_params)
    
    def params_to_network(self, params):
        """Converter vetor de parâmetros em pesos da rede"""
        idx = 0
        
        # Camada 1
        w1_size = self.input_dim * self.hidden_dim
        W1 = params[idx:idx+w1_size].reshape(self.input_dim, self.hidden_dim)
        idx += w1_size
        
        b1 = params[idx:idx+self.hidden_dim]
        idx += self.hidden_dim
        
        # Camada 2
        w2_size = self.hidden_dim * self.output_dim
        W2 = params[idx:idx+w2_size].reshape(self.hidden_dim, self.output_dim)
        idx += w2_size
        
        b2 = params[idx:idx+self.output_dim]
        
        return W1, b1, W2, b2
    
    def forward(self, X, params):
        """Forward pass"""
        W1, b1, W2, b2 = self.params_to_network(params)
        
        # Camada oculta
        hidden = np.tanh(np.dot(X, W1) + b1)
        
        # Camada de saída
        output = np.dot(hidden, W2) + b2
        
        return output
    
    def evaluate(self, params, X, y):
        """Avaliar performance (negativo do erro para maximização)"""
        predictions = self.forward(X, params)
        mse = np.mean((predictions - y) ** 2)
        return -mse  # Negativo porque CE maximiza
    
    def train(self, X_train, y_train, N=50, rho=0.2, max_iter=100, alpha=0.7):
        """Treinar rede usando CE"""
        historico = []
        melhor_params = None
        melhor_score = float('-inf')
        
        for it in range(max_iter):
            # 1. Gerar conjuntos de parâmetros
            param_samples = np.random.multivariate_normal(self.mu, self.sigma, N)
            
            # 2. Avaliar cada conjunto
            scores = np.array([self.evaluate(p, X_train, y_train) 
                             for p in param_samples])
            
            # 3. Selecionar elite
            num_elite = max(1, int(rho * N))
            elite_indices = np.argsort(scores)[::-1][:num_elite]
            elite = param_samples[elite_indices]
            
            # 4. Atualizar distribuição
            mu_novo = np.mean(elite, axis=0)
            sigma_novo = np.cov(elite.T) + 1e-4 * np.eye(self.n_params)
            
            # 5. Suavização
            self.mu = alpha * mu_novo + (1 - alpha) * self.mu
            self.sigma = alpha * sigma_novo + (1 - alpha) * self.sigma
            
            # 6. Registrar melhor
            if scores[elite_indices[0]] > melhor_score:
                melhor_score = scores[elite_indices[0]]
                melhor_params = param_samples[elite_indices[0]].copy()
            
            historico.append({
                'iteracao': it,
                'melhor_score': melhor_score,
                'media_elite': np.mean(scores[elite_indices])
            })
            
            if it % 10 == 0:
                print(f"Iter {it}: Melhor MSE = {-melhor_score:.6f}")
        
        return melhor_params, historico
    
    def predict(self, X, params):
        """Fazer predições"""
        return self.forward(X, params)

# Exemplo: Regressão simples
if __name__ == "__main__":
    # Gerar dados sintéticos
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.1
    y = y.reshape(-1, 1)
    
    # Criar e treinar rede
    nn = CENeuralNetwork(input_dim=3, hidden_dim=5, output_dim=1)
    params, hist = nn.train(X, y, N=50, rho=0.2, max_iter=50)
    
    # Avaliar
    predictions = nn.predict(X, params)
    mse = np.mean((predictions - y) ** 2)
    print(f"\nMSE final: {mse:.6f}")
```

---

## **4. ⚖️ Vantagens e Limitações**

### **4.1 ✅ Vantagens**

| **Vantagem** | **Descrição** | **Impacto Prático** |
|--------------|---------------|---------------------|
| **🎯 Simplicidade Conceitual** | Fácil de entender e implementar | Rápida prototipagem |
| **🌐 Versatilidade** | Aplicável a problemas contínuos e discretos | Amplo espectro de uso |
| **📊 Base Teórica Sólida** | Fundamentado em teoria da informação | Garantias de convergência |
| **🔄 Adaptativo** | Aprende distribuição ótima iterativamente | Eficiência crescente |
| **⚡ Paralelizável** | Avaliações independentes | Escalabilidade |
| **🎲 Robustez** | Lida bem com ruído e não-convexidade | Aplicável a problemas reais |

### **4.2 ❌ Limitações**

| **Limitação** | **Descrição** | **Como Mitigar** |
|---------------|---------------|------------------|
| **🎛️ Sensibilidade a Parâmetros** | N, ρ, α afetam significativamente | Usar valores padrão testados |
| **💾 Uso de Memória** | Armazena matriz de covariância completa | Usar variantes de baixo rank |
| **🐌 Convergência Prematura** | Pode convergir para ótimos locais | Usar suavização adequada |
| **📈 Dimensionalidade Alta** | Problemas em espaços muito grandes | Reduzir dimensionalidade |
| **🔧 Escolha da Distribuição** | Requer conhecimento do problema | Testar diferentes famílias |

### **4.3 🆚 Comparação com Outros Métodos**

```
Critério                 │ CE    │ SA    │ GA    │ PSO   
─────────────────────────┼───────┼───────┼───────┼───────
🎯 Convergência Global   │ ✅✅  │ ✅✅  │ ✅    │ ✅    
⚡ Velocidade            │ ✅    │ ⚠️    │ ⚠️    │ ✅✅  
🧠 Simplicidade          │ ✅✅  │ ✅    │ ⚠️    │ ✅    
📊 Base Teórica          │ ✅✅  │ ✅    │ ⚠️    │ ⚠️    
🎛️ Poucos Parâmetros    │ ✅    │ ⚠️    │ ❌    │ ⚠️    
🌐 Versatilidade         │ ✅✅  │ ✅✅  │ ✅✅  │ ✅    
💾 Uso de Memória        │ ⚠️    │ ✅✅  │ ⚠️    │ ✅    
```

---

## **5. 🎓 Variantes e Extensões**

### **5.1 🔄 CE com Múltiplas Distribuições**

```python
class MultimodalCE:
    """CE com mistura de gaussianas para problemas multimodais"""
    
    def __init__(self, n_components=3, dim=2):
        self.n_components = n_components
        self.dim = dim
        
        # Inicializar componentes da mistura
        self.weights = np.ones(n_components) / n_components
        self.means = [np.random.randn(dim) for _ in range(n_components)]
        self.covariances = [np.eye(dim) for _ in range(n_components)]
    
    def sample(self, N):
        """Amostrar da mistura de gaussianas"""
        # Escolher componente para cada amostra
        components = np.random.choice(self.n_components, N, p=self.weights)
        
        samples = []
        for comp in components:
            sample = np.random.multivariate_normal(
                self.means[comp], 
                self.covariances[comp]
            )
            samples.append(sample)
        
        return np.array(samples)
    
    def update(self, elite_samples):
        """Atualizar mistura usando EM"""
        # Implementar algoritmo Expectation-Maximization
        # para ajustar componentes da mistura
        pass
```

### **5.2 🎯 CE Natural (Natural Cross-Entropy)**

Usa geometria de informação (gradiente natural) para atualização mais eficiente:

```python
def natural_ce_update(elite, mu, sigma):
    """
    Atualização usando gradiente natural
    Mais eficiente para alta dimensionalidade
    """
    n_elite = len(elite)
    
    # Gradiente natural da média
    grad_mu = np.mean(elite - mu, axis=0)
    
    # Gradiente natural da covariância (usando matriz de Fisher)
    centered = elite - mu
    grad_sigma = np.mean([np.outer(c, c) for c in centered], axis=0) - sigma
    
    # Atualização com passo adaptativo
    learning_rate_mu = 1.0 / np.sqrt(n_elite)
    learning_rate_sigma = 0.5 / np.sqrt(n_elite)
    
    mu_novo = mu + learning_rate_mu * grad_mu
    sigma_novo = sigma + learning_rate_sigma * grad_sigma
    
    # Garantir matriz positiva definida
    sigma_novo = (sigma_novo + sigma_novo.T) / 2
    sigma_novo += 1e-6 * np.eye(len(mu))
    
    return mu_novo, sigma_novo
```

### **5.3 📊 CE Multi-objetivo**

```python
class MultiObjectiveCE:
    """CE para otimização multi-objetivo"""
    
    def __init__(self, objectives, dim):
        self.objectives = objectives  # Lista de funções objetivo
        self.dim = dim
        self.mu = np.zeros(dim)
        self.sigma = np.eye(dim)
    
    def pareto_selection(self, samples, N, rho):
        """Selecionar elite baseado em dominância de Pareto"""
        n_obj = len(self.objectives)
        
        # Calcular valores para todos os objetivos
        objective_values = np.zeros((N, n_obj))
        for i, sample in enumerate(samples):
            for j, obj_func in enumerate(self.objectives):
                objective_values[i, j] = obj_func(sample)
        
        # Encontrar frente de Pareto
        pareto_front = []
        for i in range(N):
            dominated = False
            for j in range(N):
                if i != j:
                    # Verificar se j domina i
                    if all(objective_values[j] >= objective_values[i]) and \
                       any(objective_values[j] > objective_values[i]):
                        dominated = True
                        break
            if not dominated:
                pareto_front.append(i)
        
        # Se frente de Pareto é maior que elite, usar crowding distance
        num_elite = int(rho * N)
        if len(pareto_front) > num_elite:
            # Calcular crowding distance
            distances = self.crowding_distance(objective_values[pareto_front])
            sorted_indices = np.argsort(distances)[::-1][:num_elite]
            elite_indices = [pareto_front[i] for i in sorted_indices]
        else:
            elite_indices = pareto_front
        
        return elite_indices
    
    def crowding_distance(self, front_values):
        """Calcular crowding distance para diversidade"""
        n = len(front_values)
        distances = np.zeros(n)
        
        for obj_idx in range(front_values.shape[1]):
            # Ordenar por objetivo
            sorted_indices = np.argsort(front_values[:, obj_idx])
            
            # Extremos têm distância infinita
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calcular distâncias intermediárias
            obj_range = (front_values[sorted_indices[-1], obj_idx] - 
                        front_values[sorted_indices[0], obj_idx])
            
            if obj_range > 0:
                for i in range(1, n-1):
                    distances[sorted_indices[i]] += (
                        front_values[sorted_indices[i+1], obj_idx] -
                        front_values[sorted_indices[i-1], obj_idx]
                    ) / obj_range
        
        return distances
```

---

## **6. 📚 Configuração de Parâmetros e Boas Práticas**

### **6.1 🎛️ Guia de Configuração de Parâmetros**

#### **Tamanho da Amostra (N)**
```
🔹 Regra geral: N ≥ 10 × dimensionalidade

Problemas simples (dim < 10):    N = 50-100
Problemas médios (dim 10-100):   N = 100-500
Problemas complexos (dim > 100): N = 500-2000

💡 Dica: Começar com N = 100 e ajustar baseado em performance
```

#### **Percentil de Elite (ρ)**
```
🔹 Valores típicos: ρ = 0.01 a 0.20

ρ = 0.01-0.05: Mais exploração, convergência lenta
ρ = 0.10:      Balanceado (recomendado)
ρ = 0.15-0.20: Menos exploração, convergência rápida

💡 Dica: ρ = 0.10 funciona bem na maioria dos casos
```

#### **Parâmetro de Suavização (α)**
```
🔹 Valores típicos: α = 0.5 a 0.9

α = 0.5-0.6:   Mais conservador, evita convergência prematura
α = 0.7:       Balanceado (recomendado)
α = 0.8-0.9:   Mais agressivo, convergência rápida

💡 Dica: Começar com α = 0.7 e aumentar se convergência é lenta
```

### **6.2 ✅ Boas Práticas**

#### **1. Inicialização**
```python
def boa_inicializacao(problema):
    """
    Inicializar CE de forma inteligente
    """
    # ✅ BOM: Usar conhecimento do problema
    if problema.tem_bounds:
        mu = (problema.lower + problema.upper) / 2
        range_val = problema.upper - problema.lower
        sigma = np.diag((range_val / 4) ** 2)
    
    # ❌ RUIM: Inicialização arbitrária
    # mu = np.zeros(dim)
    # sigma = np.eye(dim) * 1000
    
    return mu, sigma
```

#### **2. Monitoramento de Convergência**
```python
def verificar_convergencia(historico, janela=10):
    """
    Critérios múltiplos para convergência
    """
    if len(historico) < janela:
        return False
    
    # Critério 1: Variância pequena
    var_atual = historico[-1]['variancia']
    if var_atual < 1e-8:
        return True
    
    # Critério 2: Pouca melhoria recente
    valores_recentes = [h['melhor_valor'] for h in historico[-janela:]]
    melhoria = (valores_recentes[0] - valores_recentes[-1]) / abs(valores_recentes[0])
    if melhoria < 1e-6:
        return True
    
    # Critério 3: Elite muito concentrada
    if historico[-1].get('dispersao_elite', 1) < 1e-6:
        return True
    
    return False
```

#### **3. Regularização**
```python
def regularizar_covariancia(sigma, metodo='diagonal', epsilon=1e-6):
    """
    Evitar singularidade da matriz de covariância
    """
    if metodo == 'diagonal':
        # Adicionar ruído na diagonal
        sigma_reg = sigma + epsilon * np.eye(len(sigma))
    
    elif metodo == 'minimo_eigenvalue':
        # Garantir eigenvalues mínimos
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        eigenvalues = np.maximum(eigenvalues, epsilon)
        sigma_reg = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    elif metodo == 'shrinkage':
        # Shrinkage em direção à identidade
        alpha_shrink = 0.1
        sigma_reg = (1 - alpha_shrink) * sigma + alpha_shrink * np.eye(len(sigma))
    
    return sigma_reg
```

#### **4. Tratamento de Restrições**
```python
def aplicar_restricoes(amostra, problema):
    """
    Lidar com restrições do problema
    """
    # Método 1: Projeção
    amostra_valida = np.clip(amostra, problema.lower, problema.upper)
    
    # Método 2: Penalização
    violacao = calcular_violacao(amostra, problema)
    penalidade = 1000 * violacao
    
    # Método 3: Reparação
    if not problema.is_feasible(amostra):
        amostra_valida = problema.repair(amostra)
    
    return amostra_valida
```

### **6.3 🚨 Problemas Comuns e Soluções**

| **Problema** | **Sintoma** | **Solução** |
|--------------|-------------|-------------|
| **Convergência Prematura** | Preso em ótimo local cedo | Diminuir α, aumentar N, usar α adaptativo |
| **Convergência Lenta** | Muitas iterações sem melhoria | Aumentar α, diminuir ρ |
| **Matriz Singular** | Erro numérico em amostragem | Adicionar regularização |
| **Explosão de Variância** | Variância cresce descontroladamente | Limitar variância máxima |
| **Elite Muito Pequena** | Poucos elementos na elite | Aumentar ρ ou N |
| **Amostragem Fora dos Bounds** | Soluções inválidas | Usar projeção ou penalização |

---

## **7. 📖 Referências e Recursos**

### **7.1 📚 Literatura Fundamental**

#### **Artigos Clássicos**
1. **Rubinstein, R. Y. (1997).** *"Optimization of computer simulation models with rare events"*. European Journal of Operational Research, 99(1), 89-112.
   - 🌟 **Marco inicial:** Introdução do método CE
   
2. **Rubinstein, R. Y., & Kroese, D. P. (2004).** *The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte-Carlo Simulation, and Machine Learning*. Springer.
   - 📖 **Livro definitivo:** Cobertura completa do método
   
3. **De Boer, P. T., et al. (2005).** *"A tutorial on the cross-entropy method"*. Annals of Operations Research, 134(1), 19-67.
   - 🎓 **Tutorial excelente:** Introdução detalhada e acessível

#### **Aplicações e Extensões**
4. **Kroese, D. P., et al. (2006).** *"The cross-entropy method for continuous multi-extremal optimization"*. Methodology and Computing in Applied Probability, 8(3), 383-407.
   
5. **Hu, J., et al. (2007).** *"A model reference adaptive search method for global optimization"*. Operations Research, 55(3), 549-568.

### **7.2 🌐 Recursos Online**

#### **Implementações**
```python
# Bibliotecas Python
import numpy as np
from scipy.stats import multivariate_normal

# Implementação de referência disponível em:
# - GitHub: rubinstein-group/cross-entropy
# - PyPI: pip install cross-entropy-method
```

#### **Tutoriais e Cursos**
- 📹 **YouTube:** "Cross-Entropy Method Explained" - StatQuest
- 📝 **Blogs:** Towards Data Science - "Understanding CE Method"
- 🎓 **Coursera:** "Simulation and Modeling" - University of Colorado

### **7.3 🔗 Links Úteis**

- **Documentação Oficial:** [Cross-Entropy Method Documentation](http://www.cemethod.org)
- **Código Fonte:** [GitHub - CE Implementations](https://github.com/topics/cross-entropy-method)
- **Comunidade:** Stack Overflow tag [cross-entropy-method]

---

## **8. 🎯 Conclusão**

### **8.1 💡 Principais Aprendizados**

O Método de Entropia Cruzada representa uma **abordagem elegante e eficiente** para otimização estocástica, combinando:

1. **📊 Fundamentação Teórica Sólida:** Baseado em teoria da informação
2. **🎯 Simplicidade Prática:** Fácil de implementar e entender
3. **🌐 Versatilidade:** Aplicável a diversos tipos de problemas
4. **⚡ Eficiência:** Convergência rápida em muitos cenários

### **8.2 🔑 Quando Usar CE**

#### **✅ Cenários Ideais:**
- Problemas de otimização com funções objetivo ruidosas
- Estimação de probabilidades de eventos raros
- Otimização combinatória de médio porte
- Quando não há gradientes disponíveis
- Problemas com múltiplos ótimos locais

#### **❌ Cenários Problemáticos:**
- Espaços de busca extremamente grandes
- Quando requer convergência garantida para ótimo global
- Problemas com muitas restrições complexas
- Quando há poucos recursos computacionais

### **8.3 🚀 Direções Futuras**

- **Integração com Deep Learning:** CE para otimização de arquiteturas neurais
- **CE Quântico:** Adaptações para computação quântica
- **CE Federado:** Aplicações em aprendizado federado
- **CE Multi-fidelidade:** Combinar simulações de diferentes precisões

### **8.4 🌟 Mensagem Final**

O Método de Entropia Cruzada nos ensina uma lição valiosa sobre aprendizado adaptativo:

> **"Ao invés de buscar exaustivamente, podemos aprender onde buscar, concentrando nossos esforços nas regiões mais promissoras do espaço de soluções."**

Esta filosofia de **aprendizado direcionado pela experiência** é aplicável não apenas em otimização, mas em muitos aspectos da resolução de problemas e tomada de decisão.

---

**🔗 Continue Explorando:**
- 📖 Veja também: [**Simulated Annealing**](../metaheuristics/simulated_annealing.md)
- 🧬 Próximo: [**Algoritmos Genéticos**](../metaheuristics/genetic_algorithms.md)
- 📊 Relacionado: [**Gaussian Process Regression**](../statistical_learning/gaussian_process_regression.md)

**🎓 Obrigado por explorar o Método de Entropia Cruzada!**

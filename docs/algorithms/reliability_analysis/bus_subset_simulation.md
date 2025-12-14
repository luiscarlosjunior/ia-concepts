# BUS com Subset Simulation (SUS)

O **BUS com Subset Simulation** (Bayesian Updating with Structural reliability methods combined with Subset Simulation) é um método avançado de análise de confiabilidade estrutural e estimação de probabilidades de eventos raros que combina técnicas de atualização bayesiana com simulação de subconjuntos. Este método é especialmente eficaz para estimar probabilidades muito pequenas que seriam impraticáveis de calcular por Monte Carlo simples.

![Subset Simulation Concept](../../images/sus_concept.png)

Desenvolvido para engenharia estrutural e análise de risco, o método permite calcular probabilidades de falha da ordem de 10⁻⁶ ou menores com eficiência computacional, sendo amplamente utilizado em análise de confiabilidade de estruturas, avaliação de riscos sísmicos, engenharia nuclear e sistemas complexos.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 O Problema de Eventos Raros**

Em análise de confiabilidade estrutural, frequentemente precisamos estimar:

```
Pf = P(g(X) ≤ 0)
```

Onde:
- **X:** Vetor de variáveis aleatórias (carregamentos, propriedades dos materiais, etc.)
- **g(X):** Função de desempenho ou função limite de estado
- **g(X) > 0:** Estado seguro
- **g(X) ≤ 0:** Estado de falha
- **Pf:** Probabilidade de falha

**Desafio:**
> Em estruturas bem projetadas, Pf é tipicamente muito pequeno (10⁻³ a 10⁻⁷), tornando a estimação por Monte Carlo simples extremamente ineficiente.

#### **🎲 Problema com Monte Carlo Simples**

```python
# Monte Carlo Simples
N = 1_000_000  # Número de amostras
samples = generate_random_samples(N)
failures = sum(g(x) <= 0 for x in samples)
Pf_estimate = failures / N

# Para Pf = 10⁻⁶:
# - Precisaríamos N ≈ 10⁹ amostras
# - Se cada avaliação leva 1ms, total = ~11 dias!
# ❌ Impraticável
```

**Coeficiente de Variação:**
```
COV = √((1-Pf)/(N·Pf))

Para Pf = 10⁻⁶ e COV = 10%:
N ≈ 10⁹ amostras necessárias!
```

### **1.2 Conceito de Subset Simulation**

A **Subset Simulation** (SUS) resolve este problema decompondo o evento raro em uma sequência de eventos intermediários mais prováveis:

```
F = F₁ ⊃ F₂ ⊃ ... ⊃ Fₘ

Onde:
- F: Evento de falha original
- Fᵢ: Eventos intermediários
- P(Fᵢ₊₁|Fᵢ) ≈ p₀ (probabilidade condicional constante, ex: 0.1)
```

**Fatorização:**
```
P(F) = P(F₁) × P(F₂|F₁) × ... × P(Fₘ|Fₘ₋₁)
```

**Vantagem:**
> Cada probabilidade condicional P(Fᵢ₊₁|Fᵢ) é maior (~0.1), permitindo estimação eficiente!

#### **📊 Ilustração Visual**

```
        Espaço Amostral Completo
    ┌─────────────────────────────┐
    │                             │
    │  ┌─────────────────────┐    │ F₁: Região mais ampla
    │  │                     │    │ P(F₁) ≈ 0.1
    │  │  ┌─────────────┐    │    │
    │  │  │             │    │    │ F₂: Região intermediária
    │  │  │  ┌─────┐    │    │    │ P(F₂|F₁) ≈ 0.1
    │  │  │  │  F  │    │    │    │
    │  │  │  └─────┘    │    │    │ F: Evento raro
    │  │  └─────────────┘    │    │ P(F|F₂) ≈ 0.1
    │  └─────────────────────┘    │
    └─────────────────────────────┘

P(F) = 0.1 × 0.1 × 0.1 = 10⁻³
```

### **1.3 BUS: Bayesian Updating with Subset Simulation**

O **BUS** integra Subset Simulation com inferência bayesiana para atualização de modelos:

**Teorema de Bayes:**
```
P(θ|D) = P(D|θ) × P(θ) / P(D)

Onde:
- θ: Parâmetros do modelo (incertos)
- D: Dados observados
- P(θ): Prior (conhecimento inicial)
- P(D|θ): Verossimilhança (compatibilidade com dados)
- P(θ|D): Posterior (conhecimento atualizado)
```

**Desafio no BUS:**
```
P(D) = ∫ P(D|θ) P(θ) dθ

Geralmente intratável analiticamente!
```

**Solução BUS:**
> Usar Subset Simulation para amostrar eficientemente do posterior, especialmente quando P(D|θ) é pequeno (dados raros ou extremos).

---

## **2. 🔧 Algoritmo Subset Simulation**

### **2.1 Algoritmo Completo**

```
🚀 1. INICIALIZAÇÃO
   ├── Definir função de desempenho g(X)
   ├── Definir N (tamanho da população, tipicamente 1000-5000)
   ├── Definir p₀ (probabilidade condicional alvo, tipicamente 0.1-0.2)
   └── m ← 0 (nível atual)

📊 2. NÍVEL 0: Monte Carlo Simples
   ├── Gerar N amostras de X ~ fₓ(x)
   ├── Avaliar g(X) para todas as amostras
   ├── Ordenar amostras por g(X)
   └── Estimar P(F₁) = #{g(X) ≤ g₁} / N
       onde g₁ é o (p₀×N)-ésimo menor valor

🔄 3. NÍVEIS INTERMEDIÁRIOS (m = 1, 2, ...)
   ├── Para cada amostra que satisfaz g(X) ≤ gₘ:
   │   └── Gerar 1/p₀ novas amostras usando MCMC
   │       (condicionado a g(X) ≤ gₘ)
   │
   ├── Avaliar g(X) para novas amostras
   ├── Ordenar por g(X)
   └── Se g_{(p₀×N)} > 0:
       │   └── gₘ₊₁ = g_{(p₀×N)} (continuar)
       └── Senão:
           └── Último nível alcançado

🏁 4. ÚLTIMO NÍVEL
   ├── Calcular #{g(X) ≤ 0} / N no último nível
   └── Pf = P(F₁) × ∏ₘ₌₁ᴹ⁻¹ p₀ × P(Fₘ|Fₘ₋₁)

📈 5. RETORNAR
   └── Probabilidade de falha Pf e amostras de falha
```

### **2.2 Componente MCMC: Modified Metropolis-Hastings**

O componente crucial é gerar novas amostras condicionadas a g(X) ≤ gₘ:

```python
def modified_metropolis_hastings(x_current, g_threshold, g_function, 
                                 proposal_std, max_iterations=1):
    """
    Modified Metropolis-Hastings para SUS
    
    Args:
        x_current: Amostra atual (já satisfaz g(x) ≤ g_threshold)
        g_threshold: Limiar atual
        g_function: Função de desempenho
        proposal_std: Desvio padrão da proposta
        max_iterations: Número de passos MCMC
    
    Returns:
        Nova amostra que satisfaz g(x) ≤ g_threshold
    """
    x = x_current.copy()
    
    for _ in range(max_iterations):
        # Propor nova amostra (caminhada aleatória gaussiana)
        x_proposed = x + np.random.normal(0, proposal_std, size=len(x))
        
        # Avaliar função de desempenho
        g_proposed = g_function(x_proposed)
        g_current = g_function(x)
        
        # Critério de aceitação modificado
        if g_proposed <= g_threshold:
            # Dentro da região: aceitar com probabilidade Metropolis
            # Para distribuições simétricas no espaço padrão:
            accept_prob = min(1.0, 
                             pdf_prior(x_proposed) / pdf_prior(x))
            
            if np.random.rand() < accept_prob:
                x = x_proposed
        
        # Se g_proposed > g_threshold: rejeitar automaticamente
    
    return x
```

---

## **3. 💻 Implementação Completa de Subset Simulation**

### **3.1 🔧 Implementação Básica**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

class SubsetSimulation:
    """
    Implementação de Subset Simulation para análise de confiabilidade
    """
    
    def __init__(self, performance_function, dim, p0=0.1, N=1000):
        """
        Args:
            performance_function: Função g(X), falha quando g(X) ≤ 0
            dim: Dimensão do vetor de variáveis aleatórias
            p0: Probabilidade condicional alvo para cada nível
            N: Tamanho da população em cada nível
        """
        self.g = performance_function
        self.dim = dim
        self.p0 = p0
        self.N = N
        
        # Número de sementes em cada nível
        self.n_seeds = int(p0 * N)
        
        # Número de cadeias por semente
        self.n_chains = int(1 / p0)
        
        # Histórico
        self.levels = []
        self.thresholds = []
        self.samples_per_level = []
    
    def sample_prior(self, n_samples):
        """
        Gerar amostras da distribuição prior
        
        Por padrão, assume distribuição gaussiana padrão multivariada
        Sobrescrever para outras distribuições
        """
        return np.random.randn(n_samples, self.dim)
    
    def prior_pdf(self, x):
        """
        PDF da distribuição prior
        """
        # Gaussiana padrão multivariada
        return np.prod(norm.pdf(x))
    
    def mcmc_step(self, x_current, g_threshold, proposal_std=1.0):
        """
        Um passo de MCMC condicionado a g(X) ≤ g_threshold
        """
        # Proposta: caminhada aleatória gaussiana
        x_proposed = x_current + np.random.normal(0, proposal_std, size=self.dim)
        
        # Avaliar proposta
        g_proposed = self.g(x_proposed)
        
        # Critério de aceitação modificado
        if g_proposed <= g_threshold:
            # Calcular razão de aceitação
            accept_ratio = self.prior_pdf(x_proposed) / self.prior_pdf(x_current)
            
            if np.random.rand() < min(1.0, accept_ratio):
                return x_proposed, g_proposed
        
        # Manter amostra atual
        g_current = self.g(x_current)
        return x_current, g_current
    
    def run(self, max_levels=20, verbose=True):
        """
        Executar Subset Simulation
        
        Returns:
            Pf: Probabilidade de falha estimada
            COV: Coeficiente de variação
            samples: Amostras do último nível (região de falha)
        """
        if verbose:
            print("="*60)
            print("SUBSET SIMULATION")
            print("="*60)
        
        # ===== NÍVEL 0: Monte Carlo Simples =====
        if verbose:
            print(f"\n📊 NÍVEL 0: Monte Carlo Simples")
            print(f"   Gerando {self.N} amostras...")
        
        samples = self.sample_prior(self.N)
        g_values = np.array([self.g(x) for x in samples])
        
        # Ordenar por valor de g
        sorted_indices = np.argsort(g_values)
        samples = samples[sorted_indices]
        g_values = g_values[sorted_indices]
        
        # Determinar limiar do primeiro nível
        threshold_idx = self.n_seeds
        g_threshold = g_values[threshold_idx]
        
        # Probabilidade do primeiro nível
        if g_threshold > 0:
            p_level = self.p0
        else:
            # Já alcançamos região de falha no primeiro nível
            p_level = np.sum(g_values <= 0) / self.N
            
            if verbose:
                print(f"   ✅ Região de falha alcançada no primeiro nível!")
                print(f"   Pf = {p_level:.6e}")
            
            self.levels.append(0)
            self.thresholds.append(0.0)
            self.samples_per_level.append(samples)
            
            return p_level, 0.0, samples[g_values <= 0]
        
        if verbose:
            print(f"   Limiar: g₁ = {g_threshold:.4f}")
            print(f"   P(F₁) = {p_level:.4f}")
        
        self.levels.append(0)
        self.thresholds.append(g_threshold)
        self.samples_per_level.append(samples.copy())
        
        # Produto acumulado de probabilidades
        prob_product = p_level
        
        # ===== NÍVEIS INTERMEDIÁRIOS =====
        level = 1
        
        while level < max_levels:
            if verbose:
                print(f"\n🔄 NÍVEL {level}")
            
            # Sementes: top p0*N amostras do nível anterior
            seeds = samples[:self.n_seeds]
            
            # Gerar novas amostras via MCMC
            new_samples = []
            new_g_values = []
            
            for seed in seeds:
                # Para cada semente, gerar n_chains amostras
                x = seed.copy()
                g_val = self.g(x)
                
                new_samples.append(x)
                new_g_values.append(g_val)
                
                # Gerar cadeias
                for _ in range(self.n_chains - 1):
                    x, g_val = self.mcmc_step(x, g_threshold)
                    new_samples.append(x.copy())
                    new_g_values.append(g_val)
            
            samples = np.array(new_samples)
            g_values = np.array(new_g_values)
            
            # Ordenar
            sorted_indices = np.argsort(g_values)
            samples = samples[sorted_indices]
            g_values = g_values[sorted_indices]
            
            # Novo limiar
            threshold_idx = self.n_seeds
            g_threshold_new = g_values[threshold_idx]
            
            if g_threshold_new > 0:
                # Continuar para próximo nível
                g_threshold = g_threshold_new
                p_level = self.p0
                prob_product *= p_level
                
                if verbose:
                    print(f"   Limiar: g_{level+1} = {g_threshold:.4f}")
                    print(f"   P(F_{level+1}|F_{level}) = {p_level:.4f}")
                    print(f"   P acumulada = {prob_product:.6e}")
                
                self.levels.append(level)
                self.thresholds.append(g_threshold)
                self.samples_per_level.append(samples.copy())
                
                level += 1
            else:
                # Último nível alcançado
                p_level = np.sum(g_values <= 0) / self.N
                prob_product *= p_level
                
                if verbose:
                    print(f"   ✅ Último nível alcançado!")
                    print(f"   P(Falha|F_{level}) = {p_level:.4f}")
                    print(f"   Pf = {prob_product:.6e}")
                
                self.levels.append(level)
                self.thresholds.append(0.0)
                self.samples_per_level.append(samples.copy())
                
                break
        
        # Calcular coeficiente de variação
        # Para SUS: COV ≈ √[(1-p0)/(p0*N*m)]
        m = len(self.levels)
        cov = np.sqrt((1 - self.p0) / (self.p0 * self.N * m))
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RESULTADO FINAL:")
            print(f"  Pf = {prob_product:.6e}")
            print(f"  COV = {cov:.4f} ({cov*100:.2f}%)")
            print(f"  Número de níveis: {m}")
            print(f"  Total de avaliações: {self.N + (m-1)*self.N}")
            print(f"{'='*60}")
        
        # Retornar amostras de falha
        failure_samples = samples[g_values <= 0]
        
        return prob_product, cov, failure_samples
    
    def plot_levels(self, figsize=(14, 8)):
        """
        Visualizar evolução dos níveis (apenas para 2D)
        """
        if self.dim != 2:
            print("Visualização disponível apenas para problemas 2D")
            return
        
        n_levels = len(self.levels)
        
        fig, axes = plt.subplots(2, (n_levels + 1) // 2, 
                                figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for i, level in enumerate(self.levels):
            ax = axes[i]
            
            samples = self.samples_per_level[i]
            g_vals = np.array([self.g(x) for x in samples])
            
            # Plotar amostras
            scatter = ax.scatter(samples[:, 0], samples[:, 1], 
                               c=g_vals, cmap='RdYlGn_r', 
                               s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Linha g(X) = 0 (região de falha)
            x_range = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 100)
            # Esta visualização assume conhecimento da forma de g
            # Em geral, seria necessário contour plot
            
            ax.set_xlabel('X₁')
            ax.set_ylabel('X₂')
            ax.set_title(f'Nível {level}: g ≤ {self.thresholds[i]:.2f}')
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label='g(X)')
        
        # Ocultar eixos não usados
        for i in range(n_levels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Exemplo: Problema de confiabilidade estrutural simples
def example_linear_performance():
    """
    Exemplo com função de desempenho linear
    g(X) = 3 - X₁ - X₂
    """
    
    def performance_function(x):
        return 3.0 - x[0] - x[1]
    
    # Executar SUS
    sus = SubsetSimulation(
        performance_function=performance_function,
        dim=2,
        p0=0.1,
        N=1000
    )
    
    Pf, COV, failure_samples = sus.run(verbose=True)
    
    # Calcular solução analítica (para validação)
    # P(X₁ + X₂ > 3) onde X₁, X₂ ~ N(0,1)
    # X₁ + X₂ ~ N(0, 2), então P(X₁ + X₂ > 3) = P(Z > 3/√2)
    from scipy.stats import norm
    Pf_exact = 1 - norm.cdf(3 / np.sqrt(2))
    
    print(f"\n{'='*60}")
    print(f"VALIDAÇÃO:")
    print(f"  Pf (SUS):   {Pf:.6e}")
    print(f"  Pf (Exato): {Pf_exact:.6e}")
    print(f"  Erro:       {abs(Pf - Pf_exact)/Pf_exact * 100:.2f}%")
    print(f"{'='*60}")
    
    # Visualizar
    sus.plot_levels()
    
    return Pf, COV

if __name__ == "__main__":
    example_linear_performance()
```

### **3.2 🎯 Exemplo: Função de Desempenho Não-Linear**

```python
def example_nonlinear_performance():
    """
    Exemplo mais complexo: função de desempenho não-linear
    g(X) = 5 - X₁² - X₂²  (círculo)
    """
    
    def performance_function(x):
        return 5.0 - x[0]**2 - x[1]**2
    
    # Executar SUS
    sus = SubsetSimulation(
        performance_function=performance_function,
        dim=2,
        p0=0.1,
        N=2000
    )
    
    Pf, COV, failure_samples = sus.run(verbose=True)
    
    # Visualizar espaço de falha
    plt.figure(figsize=(10, 8))
    
    # Gerar grid para contour
    x1 = np.linspace(-4, 4, 200)
    x2 = np.linspace(-4, 4, 200)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Calcular g em todo o grid
    G = np.zeros_like(X1)
    for i in range(len(x1)):
        for j in range(len(x2)):
            G[j, i] = performance_function(np.array([X1[j, i], X2[j, i]]))
    
    # Plot contour
    plt.contour(X1, X2, G, levels=[0], colors='red', linewidths=3, 
               label='Região de Falha (g=0)')
    plt.contourf(X1, X2, G, levels=[-100, 0], colors='red', alpha=0.2)
    
    # Plot amostras de falha
    if len(failure_samples) > 0:
        plt.scatter(failure_samples[:, 0], failure_samples[:, 1], 
                   c='darkred', s=50, marker='x', label='Amostras de Falha')
    
    # Plot últimas amostras de nível intermediário
    last_samples = sus.samples_per_level[-1]
    plt.scatter(last_samples[:, 0], last_samples[:, 1], 
               c='blue', s=20, alpha=0.5, label='Último Nível SUS')
    
    plt.xlabel('X₁')
    plt.ylabel('X₂')
    plt.title(f'Subset Simulation - Pf = {Pf:.6e}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # Calcular solução semi-analítica
    # P(X₁² + X₂² > 5) onde X₁, X₂ ~ N(0,1)
    # X₁² + X₂² ~ χ²(2)
    from scipy.stats import chi2
    Pf_exact = 1 - chi2.cdf(5, df=2)
    
    print(f"\n{'='*60}")
    print(f"VALIDAÇÃO:")
    print(f"  Pf (SUS):   {Pf:.6e}")
    print(f"  Pf (Exato): {Pf_exact:.6e}")
    print(f"  Erro:       {abs(Pf - Pf_exact)/Pf_exact * 100:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    example_nonlinear_performance()
```

---

## **4. 📊 BUS: Bayesian Updating with Subset Simulation**

### **4.1 🔧 Implementação do BUS**

```python
class BayesianUpdatingSubsetSimulation:
    """
    BUS - Bayesian Updating com Subset Simulation
    
    Para atualizar distribuição de parâmetros θ dado dados D
    usando Subset Simulation para amostrar do posterior
    """
    
    def __init__(self, prior_sampler, likelihood_function, dim_theta, 
                 p0=0.1, N=1000):
        """
        Args:
            prior_sampler: Função para amostrar do prior P(θ)
            likelihood_function: Função L(D|θ)
            dim_theta: Dimensão do vetor de parâmetros
            p0: Probabilidade condicional alvo
            N: Tamanho da população
        """
        self.sample_prior = prior_sampler
        self.likelihood = likelihood_function
        self.dim = dim_theta
        self.p0 = p0
        self.N = N
        
        self.n_seeds = int(p0 * N)
        self.n_chains = int(1 / p0)
        
        # Histórico
        self.levels = []
        self.thresholds = []
        self.samples_per_level = []
    
    def log_likelihood(self, theta):
        """
        Calcular log-verossimilhança
        """
        return np.log(self.likelihood(theta) + 1e-300)  # Evitar log(0)
    
    def mcmc_step(self, theta_current, log_L_threshold, proposal_std=0.5):
        """
        Passo MCMC para BUS
        """
        # Proposta
        theta_proposed = theta_current + np.random.normal(
            0, proposal_std, size=self.dim
        )
        
        # Log-verossimilhança da proposta
        log_L_proposed = self.log_likelihood(theta_proposed)
        
        # Critério de aceitação
        if log_L_proposed >= log_L_threshold:
            # Dentro da região: aceitar com probabilidade MH
            # Para prior uniforme ou simétrico, aceitar sempre
            return theta_proposed, log_L_proposed
        
        # Fora da região: rejeitar
        log_L_current = self.log_likelihood(theta_current)
        return theta_current, log_L_current
    
    def run(self, max_levels=20, target_log_L=None, verbose=True):
        """
        Executar BUS para amostrar do posterior
        
        Args:
            max_levels: Número máximo de níveis
            target_log_L: Log-verossimilhança alvo (opcional)
            verbose: Imprimir progresso
        
        Returns:
            posterior_samples: Amostras do posterior
            evidence: Estimativa da evidência P(D)
        """
        if verbose:
            print("="*60)
            print("BAYESIAN UPDATING WITH SUBSET SIMULATION")
            print("="*60)
        
        # ===== NÍVEL 0 =====
        if verbose:
            print(f"\n📊 NÍVEL 0: Amostragem do Prior")
        
        samples = np.array([self.sample_prior() for _ in range(self.N)])
        log_L_values = np.array([self.log_likelihood(theta) 
                                 for theta in samples])
        
        # Ordenar por log-verossimilhança (maior = melhor)
        sorted_indices = np.argsort(log_L_values)[::-1]
        samples = samples[sorted_indices]
        log_L_values = log_L_values[sorted_indices]
        
        # Limiar do primeiro nível
        threshold_idx = self.n_seeds
        log_L_threshold = log_L_values[threshold_idx]
        
        # Verificar se já atingimos alvo
        if target_log_L is not None and log_L_threshold >= target_log_L:
            if verbose:
                print(f"   ✅ Alvo alcançado no primeiro nível!")
            
            posterior_samples = samples[log_L_values >= target_log_L]
            evidence = len(posterior_samples) / self.N
            
            return posterior_samples, evidence
        
        if verbose:
            print(f"   Limiar: log L ≥ {log_L_threshold:.4f}")
            print(f"   P(L ≥ L₁) = {self.p0:.4f}")
        
        self.levels.append(0)
        self.thresholds.append(log_L_threshold)
        self.samples_per_level.append(samples.copy())
        
        # Produto de probabilidades
        prob_product = self.p0
        
        # ===== NÍVEIS INTERMEDIÁRIOS =====
        level = 1
        
        while level < max_levels:
            if verbose:
                print(f"\n🔄 NÍVEL {level}")
            
            # Sementes
            seeds = samples[:self.n_seeds]
            
            # MCMC para gerar novas amostras
            new_samples = []
            new_log_L = []
            
            for seed in seeds:
                theta = seed.copy()
                log_L = self.log_likelihood(theta)
                
                new_samples.append(theta)
                new_log_L.append(log_L)
                
                for _ in range(self.n_chains - 1):
                    theta, log_L = self.mcmc_step(theta, log_L_threshold)
                    new_samples.append(theta.copy())
                    new_log_L.append(log_L)
            
            samples = np.array(new_samples)
            log_L_values = np.array(new_log_L)
            
            # Ordenar
            sorted_indices = np.argsort(log_L_values)[::-1]
            samples = samples[sorted_indices]
            log_L_values = log_L_values[sorted_indices]
            
            # Novo limiar
            log_L_threshold_new = log_L_values[self.n_seeds]
            
            # Verificar se atingimos alvo
            if target_log_L is not None and log_L_threshold_new >= target_log_L:
                p_level = np.sum(log_L_values >= target_log_L) / self.N
                prob_product *= p_level
                
                if verbose:
                    print(f"   ✅ Alvo alcançado!")
                    print(f"   P(L ≥ Lalvo|F_{level}) = {p_level:.4f}")
                    print(f"   Evidence P(D) ≈ {prob_product:.6e}")
                
                posterior_samples = samples[log_L_values >= target_log_L]
                
                return posterior_samples, prob_product
            
            # Continuar
            log_L_threshold = log_L_threshold_new
            prob_product *= self.p0
            
            if verbose:
                print(f"   Limiar: log L ≥ {log_L_threshold:.4f}")
                print(f"   P acumulada = {prob_product:.6e}")
            
            self.levels.append(level)
            self.thresholds.append(log_L_threshold)
            self.samples_per_level.append(samples.copy())
            
            level += 1
        
        # Se não especificou alvo, retornar amostras do último nível
        if verbose:
            print(f"\n{'='*60}")
            print(f"Alcançado número máximo de níveis")
            print(f"Evidence (aproximada) ≈ {prob_product:.6e}")
            print(f"{'='*60}")
        
        return samples, prob_product

# Exemplo: Atualização bayesiana de parâmetro de modelo
def example_bus_parameter_estimation():
    """
    Exemplo: Estimar média μ de distribuição normal 
    dado observações
    """
    # Dados observados
    np.random.seed(42)
    true_mu = 2.0
    true_sigma = 1.0
    n_obs = 10
    observations = np.random.normal(true_mu, true_sigma, n_obs)
    
    print(f"Dados observados: média = {observations.mean():.4f}")
    
    # Prior: μ ~ N(0, 5²)
    def prior_sampler():
        return np.array([np.random.normal(0, 5)])
    
    # Likelihood: P(D|μ) assumindo σ conhecido
    def likelihood_function(theta):
        mu = theta[0]
        # Produto de normais
        log_L = -0.5 * np.sum((observations - mu)**2) / (true_sigma**2)
        log_L -= 0.5 * n_obs * np.log(2 * np.pi * true_sigma**2)
        return np.exp(log_L)
    
    # Executar BUS
    bus = BayesianUpdatingSubsetSimulation(
        prior_sampler=prior_sampler,
        likelihood_function=likelihood_function,
        dim_theta=1,
        p0=0.1,
        N=1000
    )
    
    posterior_samples, evidence = bus.run(
        max_levels=10,
        target_log_L=None,  # Amostrar de todo o posterior
        verbose=True
    )
    
    # Comparar com solução analítica
    # Posterior: μ ~ N(μ_post, σ_post²)
    sigma_prior = 5.0
    mu_post = (observations.sum() / true_sigma**2) / \
              (n_obs / true_sigma**2 + 1 / sigma_prior**2)
    sigma_post = 1 / np.sqrt(n_obs / true_sigma**2 + 1 / sigma_prior**2)
    
    print(f"\n{'='*60}")
    print(f"COMPARAÇÃO COM SOLUÇÃO ANALÍTICA:")
    print(f"  μ verdadeiro: {true_mu:.4f}")
    print(f"  BUS - μ posterior: {posterior_samples[:, 0].mean():.4f} ± "
          f"{posterior_samples[:, 0].std():.4f}")
    print(f"  Analítico - μ posterior: {mu_post:.4f} ± {sigma_post:.4f}")
    print(f"{'='*60}")
    
    # Visualizar posterior
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(posterior_samples[:, 0], bins=50, density=True, 
            alpha=0.7, label='BUS')
    
    # Posterior analítico
    x = np.linspace(posterior_samples[:, 0].min(), 
                   posterior_samples[:, 0].max(), 200)
    plt.plot(x, norm.pdf(x, mu_post, sigma_post), 
            'r-', linewidth=2, label='Analítico')
    
    plt.axvline(true_mu, color='green', linestyle='--', 
               linewidth=2, label='Verdadeiro')
    plt.xlabel('μ')
    plt.ylabel('Densidade')
    plt.title('Distribuição Posterior de μ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Trace plot
    plt.plot(posterior_samples[:, 0], 'b-', alpha=0.5)
    plt.axhline(true_mu, color='green', linestyle='--', 
               linewidth=2, label='Verdadeiro')
    plt.xlabel('Índice da Amostra')
    plt.ylabel('μ')
    plt.title('Trace Plot - Amostras do Posterior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    example_bus_parameter_estimation()
```

---

## **5. ⚖️ Vantagens e Limitações**

### **5.1 ✅ Vantagens**

| **Vantagem** | **Descrição** |
|--------------|---------------|
| **⚡ Eficiência** | Estima eventos raros com ~10⁴ avaliações vs ~10⁹ em MC |
| **🎯 Robustez** | COV não depende fortemente de Pf |
| **🔧 Simples** | Algoritmo relativamente fácil de implementar |
| **🌐 Generalidade** | Aplicável a funções de desempenho gerais |
| **📊 Amostras de Falha** | Fornece amostras da região de falha |
| **🧮 Integração Bayesiana** | BUS permite atualização eficiente |

### **5.2 ❌ Limitações**

| **Limitação** | **Descrição** | **Mitigação** |
|---------------|---------------|---------------|
| **🎛️ Parâmetro p0** | Sensível à escolha de p0 | Usar p0 = 0.1 a 0.2 |
| **🔗 Correlação MCMC** | Cadeias MCMC podem ser correlacionadas | Aumentar comprimento das cadeias |
| **📊 Regiões Desconexas** | Dificuldade com múltiplas regiões de falha | Algoritmos adaptativos |
| **🔄 Convergência MCMC** | MCMC pode não convergir adequadamente | Diagnósticos de convergência |
| **💻 Custo por Avaliação** | Ainda requer muitas avaliações de g(X) | Usar modelos substitutos |

---

## **6. 📚 Referências e Recursos**

### **6.1 📖 Literatura Fundamental**

1. **Au, S. K., & Beck, J. L. (2001).** *"Estimation of small failure probabilities in high dimensions by subset simulation"*. Probabilistic Engineering Mechanics, 16(4), 263-277.
   - 📘 **Artigo original:** Introdução do Subset Simulation

2. **Straub, D., & Papaioannou, I. (2015).** *"Bayesian updating with structural reliability methods"*. Journal of Engineering Mechanics, 141(3).
   - 🎯 **BUS:** Método BUS completo

3. **Au, S. K., & Beck, J. L. (2003).** *"Subset simulation and its application to seismic risk based on dynamic analysis"*. Journal of Engineering Mechanics, 129(8), 901-917.
   - 🏗️ **Aplicação:** Engenharia sísmica

4. **Papaioannou, I., et al. (2015).** *"MCMC algorithms for subset simulation"*. Probabilistic Engineering Mechanics, 41, 89-103.
   - 🔄 **MCMC:** Comparação de algoritmos MCMC

### **6.2 🌐 Recursos Práticos**

```python
# Bibliotecas Python para análise de confiabilidade

# UQpy: Uncertainty Quantification with Python
from UQpy import SubsetSimulation

# OpenCOSSAN: Open-source COmputational platform for 
# Safety, Reliability ANalysis
import opencossan

# PyRe: Python Reliability
from pyre import subset_simulation
```

### **6.3 🔗 Links Úteis**

- **ERA Group (TU München):** https://www.cee.ed.tum.de/era/software/
- **UQpy:** https://github.com/SURGroup/UQpy
- **Tutorial Subset Simulation:** https://arxiv.org/abs/1505.03506

---

## **7. 🎯 Conclusão**

### **7.1 💡 Principais Aprendizados**

Subset Simulation e BUS representam avanços fundamentais em análise de confiabilidade:

1. **Decomposição Inteligente:** Quebrar problema difícil em subproblemas tratáveis
2. **MCMC Condicional:** Amostrar eficientemente de regiões raras
3. **Integração Bayesiana:** Combinar com inferência bayesiana para atualização de modelos
4. **Eficiência:** Ordens de magnitude mais eficiente que Monte Carlo

### **7.2 🔑 Quando Usar SUS/BUS**

**✅ Cenários Ideais:**
- Análise de confiabilidade estrutural
- Eventos raros (Pf < 10⁻³)
- Função de desempenho cara de avaliar
- Atualização bayesiana com dados limitados
- Necessidade de amostras da região de falha

**❌ Cenários Problemáticos:**
- Probabilidades não muito pequenas (usar MC direto)
- Quando gradientes estão disponíveis (usar FORM/SORM)
- Regiões de falha extremamente desconexas
- Dimensionalidade muito alta (>100)

### **7.3 🌟 Mensagem Final**

> **"Subset Simulation nos ensina que problemas aparentemente intratáveis podem se tornar tratáveis através de decomposição inteligente e amostragem adaptativa."**

A combinação de Subset Simulation com inferência bayesiana (BUS) representa uma ferramenta poderosa para análise de confiabilidade e atualização de modelos em face de incerteza e dados limitados.

---

**🔗 Continue Explorando:**
- 📖 Veja também: [**Cross-Entropy Method**](../optimization/cross_entropy_method.md)
- 🎯 Relacionado: [**Dynamic Bayesian Networks**](../probabilistic_models/dynamic_bayesian_networks.md)
- 🔬 Aplicações: [**Structural Reliability Analysis**](../applications/structural_reliability.md)

**🎓 Obrigado por explorar BUS com Subset Simulation!**

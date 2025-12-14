# Regressão de Processo Gaussiano (Gaussian Process Regression - GPR)

A **Regressão de Processo Gaussiano** (Gaussian Process Regression - GPR) é um método poderoso e elegante de aprendizado de máquina não-paramétrico que fornece não apenas predições, mas também **incerteza quantificada** sobre essas predições. Fundamentado em teoria de probabilidade bayesiana, GPR é amplamente utilizado em otimização bayesiana, modelagem de sistemas complexos, e aplicações onde quantificar incerteza é crucial.

![Gaussian Process Visualization](../../images/gpr_concept.png)

Diferentemente de métodos paramétricos tradicionais que assumem uma forma funcional fixa, GPR trabalha diretamente com **distribuições sobre funções**, oferecendo flexibilidade excepcional e capacidade de modelar relações não-lineares complexas.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 O Que É um Processo Gaussiano?**

Um **Processo Gaussiano** é uma **distribuição de probabilidade sobre funções**. Formalmente:

```
Um Processo Gaussiano é uma coleção de variáveis aleatórias, 
qualquer subconjunto finito das quais possui uma distribuição 
gaussiana (normal) multivariada conjunta.
```

**Intuição:**
> "Imagine que ao invés de ter incerteza sobre um número (distribuição gaussiana univariada) ou sobre um vetor (distribuição gaussiana multivariada), temos incerteza sobre uma **função inteira**. O Processo Gaussiano nos dá uma maneira matemática rigorosa de representar essa incerteza."

### **1.2 Definição Matemática**

Um Processo Gaussiano é completamente especificado por:

1. **Função Média m(x):** Representa o valor esperado da função em cada ponto
   ```
   m(x) = E[f(x)]
   ```

2. **Função de Covariância (Kernel) k(x, x'):** Descreve como valores da função em diferentes pontos se relacionam
   ```
   k(x, x') = E[(f(x) - m(x))(f(x') - m(x'))]
   ```

**Notação:**
```
f(x) ~ GP(m(x), k(x, x'))
```

Isso significa: "a função f é distribuída como um Processo Gaussiano com média m e covariância k"

### **1.3 Por Que Processos Gaussianos?**

#### **🎲 Comparação com Métodos Tradicionais**

| **Aspecto** | **Regressão Linear** | **Redes Neurais** | **Processos Gaussianos** |
|-------------|---------------------|-------------------|--------------------------|
| **Forma da Função** | Fixa (linear) | Fixa (arquitetura) | Flexível (qualquer função) |
| **Incerteza** | Apenas nos parâmetros | Difícil de quantificar | Incerteza completa |
| **Interpretabilidade** | ✅ Alta | ❌ Baixa | ✅ Alta |
| **Dados Pequenos** | ⚠️ Limitado | ❌ Ruim | ✅✅ Excelente |
| **Custo Computacional** | ✅ Baixo | ⚠️ Médio | ❌ Alto (O(n³)) |

#### **🌟 Principais Vantagens de GPR**

1. **📊 Quantificação de Incerteza:** Fornece intervalos de confiança naturalmente
2. **🎯 Não-Paramétrico:** Não assume forma funcional específica
3. **🧮 Fundamentação Bayesiana:** Incorpora conhecimento prévio de forma principled
4. **🔧 Flexível:** Através da escolha do kernel
5. **📈 Interpretável:** Comportamento do modelo é compreensível

---

## **2. 🔧 Matemática dos Processos Gaussianos**

### **2.1 Prior de Processo Gaussiano**

Antes de observar dados, especificamos nossas crenças iniciais sobre a função:

```python
# Prior: função distribuída como GP
f(x) ~ GP(0, k(x, x'))

# Onde:
# - Média 0 (comum assumir zero, dados podem estar centrados)
# - Kernel k define correlação entre pontos
```

**Interpretação Visual:**
```
    f(x)
     │     ╱╲    ╱╲
     │    ╱  ╲  ╱  ╲
     │   ╱    ╲╱    ╲
     │  ╱            ╲
     └──────────────────── x
     
     Cada possível função tem uma probabilidade
     Antes de ver dados, todas são igualmente plausíveis
```

### **2.2 Funções de Covariância (Kernels)**

O kernel é o **coração** do GP, definindo quais tipos de funções são prováveis.

#### **📐 Kernel RBF (Radial Basis Function) / Squared Exponential**

O kernel mais popular, suave e infinitamente diferenciável:

```python
def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Kernel RBF (Squared Exponential)
    
    Args:
        x1, x2: Pontos de entrada
        length_scale: Controla a "largura" da correlação
        variance: Amplitude do sinal
    
    Returns:
        Covariância entre x1 e x2
    """
    distance = np.linalg.norm(x1 - x2)
    return variance * np.exp(-distance**2 / (2 * length_scale**2))
```

**Características:**
- ✅ **Suavidade infinita:** Funções são muito suaves
- ✅ **Decaimento exponencial:** Correlação diminui rapidamente com distância
- 🎯 **Uso:** Quando esperamos funções suaves

**Hiperparâmetros:**
```
length_scale (ℓ):
  - Pequeno → função varia rapidamente
  - Grande → função varia lentamente

variance (σ²):
  - Controla amplitude vertical das funções
```

#### **📊 Kernel Matérn**

Mais flexível que RBF, controla suavidade:

```python
def matern_kernel(x1, x2, length_scale=1.0, nu=1.5):
    """
    Kernel Matérn
    
    Args:
        nu: Parâmetro de suavidade
            nu = 0.5: Não diferenciável (similar a Exponential)
            nu = 1.5: Uma vez diferenciável
            nu = 2.5: Duas vezes diferenciável
            nu → ∞: Converge para RBF
    """
    from scipy.special import kv, gamma
    
    distance = np.linalg.norm(x1 - x2)
    if distance == 0:
        return 1.0
    
    sqrt_term = np.sqrt(2 * nu) * distance / length_scale
    
    coefficient = (2 ** (1 - nu)) / gamma(nu)
    bessel_term = kv(nu, sqrt_term)
    
    return coefficient * (sqrt_term ** nu) * bessel_term
```

**Quando usar:**
- 📈 **nu = 0.5:** Dados ruidosos, função não precisa ser suave
- 📊 **nu = 1.5:** Balanço entre suavidade e flexibilidade (padrão)
- 🎯 **nu = 2.5:** Funções mais suaves
- ✨ **nu → ∞:** Máxima suavidade (equivalente a RBF)

#### **📉 Kernel Linear**

Para relações lineares:

```python
def linear_kernel(x1, x2, variance=1.0, offset=0.0):
    """
    Kernel Linear: k(x1, x2) = σ² (x1 · x2 + c)
    """
    return variance * (np.dot(x1, x2) + offset)
```

#### **🔄 Kernel Periódico**

Para padrões que se repetem:

```python
def periodic_kernel(x1, x2, period=1.0, length_scale=1.0):
    """
    Kernel Periódico: Para funções com padrão repetitivo
    """
    distance = np.abs(x1 - x2)
    sin_term = np.sin(np.pi * distance / period)
    return np.exp(-2 * (sin_term / length_scale) ** 2)
```

#### **➕ Composição de Kernels**

Kernels podem ser combinados para criar priors mais expressivos:

```python
# Soma: Captura múltiplas características
k_total = k_rbf + k_periodic  # Tendência suave + padrão periódico

# Produto: Modulação
k_modulated = k_rbf * k_periodic  # Padrão periódico com envelope suave

# Exemplo prático
def combined_kernel(x1, x2):
    """
    Kernel que captura:
    - Tendência linear
    - Variação suave
    - Padrão periódico
    """
    k_lin = linear_kernel(x1, x2, variance=0.5)
    k_rbf = rbf_kernel(x1, x2, length_scale=1.0)
    k_per = periodic_kernel(x1, x2, period=2.0)
    
    return k_lin + k_rbf + 0.5 * k_per
```

### **2.3 Posterior de Processo Gaussiano**

Após observar dados de treinamento **D = {(x₁, y₁), ..., (xₙ, yₙ)}**, atualizamos nossas crenças:

**Dados:**
```python
X_train = [x₁, x₂, ..., xₙ]  # Entradas de treino
y_train = [y₁, y₂, ..., yₙ]  # Saídas observadas
```

**Modelo:**
```
yᵢ = f(xᵢ) + ε
onde ε ~ N(0, σₙ²) é ruído gaussiano
```

**Predição em novo ponto x*:**

A distribuição posterior é também gaussiana:

```
f* | X, y, x* ~ N(μ*, σ²*)

onde:

Média posterior (predição):
μ* = K(x*, X) [K(X, X) + σₙ²I]⁻¹ y

Variância posterior (incerteza):
σ²* = K(x*, x*) - K(x*, X) [K(X, X) + σₙ²I]⁻¹ K(X, x*)
```

**Notação:**
- **K(X, X):** Matriz de covariância entre pontos de treino (n × n)
- **K(x*, X):** Vetor de covariância entre ponto teste e treino (1 × n)
- **K(x*, x*):** Covariância do ponto teste consigo mesmo (escalar)
- **σₙ²:** Variância do ruído
- **I:** Matriz identidade

---

## **3. 💻 Implementação de Regressão com Processos Gaussianos**

### **3.1 🔧 Implementação Básica**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, cho_solve
from scipy.spatial.distance import cdist

class GaussianProcessRegressor:
    """
    Implementação de Regressão com Processos Gaussianos
    """
    
    def __init__(self, kernel='rbf', length_scale=1.0, variance=1.0, 
                 noise=1e-5):
        """
        Args:
            kernel: Tipo de kernel ('rbf', 'matern', 'linear', 'periodic')
            length_scale: Parâmetro de escala do kernel
            variance: Variância do sinal
            noise: Variância do ruído
        """
        self.kernel_name = kernel
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        
        self.X_train = None
        self.y_train = None
        self.L = None  # Decomposição de Cholesky
        self.alpha = None  # Pesos para predição
    
    def kernel(self, X1, X2):
        """
        Calcula matriz de covariância entre conjuntos de pontos
        
        Args:
            X1: Matriz (n1, d)
            X2: Matriz (n2, d)
        
        Returns:
            K: Matriz de covariância (n1, n2)
        """
        if self.kernel_name == 'rbf':
            # RBF (Squared Exponential) Kernel
            dists = cdist(X1, X2, metric='sqeuclidean')
            K = self.variance * np.exp(-dists / (2 * self.length_scale**2))
        
        elif self.kernel_name == 'matern':
            # Matérn kernel (nu = 1.5)
            from scipy.special import kv
            dists = cdist(X1, X2, metric='euclidean')
            sqrt_3_dists = np.sqrt(3) * dists / self.length_scale
            K = self.variance * (1 + sqrt_3_dists) * np.exp(-sqrt_3_dists)
        
        elif self.kernel_name == 'linear':
            # Linear kernel
            K = self.variance * (X1 @ X2.T)
        
        elif self.kernel_name == 'periodic':
            # Periodic kernel
            dists = cdist(X1, X2, metric='euclidean')
            sin_term = np.sin(np.pi * dists / self.length_scale)
            K = self.variance * np.exp(-2 * (sin_term ** 2))
        
        else:
            raise ValueError(f"Kernel desconhecido: {self.kernel_name}")
        
        return K
    
    def fit(self, X, y):
        """
        Treinar GP com dados observados
        
        Args:
            X: Entradas (n, d)
            y: Saídas (n,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y).flatten()
        
        # Calcular matriz de covariância
        K = self.kernel(self.X_train, self.X_train)
        
        # Adicionar ruído na diagonal
        K_y = K + self.noise * np.eye(len(self.X_train))
        
        # Decomposição de Cholesky para estabilidade numérica
        try:
            self.L = cholesky(K_y, lower=True)
        except np.linalg.LinAlgError:
            # Se falhar, adicionar mais regularização
            K_y += 1e-6 * np.eye(len(self.X_train))
            self.L = cholesky(K_y, lower=True)
        
        # Calcular alpha = K⁻¹ y
        self.alpha = cho_solve((self.L, True), self.y_train)
        
        # Calcular log-verossimilhança marginal (para seleção de hiperparâmetros)
        self.log_marginal_likelihood_ = self._compute_log_marginal_likelihood()
    
    def predict(self, X_test, return_std=False, return_cov=False):
        """
        Fazer predições em novos pontos
        
        Args:
            X_test: Pontos de teste (m, d)
            return_std: Se True, retorna desvio padrão
            return_cov: Se True, retorna matriz de covariância completa
        
        Returns:
            mean: Média posterior (m,)
            std: Desvio padrão (m,) [se return_std=True]
            cov: Covariância (m, m) [se return_cov=True]
        """
        X_test = np.array(X_test)
        
        # Covariância entre teste e treino
        K_star = self.kernel(X_test, self.X_train)
        
        # Média posterior
        mean = K_star @ self.alpha
        
        if not (return_std or return_cov):
            return mean
        
        # Covariância teste-teste
        K_star_star = self.kernel(X_test, X_test)
        
        # Resolver sistema linear para v = L⁻¹ K*
        v = cho_solve((self.L, True), K_star.T)
        
        # Variância posterior
        cov = K_star_star - K_star @ v
        
        if return_cov:
            return mean, cov
        
        # Desvio padrão (diagonal da covariância)
        std = np.sqrt(np.maximum(np.diag(cov), 0))
        
        return mean, std
    
    def sample_prior(self, X, n_samples=5):
        """
        Amostrar funções do prior
        
        Args:
            X: Pontos onde amostrar (n, d)
            n_samples: Número de funções a amostrar
        
        Returns:
            samples: Amostras do prior (n_samples, n)
        """
        X = np.array(X)
        K = self.kernel(X, X)
        
        # Adicionar ruído para estabilidade numérica
        K += 1e-8 * np.eye(len(X))
        
        # Amostrar de N(0, K)
        samples = np.random.multivariate_normal(
            mean=np.zeros(len(X)),
            cov=K,
            size=n_samples
        )
        
        return samples
    
    def sample_posterior(self, X, n_samples=5):
        """
        Amostrar funções do posterior (após ver dados)
        
        Args:
            X: Pontos onde amostrar (n, d)
            n_samples: Número de funções a amostrar
        
        Returns:
            samples: Amostras do posterior (n_samples, n)
        """
        mean, cov = self.predict(X, return_cov=True)
        
        # Adicionar pequeno ruído na diagonal para estabilidade
        cov += 1e-8 * np.eye(len(X))
        
        samples = np.random.multivariate_normal(mean, cov, size=n_samples)
        
        return samples
    
    def _compute_log_marginal_likelihood(self):
        """
        Calcular log-verossimilhança marginal log p(y|X)
        Usado para otimizar hiperparâmetros
        """
        n = len(self.y_train)
        
        # Termo 1: -0.5 * y^T K^{-1} y
        term1 = -0.5 * self.y_train @ self.alpha
        
        # Termo 2: -0.5 * log|K|
        term2 = -np.sum(np.log(np.diag(self.L)))
        
        # Termo 3: -n/2 * log(2π)
        term3 = -0.5 * n * np.log(2 * np.pi)
        
        return term1 + term2 + term3
    
    def plot_fit(self, X_test=None, n_samples=3, figsize=(12, 5)):
        """
        Visualizar GP ajustado
        """
        if self.X_train is None:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        
        # Gerar pontos de teste se não fornecidos
        if X_test is None:
            x_min = self.X_train.min() - 1
            x_max = self.X_train.max() + 1
            X_test = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        
        # Predições
        mean, std = self.predict(X_test, return_std=True)
        
        # Amostras do posterior
        samples = self.sample_posterior(X_test, n_samples=n_samples)
        
        # Plotar
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Média e intervalo de confiança
        ax1.plot(X_test, mean, 'b-', linewidth=2, label='Média Posterior')
        ax1.fill_between(
            X_test.flatten(),
            mean - 2*std,
            mean + 2*std,
            alpha=0.3,
            color='blue',
            label='95% Intervalo de Confiança'
        )
        ax1.scatter(
            self.X_train.flatten(),
            self.y_train,
            c='red',
            s=100,
            zorder=10,
            edgecolors='black',
            label='Dados de Treino'
        )
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Predição do Processo Gaussiano')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Amostras do posterior
        for i, sample in enumerate(samples):
            ax2.plot(X_test, sample, alpha=0.7, 
                    label=f'Amostra {i+1}' if i < 3 else None)
        ax2.scatter(
            self.X_train.flatten(),
            self.y_train,
            c='red',
            s=100,
            zorder=10,
            edgecolors='black',
            label='Dados de Treino'
        )
        ax2.set_xlabel('x')
        ax2.set_ylabel('f(x)')
        ax2.set_title('Amostras do Posterior')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Exemplo de uso básico
if __name__ == "__main__":
    # Função verdadeira (desconhecida)
    def true_function(x):
        return np.sin(x) + 0.5 * np.cos(2*x)
    
    # Gerar dados de treino
    np.random.seed(42)
    X_train = np.array([1, 3, 5, 6, 8]).reshape(-1, 1)
    y_train = true_function(X_train) + np.random.normal(0, 0.1, X_train.shape)
    
    # Treinar GP
    gp = GaussianProcessRegressor(
        kernel='rbf',
        length_scale=1.0,
        variance=1.0,
        noise=0.1
    )
    gp.fit(X_train, y_train)
    
    # Visualizar
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    gp.plot_fit(X_test, n_samples=5)
    
    print(f"Log-verossimilhança marginal: {gp.log_marginal_likelihood_:.4f}")
```

### **3.2 🎛️ Otimização de Hiperparâmetros**

Os hiperparâmetros do kernel (length_scale, variance, noise) podem ser otimizados maximizando a log-verossimilhança marginal:

```python
from scipy.optimize import minimize

class GPRWithHyperparameterOptimization(GaussianProcessRegressor):
    """
    GP com otimização automática de hiperparâmetros
    """
    
    def fit(self, X, y, optimize=True):
        """
        Treinar GP e opcionalmente otimizar hiperparâmetros
        """
        if not optimize:
            return super().fit(X, y)
        
        # Valores iniciais
        initial_params = np.array([
            np.log(self.length_scale),
            np.log(self.variance),
            np.log(self.noise)
        ])
        
        # Função objetivo: negativo da log-verossimilhança
        def objective(params):
            self.length_scale = np.exp(params[0])
            self.variance = np.exp(params[1])
            self.noise = np.exp(params[2])
            
            # Fit com parâmetros atuais
            super(GPRWithHyperparameterOptimization, self).fit(X, y)
            
            # Retornar negativo (para minimização)
            return -self.log_marginal_likelihood_
        
        # Gradiente da log-verossimilhança (opcional, para convergência mais rápida)
        def gradient(params):
            self.length_scale = np.exp(params[0])
            self.variance = np.exp(params[1])
            self.noise = np.exp(params[2])
            
            # Calcular gradientes numericamente
            epsilon = 1e-5
            grad = np.zeros_like(params)
            
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += epsilon
                
                params_minus = params.copy()
                params_minus[i] -= epsilon
                
                loss_plus = objective(params_plus)
                loss_minus = objective(params_minus)
                
                grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            return grad
        
        # Otimizar
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': 100, 'disp': False}
        )
        
        # Usar melhores parâmetros
        best_params = result.x
        self.length_scale = np.exp(best_params[0])
        self.variance = np.exp(best_params[1])
        self.noise = np.exp(best_params[2])
        
        # Fit final
        super(GPRWithHyperparameterOptimization, self).fit(X, y)
        
        print(f"Hiperparâmetros otimizados:")
        print(f"  length_scale: {self.length_scale:.4f}")
        print(f"  variance: {self.variance:.4f}")
        print(f"  noise: {self.noise:.4f}")
        print(f"  log-likelihood: {self.log_marginal_likelihood_:.4f}")

# Exemplo
if __name__ == "__main__":
    X_train = np.array([1, 3, 5, 6, 8]).reshape(-1, 1)
    y_train = np.sin(X_train).flatten() + np.random.normal(0, 0.1, len(X_train))
    
    gp_opt = GPRWithHyperparameterOptimization()
    gp_opt.fit(X_train, y_train, optimize=True)
    
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    gp_opt.plot_fit(X_test)
```

---

## **4. 📊 Aplicações de Processos Gaussianos**

### **4.1 🎯 Otimização Bayesiana**

GPR é fundamental para otimização bayesiana, permitindo otimizar funções custosas de avaliar:

```python
class BayesianOptimization:
    """
    Otimização Bayesiana usando Processos Gaussianos
    """
    
    def __init__(self, objective_function, bounds, kernel='rbf'):
        """
        Args:
            objective_function: Função a otimizar (cara de avaliar)
            bounds: Lista de tuplas [(min, max) para cada dimensão]
            kernel: Kernel do GP
        """
        self.objective = objective_function
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        
        self.gp = GPRWithHyperparameterOptimization(kernel=kernel)
        
        self.X_observed = []
        self.y_observed = []
    
    def acquisition_function(self, X, method='ei', xi=0.01):
        """
        Calcular função de aquisição
        
        Args:
            X: Pontos candidatos
            method: 'ei' (Expected Improvement) ou 'ucb' (Upper Confidence Bound)
            xi: Parâmetro de exploração
        
        Returns:
            Valores da função de aquisição
        """
        mean, std = self.gp.predict(X, return_std=True)
        
        if method == 'ei':
            # Expected Improvement
            if len(self.y_observed) == 0:
                return np.ones(len(X))
            
            best_y = np.max(self.y_observed)
            
            with np.errstate(divide='warn'):
                improvement = mean - best_y - xi
                Z = improvement / std
                ei = improvement * self._normal_cdf(Z) + std * self._normal_pdf(Z)
                ei[std == 0.0] = 0.0
            
            return ei
        
        elif method == 'ucb':
            # Upper Confidence Bound
            kappa = 2.0  # Parâmetro de exploração
            return mean + kappa * std
        
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
    def _normal_cdf(self, x):
        """CDF da distribuição normal padrão"""
        from scipy.stats import norm
        return norm.cdf(x)
    
    def _normal_pdf(self, x):
        """PDF da distribuição normal padrão"""
        from scipy.stats import norm
        return norm.pdf(x)
    
    def suggest_next_point(self, n_candidates=1000, method='ei'):
        """
        Sugerir próximo ponto a avaliar
        """
        # Gerar pontos candidatos
        candidates = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_candidates, self.dim)
        )
        
        # Calcular função de aquisição
        acq_values = self.acquisition_function(candidates, method=method)
        
        # Escolher melhor candidato
        best_idx = np.argmax(acq_values)
        next_point = candidates[best_idx]
        
        return next_point
    
    def optimize(self, n_iterations=20, initial_samples=5, verbose=True):
        """
        Executar otimização bayesiana
        """
        # Amostragem inicial aleatória
        if len(self.X_observed) < initial_samples:
            for _ in range(initial_samples - len(self.X_observed)):
                x = np.random.uniform(
                    self.bounds[:, 0],
                    self.bounds[:, 1]
                )
                y = self.objective(x)
                
                self.X_observed.append(x)
                self.y_observed.append(y)
                
                if verbose:
                    print(f"Amostra inicial: x={x}, y={y:.4f}")
        
        # Converter para arrays
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Loop principal de otimização
        for iteration in range(n_iterations):
            # Treinar GP
            self.gp.fit(X, y, optimize=True)
            
            # Sugerir próximo ponto
            next_x = self.suggest_next_point()
            next_y = self.objective(next_x)
            
            # Adicionar observação
            self.X_observed.append(next_x)
            self.y_observed.append(next_y)
            
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            
            if verbose:
                best_y_so_far = np.max(y)
                print(f"Iteração {iteration+1}: "
                      f"x={next_x}, y={next_y:.4f}, "
                      f"Melhor até agora={best_y_so_far:.4f}")
        
        # Retornar melhor ponto encontrado
        best_idx = np.argmax(self.y_observed)
        best_x = self.X_observed[best_idx]
        best_y = self.y_observed[best_idx]
        
        return best_x, best_y
    
    def plot_optimization(self, true_function=None):
        """
        Visualizar processo de otimização (apenas para 1D)
        """
        if self.dim != 1:
            print("Visualização disponível apenas para problemas 1D")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Pontos de teste
        X_test = np.linspace(
            self.bounds[0, 0],
            self.bounds[0, 1],
            200
        ).reshape(-1, 1)
        
        # Predição do GP
        mean, std = self.gp.predict(X_test, return_std=True)
        
        # Plot 1: GP e pontos observados
        ax1.plot(X_test, mean, 'b-', label='Média GP')
        ax1.fill_between(
            X_test.flatten(),
            mean - 2*std,
            mean + 2*std,
            alpha=0.3,
            label='95% IC'
        )
        
        if true_function is not None:
            y_true = [true_function(x) for x in X_test]
            ax1.plot(X_test, y_true, 'g--', label='Função Verdadeira', alpha=0.5)
        
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        ax1.scatter(X_obs, y_obs, c='red', s=100, zorder=10, 
                   label='Observações', edgecolors='black')
        
        best_idx = np.argmax(y_obs)
        ax1.scatter(X_obs[best_idx], y_obs[best_idx], 
                   c='gold', s=200, marker='*', zorder=11,
                   label='Melhor', edgecolors='black')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.set_title('Modelo do Processo Gaussiano')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Função de aquisição
        acq = self.acquisition_function(X_test, method='ei')
        ax2.plot(X_test, acq, 'r-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('Expected Improvement')
        ax2.set_title('Função de Aquisição')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Exemplo: Otimizar função complexa
if __name__ == "__main__":
    # Função objetivo (cara de avaliar, com múltiplos mínimos/máximos)
    def objective(x):
        """Função com múltiplos ótimos locais"""
        return -((x - 2)**2 * np.sin(5*x) + 0.1*x)
    
    # Executar otimização bayesiana
    bounds = [(0, 5)]
    bo = BayesianOptimization(objective, bounds)
    
    best_x, best_y = bo.optimize(n_iterations=15, initial_samples=3, verbose=True)
    
    print(f"\nMelhor ponto encontrado:")
    print(f"  x = {best_x}")
    print(f"  f(x) = {best_y:.6f}")
    
    # Visualizar
    bo.plot_optimization(true_function=objective)
```

### **4.2 🌊 Interpolação e Suavização de Dados**

GPR é excelente para interpolar dados esparsos com incerteza:

```python
class GPInterpolator:
    """
    Interpolação de dados usando Processos Gaussianos
    """
    
    def __init__(self, smoothness='medium'):
        """
        Args:
            smoothness: 'low', 'medium', 'high'
        """
        if smoothness == 'low':
            kernel = 'matern'
            nu = 0.5
        elif smoothness == 'medium':
            kernel = 'matern'
            nu = 1.5
        else:  # high
            kernel = 'rbf'
        
        self.gp = GPRWithHyperparameterOptimization(kernel=kernel)
    
    def interpolate(self, X, y, X_new, confidence_level=0.95):
        """
        Interpolar dados com intervalos de confiança
        
        Returns:
            y_new: Valores interpolados
            lower: Limite inferior do intervalo
            upper: Limite superior do intervalo
        """
        # Treinar GP
        self.gp.fit(X, y, optimize=True)
        
        # Predizer
        mean, std = self.gp.predict(X_new, return_std=True)
        
        # Calcular intervalos de confiança
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return mean, lower, upper

# Exemplo: Interpolar dados climáticos
if __name__ == "__main__":
    # Dados de temperatura ao longo do ano (esparsos)
    meses = np.array([1, 3, 5, 7, 9, 11]).reshape(-1, 1)  # Jan, Mar, Mai, ...
    temperaturas = np.array([10, 15, 22, 28, 20, 12])
    
    # Criar interpolador
    interp = GPInterpolator(smoothness='medium')
    
    # Interpolar para todos os dias do ano
    todos_meses = np.linspace(1, 12, 365).reshape(-1, 1)
    temp_interpolada, lower, upper = interp.interpolate(
        meses, temperaturas, todos_meses, confidence_level=0.95
    )
    
    # Visualizar
    plt.figure(figsize=(12, 6))
    plt.plot(todos_meses, temp_interpolada, 'b-', label='Interpolação GP')
    plt.fill_between(
        todos_meses.flatten(), lower, upper,
        alpha=0.3, label='95% Intervalo de Confiança'
    )
    plt.scatter(meses, temperaturas, c='red', s=100, 
               label='Medições', zorder=10, edgecolors='black')
    plt.xlabel('Mês')
    plt.ylabel('Temperatura (°C)')
    plt.title('Interpolação de Dados de Temperatura com GP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### **4.3 🔬 Análise de Sensibilidade**

GPR pode ser usado para análise de sensibilidade em sistemas complexos:

```python
class SensitivityAnalysis:
    """
    Análise de sensibilidade usando Processos Gaussianos
    """
    
    def __init__(self, function, input_names, bounds):
        """
        Args:
            function: Função a analisar
            input_names: Nomes das variáveis de entrada
            bounds: Limites de cada entrada [(min, max), ...]
        """
        self.function = function
        self.input_names = input_names
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        
        self.gp = GPRWithHyperparameterOptimization(kernel='rbf')
    
    def sample_function(self, n_samples=100):
        """
        Amostrar função em pontos aleatórios
        """
        X = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_samples, self.dim)
        )
        y = np.array([self.function(x) for x in X])
        
        return X, y
    
    def fit_surrogate(self, n_samples=100):
        """
        Ajustar modelo substituto (surrogate)
        """
        X, y = self.sample_function(n_samples)
        self.gp.fit(X, y, optimize=True)
    
    def sobol_indices(self, n_samples=1000):
        """
        Calcular índices de Sobol (sensibilidade global)
        
        Returns:
            first_order: Índices de primeira ordem (efeito individual)
            total_order: Índices totais (incluindo interações)
        """
        # Gerar amostras
        A = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_samples, self.dim)
        )
        B = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_samples, self.dim)
        )
        
        # Predições do GP
        f_A = self.gp.predict(A)
        f_B = self.gp.predict(B)
        
        # Variância total
        var_total = np.var(np.concatenate([f_A, f_B]))
        
        # Índices de primeira ordem
        first_order = np.zeros(self.dim)
        total_order = np.zeros(self.dim)
        
        for i in range(self.dim):
            # Criar matriz C_i (A com coluna i de B)
            C_i = A.copy()
            C_i[:, i] = B[:, i]
            f_C_i = self.gp.predict(C_i)
            
            # Primeira ordem: V_i = Var(E(Y|X_i))
            first_order[i] = np.mean(f_A * f_C_i) - np.mean(f_A)**2
            first_order[i] /= var_total
            
            # Total: VT_i = E(Var(Y|X_~i))
            total_order[i] = 1 - (np.mean(f_B * f_C_i) - np.mean(f_B)**2) / var_total
        
        return first_order, total_order
    
    def plot_sensitivity(self):
        """
        Visualizar análise de sensibilidade
        """
        first, total = self.sobol_indices()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(self.dim)
        width = 0.35
        
        ax.bar(x - width/2, first, width, label='Primeira Ordem', alpha=0.8)
        ax.bar(x + width/2, total, width, label='Total', alpha=0.8)
        
        ax.set_xlabel('Variáveis de Entrada')
        ax.set_ylabel('Índice de Sensibilidade')
        ax.set_title('Análise de Sensibilidade Global (Índices de Sobol)')
        ax.set_xticks(x)
        ax.set_xticklabels(self.input_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Imprimir resultados
        print("\nÍndices de Sensibilidade:")
        print("-" * 50)
        for i, name in enumerate(self.input_names):
            print(f"{name}:")
            print(f"  Primeira ordem: {first[i]:.4f}")
            print(f"  Total:          {total[i]:.4f}")

# Exemplo: Analisar função complexa
if __name__ == "__main__":
    # Função com interações entre variáveis
    def complex_function(x):
        return x[0]**2 + 2*x[1] + x[0]*x[1] + 0.5*x[2]**2
    
    # Configurar análise
    sa = SensitivityAnalysis(
        function=complex_function,
        input_names=['x1', 'x2', 'x3'],
        bounds=[(-2, 2), (-2, 2), (-2, 2)]
    )
    
    # Ajustar modelo substituto
    print("Ajustando modelo substituto...")
    sa.fit_surrogate(n_samples=200)
    
    # Análise de sensibilidade
    print("\nCalculando índices de sensibilidade...")
    sa.plot_sensitivity()
```

---

## **5. ⚖️ Vantagens e Limitações**

### **5.1 ✅ Vantagens**

| **Vantagem** | **Descrição** | **Aplicação** |
|--------------|---------------|---------------|
| **📊 Quantificação de Incerteza** | Fornece intervalos de confiança naturalmente | Decisões críticas |
| **🎯 Não-Paramétrico** | Não assume forma funcional específica | Funções desconhecidas |
| **🧮 Fundamentação Bayesiana** | Incorpora conhecimento prévio | Small data scenarios |
| **🔧 Flexível via Kernels** | Adaptável a diferentes tipos de dados | Diversos domínios |
| **📈 Interpretável** | Comportamento compreensível | Análise científica |
| **🎲 Lida com Ruído** | Modela ruído explicitamente | Dados ruidosos |

### **5.2 ❌ Limitações**

| **Limitação** | **Descrição** | **Mitigação** |
|---------------|---------------|---------------|
| **💻 Custo Computacional** | O(n³) para treino, O(n²) para predição | Métodos esparsos, inducing points |
| **📊 Escalabilidade** | Difícil com n > 10,000 | GP esparsos (SVGP, SGPR) |
| **🎛️ Escolha do Kernel** | Requer conhecimento do problema | Testar múltiplos kernels |
| **📈 Alta Dimensionalidade** | Performance degrada em dim >> 10 | Redução de dimensionalidade |
| **🔧 Hiperparâmetros** | Sensível a configuração | Otimização via likelihood |

### **5.3 🆚 Comparação com Outros Métodos**

```
Critério                  │ GPR   │ RF    │ SVM   │ NN    
──────────────────────────┼───────┼───────┼───────┼───────
📊 Incerteza              │ ✅✅  │ ⚠️    │ ❌    │ ⚠️    
🎯 Small Data             │ ✅✅  │ ⚠️    │ ✅    │ ❌    
⚡ Velocidade Treino      │ ❌    │ ✅✅  │ ⚠️    │ ⚠️    
⚡ Velocidade Predição    │ ⚠️    │ ✅✅  │ ✅    │ ✅✅  
📈 Escalabilidade         │ ❌    │ ✅✅  │ ⚠️    │ ✅    
🧠 Interpretabilidade     │ ✅✅  │ ⚠️    │ ⚠️    │ ❌    
🔧 Facilidade de Uso      │ ⚠️    │ ✅✅  │ ⚠️    │ ⚠️    
```

---

## **6. 🚀 Extensões e Variantes**

### **6.1 📊 GP Esparsos (Sparse GP)**

Para escalar a grandes datasets:

```python
class SparseGP:
    """
    Processo Gaussiano Esparso usando inducing points
    """
    
    def __init__(self, n_inducing=50, kernel='rbf'):
        self.n_inducing = n_inducing
        self.kernel_name = kernel
        self.inducing_points = None
    
    def fit(self, X, y):
        """
        Treinar GP esparso
        """
        # Selecionar pontos indutores (várias estratégias possíveis)
        if len(X) <= self.n_inducing:
            self.inducing_points = X
        else:
            # Estratégia 1: K-means
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_inducing, random_state=42)
            kmeans.fit(X)
            self.inducing_points = kmeans.cluster_centers_
        
        # Calcular matrizes relevantes
        # Kuu: Covariância entre inducing points
        # Kuf: Covariância entre inducing e dados
        # Implementação completa requer álgebra matricial cuidadosa
        pass
```

### **6.2 🌐 GP Multi-tarefa**

Para aprender múltiplas saídas relacionadas:

```python
class MultiTaskGP:
    """
    Processo Gaussiano Multi-tarefa
    """
    
    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        # Kernel que modela correlação entre tarefas
        pass
```

### **6.3 🔄 GP Profundo (Deep GP)**

Composição de múltiplas camadas de GPs:

```python
class DeepGP:
    """
    Deep Gaussian Process - múltiplas camadas de GPs
    """
    
    def __init__(self, layer_dims):
        """
        Args:
            layer_dims: Lista com dimensões de cada camada
        """
        self.layers = [GaussianProcessRegressor() 
                      for _ in range(len(layer_dims) - 1)]
```

---

## **7. 📚 Referências e Recursos**

### **7.1 📖 Literatura Fundamental**

#### **Livros**
1. **Rasmussen, C. E., & Williams, C. K. I. (2006).** *Gaussian Processes for Machine Learning*. MIT Press.
   - 📘 **Bíblia do GP:** Referência definitiva
   - 🆓 **Disponível online:** [gaussianprocess.org/gpml](http://gaussianprocess.org/gpml)

2. **Murphy, K. P. (2022).** *Probabilistic Machine Learning: Advanced Topics*. MIT Press.
   - 📊 **Capítulo sobre GP:** Perspectiva moderna
   
3. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer.
   - 🎓 **Capítulo 6:** Introdução acessível a GPs

#### **Artigos Fundamentais**
4. **Neal, R. M. (1996).** *Bayesian Learning for Neural Networks*. Springer.
   - 🧠 **Conexão:** GPs como limite de redes neurais

5. **Titsias, M. (2009).** *"Variational learning of inducing variables in sparse Gaussian processes"*. AISTATS.
   - ⚡ **GP Esparsos:** Métodos escaláveis

### **7.2 🌐 Recursos Práticos**

#### **Bibliotecas Python**
```python
# GPy: Framework completo para GPs
import GPy
model = GPy.models.GPRegression(X, y)

# scikit-learn: Implementação básica
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# GPflow: GPs em TensorFlow (escalável)
import gpflow
model = gpflow.models.GPR(data, kernel)

# PyTorch GPyTorch: GPs em PyTorch
import gpytorch
```

#### **Tutoriais Online**
- 📹 **Nando de Freitas:** GP Lectures no YouTube
- 📝 **Distill.pub:** Visualizações interativas de GPs
- 🎓 **Coursera:** Machine Learning by Andrew Ng (módulo sobre GPs)

### **7.3 🔗 Links Úteis**

- **Site Oficial:** [gaussianprocess.org](http://gaussianprocess.org)
- **Visualizador Interativo:** [GP Playground](https://chi-feng.github.io/gp-demo/)
- **Comunidade:** Reddit r/MachineLearning, Stack Overflow

---

## **8. 🎯 Conclusão**

### **8.1 💡 Principais Aprendizados**

Processos Gaussianos representam uma abordagem **elegante e principled** para regressão e predição:

1. **📊 Distribuições sobre Funções:** GP é uma distribuição de probabilidade sobre funções inteiras
2. **🎯 Incerteza Quantificada:** Fornece não apenas predições, mas confiança nelas
3. **🧮 Fundamentação Bayesiana:** Incorpora conhecimento prévio de forma natural
4. **🔧 Flexibilidade via Kernels:** Kernels permitem expressar diferentes suposições

### **8.2 🔑 Quando Usar GPR**

#### **✅ Cenários Ideais:**
- Dados pequenos a médios (n < 10,000)
- Quando incerteza é crítica
- Otimização de funções custosas (Bayesian Optimization)
- Modelagem científica com interpretabilidade
- Incorporar conhecimento do domínio via kernels

#### **❌ Cenários Problemáticos:**
- Datasets muito grandes (n > 50,000)
- Quando apenas predições pontuais são necessárias
- Recursos computacionais muito limitados
- Alta dimensionalidade sem estrutura

### **8.3 🌟 Mensagem Final**

Processos Gaussianos nos ensinam uma lição fundamental sobre modelagem:

> **"Modelar não apenas o que sabemos, mas também o que NÃO sabemos, é tão importante quanto a predição em si."**

A capacidade de quantificar incerteza torna GPR invaluável em aplicações críticas onde decisões devem considerar não apenas a melhor estimativa, mas também o risco associado.

---

**🔗 Continue Explorando:**
- 📖 Veja também: [**Cross-Entropy Method**](../optimization/cross_entropy_method.md)
- 🎯 Próximo: [**Dynamic Bayesian Networks**](../probabilistic_models/dynamic_bayesian_networks.md)
- 🔬 Relacionado: [**Bayesian Optimization**](../optimization/bayesian_optimization.md)

**🎓 Obrigado por explorar Processos Gaussianos!**

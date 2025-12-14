# Redes Bayesianas Dinâmicas (Dynamic Bayesian Networks - DBN)

**Redes Bayesianas Dinâmicas** (Dynamic Bayesian Networks - DBN) são modelos probabilísticos gráficos que representam processos estocásticos temporais, estendendo as Redes Bayesianas clássicas para modelar sistemas que evoluem ao longo do tempo. DBNs são ferramentas poderosas para modelagem, inferência e predição em domínios onde a estrutura temporal e as dependências causais são fundamentais.

![Dynamic Bayesian Network Concept](../../images/dbn_concept.png)

Amplamente utilizadas em reconhecimento de fala, bioinformática, robótica, finanças e diagnóstico de falhas, DBNs oferecem uma framework principled para raciocínio probabilístico temporal, combinando teoria dos grafos, probabilidade e aprendizado de máquina.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 O Que São Redes Bayesianas?**

Antes de entender DBNs, precisamos compreender **Redes Bayesianas** (Bayesian Networks - BN):

**Definição:**
> Uma Rede Bayesiana é um modelo gráfico probabilístico que representa um conjunto de variáveis aleatórias e suas dependências condicionais através de um grafo acíclico dirigido (DAG - Directed Acyclic Graph).

**Componentes:**
1. **Nós:** Representam variáveis aleatórias
2. **Arestas:** Representam dependências probabilísticas diretas
3. **Tabelas de Probabilidade Condicional (CPTs):** Quantificam as dependências

**Exemplo Simples:**
```
        Chuva
          ↓
      Sprinkler → Grama Molhada
          ↓
```

Neste exemplo:
- Chuva influencia diretamente se a grama está molhada
- Sprinkler também influencia a grama
- Chuva influencia se o sprinkler está ligado

### **1.2 Extensão para o Domínio Temporal: DBNs**

**Redes Bayesianas Dinâmicas** estendem BNs para modelar processos que evoluem no tempo:

```
t=0          t=1          t=2          t=3
┌──┐        ┌──┐        ┌──┐        ┌──┐
│X₀│───────→│X₁│───────→│X₂│───────→│X₃│
└──┘        └──┘        └──┘        └──┘
 ↓           ↓           ↓           ↓
┌──┐        ┌──┐        ┌──┐        ┌──┐
│Y₀│        │Y₁│        │Y₂│        │Y₃│
└──┘        └──┘        └──┘        └──┘
```

**Características:**
- **Estado Oculto (X):** Variáveis não observadas que evoluem no tempo
- **Observações (Y):** Medições ou evidências em cada instante
- **Transições:** Como o estado muda de t para t+1
- **Emissões:** Como observações são geradas a partir do estado

### **1.3 Componentes de uma DBN**

#### **📊 1. Rede de Tempo Inicial (Time-Slice 0)**

Define a distribuição inicial:
```
P(X₀)
```

#### **🔄 2. Modelo de Transição (2-Time-Slice BN)**

Define como o sistema evolui:
```
P(Xₜ₊₁ | Xₜ, Uₜ)
```
Onde:
- **Xₜ:** Estado no tempo t
- **Uₜ:** Entradas/ações no tempo t

#### **📡 3. Modelo de Observação**

Define como observações são geradas:
```
P(Yₜ | Xₜ)
```

### **1.4 Assunção de Markov**

DBNs geralmente assumem a **Propriedade de Markov:**

```
P(Xₜ₊₁ | X₀, X₁, ..., Xₜ) = P(Xₜ₊₁ | Xₜ)
```

**Interpretação:**
> "O futuro é independente do passado dado o presente."

Esta assunção simplifica significativamente a modelagem e inferência.

---

## **2. 🔧 Matemática das DBNs**

### **2.1 Fatorização da Distribuição Conjunta**

Para uma sequência temporal de comprimento T, a distribuição conjunta fatoriza como:

```
P(X₀, X₁, ..., Xₜ, Y₀, Y₁, ..., Yₜ) = 
    P(X₀) × P(Y₀|X₀) × 
    ∏ₜ₌₁ᵀ [P(Xₜ|Xₜ₋₁) × P(Yₜ|Xₜ)]
```

**Componentes:**
1. **Prior:** P(X₀) - Distribuição inicial
2. **Transição:** P(Xₜ|Xₜ₋₁) - Dinâmica temporal
3. **Emissão:** P(Yₜ|Xₜ) - Geração de observações

### **2.2 Modelos Clássicos como Casos Especiais**

#### **🎯 Hidden Markov Model (HMM)**

HMM é um caso especial de DBN onde:
- Estado é uma variável **discreta**
- Observações podem ser discretas ou contínuas
- Estrutura mais simples

```python
# Exemplo de HMM como DBN
class HiddenMarkovModel:
    def __init__(self, n_states, n_observations):
        # Distribuição inicial
        self.pi = np.ones(n_states) / n_states
        
        # Matriz de transição: P(Xₜ₊₁ | Xₜ)
        self.A = np.ones((n_states, n_states)) / n_states
        
        # Matriz de emissão: P(Yₜ | Xₜ)
        self.B = np.ones((n_states, n_observations)) / n_observations
```

#### **📈 Kalman Filter**

Kalman Filter é uma DBN com:
- Estado **contínuo**
- Transições e observações **lineares**
- Ruído **gaussiano**

```python
# Kalman Filter como DBN
class KalmanFilter:
    def __init__(self, dim_state, dim_obs):
        # Modelo de transição: Xₜ₊₁ = F·Xₜ + w
        self.F = np.eye(dim_state)  # Matriz de transição
        self.Q = np.eye(dim_state)  # Covariância do ruído de processo
        
        # Modelo de observação: Yₜ = H·Xₜ + v
        self.H = np.eye(dim_obs, dim_state)  # Matriz de observação
        self.R = np.eye(dim_obs)  # Covariância do ruído de medição
```

### **2.3 Tarefas de Inferência em DBNs**

#### **🔍 1. Filtragem (Filtering)**

Estimar estado atual dado observações passadas:
```
P(Xₜ | Y₁, Y₂, ..., Yₜ)
```

**Aplicação:** Rastreamento em tempo real

#### **🔮 2. Predição (Prediction)**

Prever estados futuros:
```
P(Xₜ₊ₖ | Y₁, Y₂, ..., Yₜ)  onde k > 0
```

**Aplicação:** Previsão de séries temporais

#### **🔄 3. Suavização (Smoothing)**

Estimar estados passados com todas as observações:
```
P(Xₜ | Y₁, Y₂, ..., Yₜ)  para todo t ≤ T
```

**Aplicação:** Análise retrospectiva, pós-processamento

#### **📊 4. Verossimilhança (Likelihood)**

Calcular probabilidade das observações:
```
P(Y₁, Y₂, ..., Yₜ)
```

**Aplicação:** Comparação de modelos, detecção de anomalias

#### **🎯 5. Viterbi (Most Likely Explanation)**

Encontrar sequência de estados mais provável:
```
argmax P(X₁, X₂, ..., Xₜ | Y₁, Y₂, ..., Yₜ)
  X₁,...,Xₜ
```

**Aplicação:** Reconhecimento de padrões, diagnóstico

---

## **3. 💻 Implementação de DBNs**

### **3.1 🔧 Hidden Markov Model (HMM)**

```python
import numpy as np
from scipy.special import logsumexp

class HMM:
    """
    Hidden Markov Model - caso especial de DBN
    """
    
    def __init__(self, n_states, n_observations):
        """
        Args:
            n_states: Número de estados ocultos
            n_observations: Número de símbolos de observação
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        # Parâmetros do modelo
        self.start_prob = np.ones(n_states) / n_states  # π
        self.trans_prob = np.ones((n_states, n_states)) / n_states  # A
        self.emiss_prob = np.ones((n_states, n_observations)) / n_observations  # B
    
    def forward(self, observations):
        """
        Algoritmo Forward para calcular P(Y₁, ..., Yₜ, Xₜ)
        
        Returns:
            alpha: Matriz forward (T, n_states)
            log_likelihood: log P(Y₁, ..., Yₜ)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Inicialização: t = 0
        alpha[0] = self.start_prob * self.emiss_prob[:, observations[0]]
        
        # Recursão: t = 1, ..., T-1
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = (alpha[t-1] @ self.trans_prob[:, j]) * \
                             self.emiss_prob[j, observations[t]]
        
        # Verossimilhança
        log_likelihood = np.log(alpha[-1].sum())
        
        return alpha, log_likelihood
    
    def backward(self, observations):
        """
        Algoritmo Backward para calcular P(Yₜ₊₁, ..., Yₜ | Xₜ)
        
        Returns:
            beta: Matriz backward (T, n_states)
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Inicialização: t = T-1
        beta[-1] = 1.0
        
        # Recursão: t = T-2, ..., 0
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.trans_prob[i, :] * 
                    self.emiss_prob[:, observations[t+1]] * 
                    beta[t+1]
                )
        
        return beta
    
    def viterbi(self, observations):
        """
        Algoritmo de Viterbi - sequência de estados mais provável
        
        Returns:
            best_path: Sequência de estados mais provável
            best_prob: Probabilidade dessa sequência
        """
        T = len(observations)
        
        # Matriz para armazenar probabilidades máximas
        delta = np.zeros((T, self.n_states))
        
        # Matriz para rastreamento (backpointer)
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Inicialização
        delta[0] = self.start_prob * self.emiss_prob[:, observations[0]]
        
        # Recursão
        for t in range(1, T):
            for j in range(self.n_states):
                # Calcular max sobre estados anteriores
                prob = delta[t-1] * self.trans_prob[:, j]
                psi[t, j] = np.argmax(prob)
                delta[t, j] = np.max(prob) * self.emiss_prob[j, observations[t]]
        
        # Terminação
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(delta[-1])
        best_prob = np.max(delta[-1])
        
        # Backtracking
        for t in range(T-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]
        
        return best_path, best_prob
    
    def baum_welch(self, observations, max_iter=100, tol=1e-6):
        """
        Algoritmo Baum-Welch (EM) para aprendizado de parâmetros
        
        Args:
            observations: Sequência de observações
            max_iter: Número máximo de iterações
            tol: Tolerância para convergência
        """
        T = len(observations)
        prev_likelihood = float('-inf')
        
        for iteration in range(max_iter):
            # E-Step: Forward-Backward
            alpha, log_likelihood = self.forward(observations)
            beta = self.backward(observations)
            
            # Calcular gamma: P(Xₜ = i | Y₁, ..., Yₜ)
            gamma = alpha * beta
            gamma = gamma / gamma.sum(axis=1, keepdims=True)
            
            # Calcular xi: P(Xₜ = i, Xₜ₊₁ = j | Y₁, ..., Yₜ)
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * 
                                      self.trans_prob[i, j] * 
                                      self.emiss_prob[j, observations[t+1]] * 
                                      beta[t+1, j])
                
                xi[t] /= xi[t].sum()
            
            # M-Step: Atualizar parâmetros
            # Atualizar start_prob
            self.start_prob = gamma[0]
            
            # Atualizar trans_prob
            self.trans_prob = xi.sum(axis=0) / gamma[:-1].sum(axis=0, keepdims=True).T
            
            # Atualizar emiss_prob
            for k in range(self.n_observations):
                mask = (observations == k)
                self.emiss_prob[:, k] = gamma[mask].sum(axis=0)
            
            self.emiss_prob /= gamma.sum(axis=0, keepdims=True).T
            
            # Verificar convergência
            if iteration > 0 and abs(log_likelihood - prev_likelihood) < tol:
                print(f"Convergiu na iteração {iteration}")
                break
            
            prev_likelihood = log_likelihood
            
            if iteration % 10 == 0:
                print(f"Iteração {iteration}: log-likelihood = {log_likelihood:.4f}")
        
        return log_likelihood
    
    def generate(self, length):
        """
        Gerar sequência de observações do modelo
        
        Args:
            length: Comprimento da sequência
        
        Returns:
            states: Sequência de estados
            observations: Sequência de observações
        """
        states = np.zeros(length, dtype=int)
        observations = np.zeros(length, dtype=int)
        
        # Estado inicial
        states[0] = np.random.choice(self.n_states, p=self.start_prob)
        observations[0] = np.random.choice(
            self.n_observations, 
            p=self.emiss_prob[states[0]]
        )
        
        # Gerar sequência
        for t in range(1, length):
            states[t] = np.random.choice(
                self.n_states, 
                p=self.trans_prob[states[t-1]]
            )
            observations[t] = np.random.choice(
                self.n_observations, 
                p=self.emiss_prob[states[t]]
            )
        
        return states, observations

# Exemplo de uso: Modelo de clima
if __name__ == "__main__":
    # Definir modelo
    # Estados: 0=Ensolarado, 1=Nublado, 2=Chuvoso
    # Observações: 0=Seco, 1=Úmido, 2=Molhado
    
    hmm = HMM(n_states=3, n_observations=3)
    
    # Definir parâmetros manualmente
    hmm.start_prob = np.array([0.6, 0.3, 0.1])
    
    hmm.trans_prob = np.array([
        [0.7, 0.2, 0.1],  # Ensolarado -> ...
        [0.3, 0.4, 0.3],  # Nublado -> ...
        [0.2, 0.3, 0.5]   # Chuvoso -> ...
    ])
    
    hmm.emiss_prob = np.array([
        [0.7, 0.25, 0.05],  # Ensolarado emite ...
        [0.2, 0.5, 0.3],    # Nublado emite ...
        [0.05, 0.25, 0.7]   # Chuvoso emite ...
    ])
    
    # Gerar dados
    print("Gerando sequência...")
    true_states, observations = hmm.generate(50)
    
    print(f"Observações: {observations[:20]}")
    print(f"Estados verdadeiros: {true_states[:20]}")
    
    # Inferência: Viterbi
    print("\nInferência com Viterbi...")
    predicted_states, prob = hmm.viterbi(observations)
    
    print(f"Estados preditos: {predicted_states[:20]}")
    print(f"Probabilidade: {prob:.6f}")
    
    # Acurácia
    accuracy = (predicted_states == true_states).mean()
    print(f"Acurácia: {accuracy:.2%}")
    
    # Aprendizado: Baum-Welch
    print("\nAprendendo parâmetros com Baum-Welch...")
    hmm_learn = HMM(n_states=3, n_observations=3)
    hmm_learn.baum_welch(observations, max_iter=50)
```

### **3.2 🎯 Kalman Filter**

```python
class KalmanFilter:
    """
    Kalman Filter - DBN com estados contínuos e modelo linear gaussiano
    """
    
    def __init__(self, dim_state, dim_obs):
        """
        Args:
            dim_state: Dimensão do vetor de estado
            dim_obs: Dimensão do vetor de observação
        """
        self.dim_x = dim_state
        self.dim_z = dim_obs
        
        # Modelo de transição: xₜ₊₁ = F·xₜ + w, w ~ N(0, Q)
        self.F = np.eye(dim_state)  # Matriz de transição de estado
        self.Q = np.eye(dim_state)  # Covariância do ruído de processo
        
        # Modelo de observação: zₜ = H·xₜ + v, v ~ N(0, R)
        self.H = np.eye(dim_obs, dim_state)  # Matriz de observação
        self.R = np.eye(dim_obs)  # Covariância do ruído de medição
        
        # Estado e covariância
        self.x = np.zeros(dim_state)  # Estimativa do estado
        self.P = np.eye(dim_state)    # Covariância do estado
    
    def predict(self, u=None):
        """
        Etapa de Predição (Time Update)
        
        Args:
            u: Vetor de controle/entrada (opcional)
        """
        # Predizer estado: x̂ₜ₊₁|ₜ = F·x̂ₜ|ₜ
        self.x = self.F @ self.x
        if u is not None:
            self.x += u
        
        # Predizer covariância: Pₜ₊₁|ₜ = F·Pₜ|ₜ·Fᵀ + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z):
        """
        Etapa de Atualização (Measurement Update)
        
        Args:
            z: Vetor de observação/medição
        """
        # Inovação: y = zₜ - H·x̂ₜ|ₜ₋₁
        y = z - self.H @ self.x
        
        # Covariância da inovação: S = H·Pₜ|ₜ₋₁·Hᵀ + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Ganho de Kalman: K = Pₜ|ₜ₋₁·Hᵀ·S⁻¹
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Atualizar estado: x̂ₜ|ₜ = x̂ₜ|ₜ₋₁ + K·y
        self.x = self.x + K @ y
        
        # Atualizar covariância: Pₜ|ₜ = (I - K·H)·Pₜ|ₜ₋₁
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
    
    def filter(self, observations, controls=None):
        """
        Filtragem completa de uma sequência
        
        Args:
            observations: Lista de vetores de observação
            controls: Lista de vetores de controle (opcional)
        
        Returns:
            states: Estimativas do estado em cada instante
            covariances: Covariâncias em cada instante
        """
        T = len(observations)
        states = np.zeros((T, self.dim_x))
        covariances = np.zeros((T, self.dim_x, self.dim_x))
        
        for t in range(T):
            # Predição
            u = controls[t] if controls is not None else None
            self.predict(u)
            
            # Atualização
            self.update(observations[t])
            
            # Armazenar
            states[t] = self.x.copy()
            covariances[t] = self.P.copy()
        
        return states, covariances
    
    def smooth(self, observations):
        """
        Suavização RTS (Rauch-Tung-Striebel)
        
        Estima estados passados usando todas as observações
        """
        T = len(observations)
        
        # Forward pass (filtragem)
        filtered_states = np.zeros((T, self.dim_x))
        filtered_covs = np.zeros((T, self.dim_x, self.dim_x))
        predicted_covs = np.zeros((T, self.dim_x, self.dim_x))
        
        for t in range(T):
            self.predict()
            predicted_covs[t] = self.P.copy()
            
            self.update(observations[t])
            filtered_states[t] = self.x.copy()
            filtered_covs[t] = self.P.copy()
        
        # Backward pass (suavização)
        smoothed_states = filtered_states.copy()
        smoothed_covs = filtered_covs.copy()
        
        for t in range(T-2, -1, -1):
            # Ganho de suavização
            C = filtered_covs[t] @ self.F.T @ np.linalg.inv(predicted_covs[t+1])
            
            # Suavizar estado
            smoothed_states[t] = (filtered_states[t] + 
                                 C @ (smoothed_states[t+1] - self.F @ filtered_states[t]))
            
            # Suavizar covariância
            smoothed_covs[t] = (filtered_covs[t] + 
                               C @ (smoothed_covs[t+1] - predicted_covs[t+1]) @ C.T)
        
        return smoothed_states, smoothed_covs

# Exemplo: Rastreamento de posição
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Modelo: posição em 2D com velocidade
    # Estado: [x, y, vx, vy]
    # Observação: [x, y]
    
    dt = 0.1  # Intervalo de tempo
    
    kf = KalmanFilter(dim_state=4, dim_obs=2)
    
    # Matriz de transição (movimento uniforme)
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Matriz de observação (observar apenas posição)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # Ruído de processo
    kf.Q = np.eye(4) * 0.1
    
    # Ruído de medição
    kf.R = np.eye(2) * 1.0
    
    # Estado inicial
    kf.x = np.array([0, 0, 1, 0.5])
    kf.P = np.eye(4) * 1.0
    
    # Simular trajetória
    T = 100
    true_states = np.zeros((T, 4))
    observations = np.zeros((T, 2))
    
    true_states[0] = kf.x.copy()
    
    for t in range(1, T):
        # Evolução verdadeira
        true_states[t] = kf.F @ true_states[t-1] + np.random.multivariate_normal(
            np.zeros(4), kf.Q
        )
        
        # Observação ruidosa
        observations[t] = kf.H @ true_states[t] + np.random.multivariate_normal(
            np.zeros(2), kf.R
        )
    
    # Resetar filtro
    kf.x = np.array([0, 0, 0, 0])
    kf.P = np.eye(4) * 5.0
    
    # Filtrar
    print("Filtrando...")
    filtered_states, _ = kf.filter(observations)
    
    # Suavizar
    print("Suavizando...")
    kf.x = np.array([0, 0, 0, 0])
    kf.P = np.eye(4) * 5.0
    smoothed_states, _ = kf.smooth(observations)
    
    # Visualizar
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(true_states[:, 0], true_states[:, 1], 'g-', 
            label='Trajetória Verdadeira', linewidth=2)
    plt.scatter(observations[:, 0], observations[:, 1], 
               c='red', s=20, alpha=0.5, label='Observações Ruidosas')
    plt.plot(filtered_states[:, 0], filtered_states[:, 1], 'b--', 
            label='Filtrado (Kalman)', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Filtragem de Kalman')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(true_states[:, 0], true_states[:, 1], 'g-', 
            label='Trajetória Verdadeira', linewidth=2)
    plt.scatter(observations[:, 0], observations[:, 1], 
               c='red', s=20, alpha=0.5, label='Observações Ruidosas')
    plt.plot(smoothed_states[:, 0], smoothed_states[:, 1], 'm-', 
            label='Suavizado (RTS)', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Suavização RTS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Calcular erros
    mse_filtered = np.mean((filtered_states[:, :2] - true_states[:, :2])**2)
    mse_smoothed = np.mean((smoothed_states[:, :2] - true_states[:, :2])**2)
    
    print(f"\nMSE Filtrado: {mse_filtered:.4f}")
    print(f"MSE Suavizado: {mse_smoothed:.4f}")
```

---

## **4. 📊 Aplicações de DBNs**

### **4.1 🗣️ Reconhecimento de Fala**

HMMs são amplamente usados em reconhecimento automático de fala (ASR):

**Estrutura:**
- **Estados:** Fonemas ou sub-unidades fonéticas
- **Observações:** Vetores de características acústicas (MFCCs)

```python
class SpeechRecognitionHMM:
    """
    Sistema simplificado de reconhecimento de fala usando HMMs
    """
    
    def __init__(self, vocabulary):
        """
        Args:
            vocabulary: Lista de palavras a reconhecer
        """
        self.vocabulary = vocabulary
        self.word_models = {}
        
        # Criar um HMM para cada palavra
        for word in vocabulary:
            # Número de estados baseado no comprimento da palavra
            n_states = max(3, len(word))
            n_observations = 64  # Dimensão do vetor de características
            
            self.word_models[word] = HMM(n_states, n_observations)
    
    def train(self, word, speech_features_list):
        """
        Treinar modelo de uma palavra
        
        Args:
            word: Palavra a treinar
            speech_features_list: Lista de sequências de características
        """
        # Treinar HMM com múltiplas gravações da palavra
        for features in speech_features_list:
            self.word_models[word].baum_welch(features, max_iter=20)
    
    def recognize(self, speech_features):
        """
        Reconhecer palavra falada
        
        Args:
            speech_features: Sequência de características acústicas
        
        Returns:
            Palavra reconhecida e pontuação de confiança
        """
        best_word = None
        best_likelihood = float('-inf')
        
        for word in self.vocabulary:
            _, log_likelihood = self.word_models[word].forward(speech_features)
            
            if log_likelihood > best_likelihood:
                best_likelihood = log_likelihood
                best_word = word
        
        return best_word, best_likelihood
```

### **4.2 🧬 Bioinformática: Predição de Estrutura de Proteínas**

DBNs modelam a estrutura secundária de proteínas:

```python
class ProteinStructurePredictor:
    """
    Predição de estrutura secundária de proteínas usando DBN
    """
    
    def __init__(self):
        # Estados: Hélice (H), Folha (E), Loop (C)
        # Observações: 20 aminoácidos
        self.hmm = HMM(n_states=3, n_observations=20)
        
        # Mapear aminoácidos para índices
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    
    def sequence_to_indices(self, sequence):
        """
        Converter sequência de aminoácidos em índices
        """
        return np.array([self.aa_to_idx[aa] for aa in sequence])
    
    def predict_structure(self, protein_sequence):
        """
        Predizer estrutura secundária
        
        Args:
            protein_sequence: String com sequência de aminoácidos
        
        Returns:
            structure: 'H' (hélice), 'E' (folha), ou 'C' (loop) para cada resíduo
        """
        indices = self.sequence_to_indices(protein_sequence)
        states, _ = self.hmm.viterbi(indices)
        
        # Mapear estados para estruturas
        state_to_structure = {0: 'H', 1: 'E', 2: 'C'}
        structure = ''.join([state_to_structure[s] for s in states])
        
        return structure
```

### **4.3 💰 Finanças: Detecção de Regimes de Mercado**

DBNs identificam regimes de mercado (bull, bear, neutro):

```python
class MarketRegimeDetector:
    """
    Detector de regimes de mercado usando HMM
    """
    
    def __init__(self, n_regimes=3):
        """
        Args:
            n_regimes: Número de regimes (tipicamente 3: bull, bear, neutro)
        """
        self.n_regimes = n_regimes
        
        # Quantizar retornos em bins
        self.n_return_bins = 10
        
        self.hmm = HMM(n_states=n_regimes, n_observations=self.n_return_bins)
    
    def discretize_returns(self, returns):
        """
        Discretizar retornos contínuos em bins
        """
        # Usar quantis para criar bins
        quantiles = np.linspace(0, 1, self.n_return_bins + 1)
        bins = np.quantile(returns, quantiles)
        
        # Discretizar
        return np.digitize(returns, bins[1:-1])
    
    def fit(self, returns):
        """
        Treinar detector com dados históricos
        
        Args:
            returns: Série temporal de retornos
        """
        # Discretizar retornos
        discrete_returns = self.discretize_returns(returns)
        
        # Treinar HMM
        self.hmm.baum_welch(discrete_returns, max_iter=50)
    
    def detect_regime(self, returns):
        """
        Detectar regime atual
        
        Returns:
            regimes: Sequência de regimes detectados
            probabilities: Probabilidades filtradas P(regime_t | returns_{1:t})
        """
        discrete_returns = self.discretize_returns(returns)
        
        # Forward para obter probabilidades filtradas
        alpha, _ = self.hmm.forward(discrete_returns)
        
        # Normalizar para obter probabilidades
        probabilities = alpha / alpha.sum(axis=1, keepdims=True)
        
        # Regime mais provável em cada instante
        regimes = np.argmax(probabilities, axis=1)
        
        return regimes, probabilities

# Exemplo de uso
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Simular retornos de mercado com mudança de regime
    np.random.seed(42)
    T = 500
    
    # Simular 3 regimes
    regime_returns = {
        0: (0.001, 0.01),   # Bull: alta média, baixa volatilidade
        1: (-0.002, 0.02),  # Bear: baixa média, alta volatilidade
        2: (0.0, 0.005)     # Neutro: média zero, baixa volatilidade
    }
    
    true_regimes = np.concatenate([
        np.zeros(150, dtype=int),
        np.ones(200, dtype=int),
        np.full(150, 2, dtype=int)
    ])
    
    returns = np.zeros(T)
    for t in range(T):
        regime = true_regimes[t]
        mean, std = regime_returns[regime]
        returns[t] = np.random.normal(mean, std)
    
    # Detectar regimes
    detector = MarketRegimeDetector(n_regimes=3)
    detector.fit(returns)
    detected_regimes, probs = detector.detect_regime(returns)
    
    # Visualizar
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    # Retornos
    ax1.plot(returns, 'b-', alpha=0.7)
    ax1.set_ylabel('Retornos')
    ax1.set_title('Retornos do Mercado')
    ax1.grid(True, alpha=0.3)
    
    # Regimes verdadeiros
    ax2.plot(true_regimes, 'g-', linewidth=2, label='Regime Verdadeiro')
    ax2.set_ylabel('Regime')
    ax2.set_title('Regimes Verdadeiros')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Bull', 'Bear', 'Neutro'])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Regimes detectados
    ax3.plot(detected_regimes, 'r-', linewidth=2, label='Regime Detectado')
    ax3.set_xlabel('Tempo')
    ax3.set_ylabel('Regime')
    ax3.set_title('Regimes Detectados pelo HMM')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Regime 0', 'Regime 1', 'Regime 2'])
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Acurácia
    accuracy = (detected_regimes == true_regimes).mean()
    print(f"Acurácia na detecção de regimes: {accuracy:.2%}")
```

---

## **5. ⚖️ Vantagens e Limitações**

### **5.1 ✅ Vantagens**

| **Vantagem** | **Descrição** |
|--------------|---------------|
| **📊 Modelagem Temporal** | Captura dependências temporais naturalmente |
| **🎯 Inferência Rigorosa** | Fundamentação probabilística para raciocínio |
| **🔧 Flexibilidade** | Modela diversos tipos de processos temporais |
| **📈 Interpretabilidade** | Estrutura de grafo facilita compreensão |
| **🎲 Lida com Incerteza** | Quantifica incerteza em predições |
| **🧩 Modularity** | Fácil adicionar/remover variáveis |

### **5.2 ❌ Limitações**

| **Limitação** | **Descrição** | **Mitigação** |
|---------------|---------------|---------------|
| **💻 Custo Computacional** | Inferência exata pode ser cara | Aproximações (Particle Filter, Variational) |
| **📊 Assunção de Markov** | Pode ser restritiva | Usar ordem superior ou modelo mais complexo |
| **🎛️ Seleção de Estrutura** | Difícil escolher estrutura ótima | Aprendizado de estrutura, validação cruzada |
| **📈 Escalabilidade** | Problemas com muitas variáveis | Aproximações, paralelização |
| **🔧 Aprendizado** | Requer dados suficientes | Regularização, priors informativos |

---

## **6. 📚 Referências e Recursos**

### **6.1 📖 Literatura Fundamental**

1. **Murphy, K. P. (2012).** *Machine Learning: A Probabilistic Perspective*. MIT Press.
   - Capítulos 17-18: DBNs e inferência temporal

2. **Koller, D., & Friedman, N. (2009).** *Probabilistic Graphical Models*. MIT Press.
   - Tratamento completo de modelos gráficos

3. **Rabiner, L. R. (1989).** *"A tutorial on hidden Markov models"*. Proceedings of the IEEE.
   - Tutorial clássico sobre HMMs

4. **Kalman, R. E. (1960).** *"A new approach to linear filtering and prediction problems"*. 
   - Artigo original do filtro de Kalman

### **6.2 🌐 Recursos Práticos**

```python
# Bibliotecas Python
import hmmlearn  # Hidden Markov Models
from filterpy.kalman import KalmanFilter  # Implementação Kalman
import pyro  # Programação probabilística (inclui DBNs)
import pomegranate  # Modelos gráficos probabilísticos
```

### **6.3 🔗 Links Úteis**

- **hmmlearn:** https://hmmlearn.readthedocs.io
- **Pyro:** https://pyro.ai
- **PGM Course:** https://www.coursera.org/learn/probabilistic-graphical-models

---

## **7. 🎯 Conclusão**

### **7.1 💡 Principais Aprendizados**

DBNs fornecem uma framework poderosa para modelagem temporal:

1. **Unificação:** Integra modelos clássicos (HMM, Kalman) em framework comum
2. **Probabilístico:** Raciocínio rigoroso sob incerteza
3. **Modular:** Estrutura permite fácil modificação e extensão
4. **Versátil:** Aplicável a diversos domínios

### **7.2 🔑 Quando Usar DBNs**

**✅ Cenários Ideais:**
- Dados sequenciais/temporais
- Necessidade de quantificar incerteza
- Estrutura causal conhecida ou deve ser descoberta
- Integração de múltiplas fontes de informação

**❌ Cenários Problemáticos:**
- Dados estáticos sem estrutura temporal
- Quando velocidade de inferência é crítica
- Relações não-estacionárias complexas
- Dados extremamente grandes sem estrutura

### **7.3 🌟 Mensagem Final**

> **"DBNs nos ensinam que modelar COMO o mundo evolui no tempo é tão importante quanto modelar o que observamos agora."**

A capacidade de raciocinar sobre processos temporais de forma probabilística torna DBNs indispensáveis em aplicações onde o tempo e a incerteza são fundamentais.

---

**🔗 Continue Explorando:**
- 📖 Relacionado: [**Gaussian Process Regression**](../statistical_learning/gaussian_process_regression.md)
- 🎯 Próximo: [**BUS with Subset Simulation**](../reliability_analysis/bus_subset_simulation.md)
- 🔄 Veja também: [**Hidden Markov Models**](../probabilistic_models/hidden_markov_models.md)

**🎓 Obrigado por explorar Redes Bayesianas Dinâmicas!**

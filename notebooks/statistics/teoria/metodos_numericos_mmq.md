# Métodos Numéricos - Mínimos Quadrados (MMQ) 🔢

O **Método dos Mínimos Quadrados** (MMQ, ou Least Squares em inglês) é uma técnica fundamental de otimização para ajustar modelos matemáticos a dados observados. É a base da regressão linear e muitos outros métodos estatísticos e de aprendizado de máquina.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 O Problema**

Dado um conjunto de observações (xᵢ, yᵢ), queremos encontrar parâmetros β que minimizem o **erro quadrático** entre valores observados e preditos.

**Formulação:**
```
Minimizar: S(β) = Σ(yᵢ - f(xᵢ, β))²

onde:
• yᵢ: valores observados
• f(xᵢ, β): valores preditos pelo modelo
• eᵢ = yᵢ - f(xᵢ, β): resíduos
```

### **1.2 Por Que Quadrados?**

**Razões Históricas e Práticas:**

1. **Penaliza Erros Grandes:** Quadrado enfatiza desvios maiores
2. **Diferenciável:** Facilita otimização analítica
3. **Solução Única:** Para problemas lineares
4. **Propriedades Estatísticas:** Sob normalidade, é estimador de máxima verossimilhança
5. **Geometria:** Projeção ortogonal no espaço vetorial

**Alternativas:**
```
Mínimos Quadrados:    Σ(eᵢ)²      ← Sensível a outliers
Mínimos Absolutos:    Σ|eᵢ|       ← Mais robusto
Minimax:              max|eᵢ|     ← Minimiza pior caso
```

### **1.3 Contexto Histórico**

**Carl Friedrich Gauss (1809):**
> Desenvolveu o método para astronomia (previsão de órbitas de asteroides)

**Adrien-Marie Legendre (1805):**
> Publicou primeiro o método (disputa de prioridade)

**Andrey Kolmogorov (1930s):**
> Fundamentação probabilística moderna

---

## **2. 📊 Mínimos Quadrados Ordinários (OLS)**

### **2.1 Modelo Linear**

**Forma Escalar:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε

onde:
• y: variável resposta
• xⱼ: variáveis preditoras
• βⱼ: coeficientes (parâmetros)
• ε: erro aleatório
```

**Forma Matricial:**
```
Y = Xβ + ε

onde:
• Y: vetor n×1 de respostas
• X: matriz n×p de preditores (design matrix)
• β: vetor p×1 de coeficientes
• ε: vetor n×1 de erros
```

**Exemplo com 3 observações:**
```
[y₁]   [1  x₁₁  x₁₂]   [β₀]   [ε₁]
[y₂] = [1  x₂₁  x₂₂] × [β₁] + [ε₂]
[y₃]   [1  x₃₁  x₃₂]   [β₂]   [ε₃]
```

### **2.2 Derivação da Solução**

**Objetivo:** Minimizar RSS (Residual Sum of Squares)
```
RSS(β) = Σeᵢ² = eᵀe = (Y - Xβ)ᵀ(Y - Xβ)
```

**Expandindo:**
```
RSS(β) = YᵀY - 2βᵀXᵀY + βᵀXᵀXβ
```

**Derivando e igualando a zero:**
```
∂RSS/∂β = -2XᵀY + 2XᵀXβ = 0

XᵀXβ = XᵀY  ← Equações Normais
```

**Solução (se XᵀX é invertível):**
```
β̂ = (XᵀX)⁻¹XᵀY
```

Esta é a **solução de mínimos quadrados ordinários**!

### **2.3 Exemplo Numérico**

**Problema:** Ajustar y = β₀ + β₁x aos dados:
```
x: [1, 2, 3]
y: [2, 4, 5]
```

**Montando as Matrizes:**
```
X = [1  1]    Y = [2]
    [1  2]        [4]
    [1  3]        [5]
```

**Calculando XᵀX:**
```
XᵀX = [1  1  1] × [1  1]  = [3   6]
      [1  2  3]   [1  2]    [6  14]
                  [1  3]
```

**Calculando XᵀY:**
```
XᵀY = [1  1  1] × [2]  = [11]
      [1  2  3]   [4]    [27]
                  [5]
```

**Invertendo XᵀX:**
```
det(XᵀX) = 3×14 - 6×6 = 42 - 36 = 6

(XᵀX)⁻¹ = (1/6) × [14  -6] = [7/3   -1]
                    [-6   3]   [-1   1/2]
```

**Calculando β̂:**
```
β̂ = (XᵀX)⁻¹XᵀY = [7/3   -1] × [11]  = [77/3 - 27]   = [50/3]   = [1.0]
                   [-1   1/2]   [27]    [-11 + 27/2]   [5.5/2]     [1.5]

Solução: y = 1.0 + 1.5x
```

**Verificação:**
```
ŷ₁ = 1.0 + 1.5×1 = 2.5  (y₁=2, erro=0.5)
ŷ₂ = 1.0 + 1.5×2 = 4.0  (y₂=4, erro=0.0)
ŷ₃ = 1.0 + 1.5×3 = 5.5  (y₃=5, erro=0.5)

RSS = 0.5² + 0² + 0.5² = 0.5
```

### **2.4 Propriedades do Estimador OLS**

Sob as **hipóteses de Gauss-Markov**:

1. **Linearidade:** E[ε|X] = 0
2. **Homocedasticidade:** Var(ε|X) = σ²I
3. **Não-correlação:** Cov(εᵢ, εⱼ) = 0 para i≠j

O estimador OLS é **BLUE** (Best Linear Unbiased Estimator):
- **Best:** Menor variância entre estimadores lineares não-viesados
- **Linear:** Combinação linear de Y
- **Unbiased:** E[β̂] = β

**Propriedades Adicionais:**
```
E[β̂] = β                           (não-viesado)
Var(β̂) = σ²(XᵀX)⁻¹                 (matriz de covariância)
Cov(β̂, e) = 0                      (ortogonalidade)
```

---

## **3. 🔧 Interpretação Geométrica**

### **3.1 Projeção Ortogonal**

O MMQ encontra a **projeção ortogonal** de Y no espaço gerado pelas colunas de X.

**Visualização (caso 2D):**
```
        Y (vetor observado)
        ↑
        │╲
        │ ╲ e (resíduo)
        │  ╲
        │   ↘
        Ŷ ────→ (projeção no espaço coluna de X)
```

**Matematicamente:**
```
Ŷ = Xβ̂ = X(XᵀX)⁻¹XᵀY = HY

onde H = X(XᵀX)⁻¹Xᵀ é a matriz "hat" (projeção)
```

### **3.2 Matriz Hat (H)**

**Propriedades:**
```
1. H² = H              (idempotente)
2. Hᵀ = H              (simétrica)
3. HX = X              (projeta X em X)
4. trace(H) = p        (rank de H)
```

**Resíduos:**
```
e = Y - Ŷ = Y - HY = (I - H)Y
```

### **3.3 Ortogonalidade**

**Propriedade Fundamental:**
```
Xᵀe = Xᵀ(Y - Xβ̂) = XᵀY - XᵀX(XᵀX)⁻¹XᵀY = 0

Os resíduos são ORTOGONAIS às colunas de X!
```

**Consequências:**
```
1. Σeᵢ = 0              (se X tem coluna de 1s)
2. Σxᵢeᵢ = 0            (resíduos não correlacionados com X)
3. Σŷᵢeᵢ = 0            (predições ortogonais a resíduos)
```

---

## **4. 📈 Mínimos Quadrados Ponderados (WLS)**

### **4.1 Motivação**

Quando as observações têm **variâncias diferentes** (heterocedasticidade):
```
Var(εᵢ) = σᵢ²  (não constante!)
```

**Solução:** Dar **pesos diferentes** às observações.

### **4.2 Formulação**

**Minimizar:**
```
S(β) = Σwᵢ(yᵢ - f(xᵢ, β))²

onde wᵢ = 1/σᵢ² (inverso da variância)
```

**Forma Matricial:**
```
S(β) = (Y - Xβ)ᵀW(Y - Xβ)

onde W = diag(w₁, w₂, ..., wₙ)
```

### **4.3 Solução WLS**

**Equações Normais Ponderadas:**
```
XᵀWXβ = XᵀWY

β̂_WLS = (XᵀWX)⁻¹XᵀWY
```

### **4.4 Escolha dos Pesos**

**Cenários Comuns:**

1. **Variância Conhecida:**
   ```
   wᵢ = 1/σᵢ²
   ```

2. **Variância Proporcional a xᵢ:**
   ```
   wᵢ = 1/xᵢ
   ```

3. **Contagens (Poisson):**
   ```
   wᵢ = 1/yᵢ
   ```

4. **Variância Estimada:**
   ```
   1. Ajustar OLS
   2. Estimar σᵢ² dos resíduos
   3. Reajustar com WLS
   4. Iterar se necessário
   ```

### **4.5 Exemplo**

**Problema:** Medições com precisão variável
```
x:      [1,    2,    3,    4,    5]
y:      [2.0,  3.8,  6.2,  7.9,  10.1]
σ:      [0.5,  0.5,  1.0,  1.5,  2.0]  (desvio padrão)

Pesos: w = 1/σ²
w:      [4.0,  4.0,  1.0,  0.44, 0.25]
```

**Interpretação:** Observações iniciais (menor σ) recebem mais peso!

---

## **5. 🎯 Mínimos Quadrados Não-Lineares**

### **5.1 Problema**

Quando o modelo é **não-linear nos parâmetros**:
```
y = f(x, β) + ε

onde f é não-linear em β
```

**Exemplos:**
```
Exponencial:  y = β₀e^(β₁x)
Logística:    y = L/(1 + e^(-k(x-x₀)))
Michaelis-Menten: y = (Vₘₐₓ×x)/(Kₘ + x)
```

### **5.2 Métodos Iterativos**

Não há solução analítica fechada. Usa **otimização iterativa**.

**Algoritmo Geral:**
```
1. Chute inicial: β⁽⁰⁾
2. Para k = 0, 1, 2, ...
   a. Calcular resíduos: eᵢ⁽ᵏ⁾ = yᵢ - f(xᵢ, β⁽ᵏ⁾)
   b. Calcular Jacobiano: J⁽ᵏ⁾
   c. Atualizar: β⁽ᵏ⁺¹⁾ = β⁽ᵏ⁾ + Δβ⁽ᵏ⁾
3. Parar quando ||Δβ⁽ᵏ⁾|| < tol
```

### **5.3 Método de Gauss-Newton**

**Linearização Local:**
```
f(x, β + Δβ) ≈ f(x, β) + J×Δβ

onde J é o Jacobiano:
Jᵢⱼ = ∂f(xᵢ, β)/∂βⱼ
```

**Passo de Atualização:**
```
Δβ = (JᵀJ)⁻¹Jᵀe

(similar a OLS com J no lugar de X)
```

**Vantagens:**
- Rápido perto da solução
- Não requer segunda derivada

**Desvantagens:**
- Pode não convergir
- Sensível ao chute inicial

### **5.4 Método de Levenberg-Marquardt**

**Híbrido** entre Gauss-Newton e Gradiente Descendente:
```
(JᵀJ + λI)Δβ = Jᵀe

onde:
• λ = 0: Gauss-Newton (rápido)
• λ → ∞: Gradiente Descendente (estável)
```

**Estratégia Adaptativa:**
```
• λ grande no início (estável)
• λ diminui à medida que converge (rápido)
• Se RSS aumenta: aumentar λ, rejeitar passo
• Se RSS diminui: diminuir λ, aceitar passo
```

### **5.5 Exemplo: Ajuste Exponencial**

**Modelo:** y = ae^(bx)

**Dados:**
```
x: [0, 1, 2, 3, 4]
y: [1.0, 2.5, 7.0, 18.0, 50.0]
```

**Linearização:** ln(y) = ln(a) + bx
```
Regressão em ln(y):
ln(y): [0, 0.92, 1.95, 2.89, 3.91]

Resultado: ln(a) ≈ 0, b ≈ 0.98
Logo: a ≈ 1, b ≈ 1

Modelo: y ≈ e^x
```

**Refinamento com NLS:**
```
Usando Levenberg-Marquardt:
a = 0.99, b = 1.01

Modelo final: y = 0.99×e^(1.01x)
```

---

## **6. 🛡️ Regularização**

### **6.1 Problema de Overfitting**

Com muitos parâmetros, MMQ pode **superajustar**:
```
• RSS muito pequeno em treino
• RSS grande em teste
• Coeficientes muito grandes
```

**Solução:** Adicionar **penalidade** aos coeficientes.

### **6.2 Ridge Regression (L2)**

**Minimizar:**
```
S(β) = RSS + λ×Σβⱼ²
     = Σ(yᵢ - ŷᵢ)² + λ×||β||²

onde λ > 0 é o parâmetro de regularização
```

**Solução:**
```
β̂_ridge = (XᵀX + λI)⁻¹XᵀY
```

**Efeitos:**
- Encolhe todos os coeficientes
- Melhora estabilidade numérica
- Reduz variância (aumenta viés)
- **Não zera coeficientes**

**Escolha de λ:**
```
λ = 0:     OLS puro
λ pequeno: Pouca regularização
λ grande:  Muita regularização (β → 0)
λ → ∞:     β̂ → 0 (modelo constante)
```

### **6.3 Lasso Regression (L1)**

**Minimizar:**
```
S(β) = RSS + λ×Σ|βⱼ|
     = Σ(yᵢ - ŷᵢ)² + λ×||β||₁
```

**Características:**
- **Zera coeficientes** (seleção de features)
- Não tem solução fechada
- Resolve por otimização convexa

**Comparação Ridge vs. Lasso:**
```
Ridge:
• Encolhe suavemente
• Mantém todas features
• Bom quando muitas features relevantes

Lasso:
• Seleciona features (zera coeficientes)
• Interpretável
• Bom quando muitas features irrelevantes
```

### **6.4 Elastic Net**

**Combinação** de Ridge e Lasso:
```
S(β) = RSS + λ₁×Σ|βⱼ| + λ₂×Σβⱼ²
```

**Vantagens:**
- Herda benefícios de ambos
- Estável com features correlacionadas
- Seleção de grupos de features

---

## **7. 📊 Diagnósticos e Validação**

### **7.1 Análise de Resíduos**

**Resíduos Padronizados:**
```
rᵢ = eᵢ / (s×√(1 - hᵢᵢ))

onde:
• s = √(RSS/(n-p)): desvio padrão residual
• hᵢᵢ: elemento diagonal de H (leverage)
```

**Gráficos:**
1. **Resíduos vs. Ajustados:** Detecta não-linearidade
2. **Q-Q Plot:** Testa normalidade
3. **Scale-Location:** Detecta heterocedasticidade
4. **Residuals vs. Leverage:** Identifica pontos influentes

### **7.2 Estatísticas de Influência**

**Leverage (hᵢᵢ):**
```
Mede quão "extremo" é xᵢ

hᵢᵢ alto → ponto influente
```

**Distância de Cook:**
```
Dᵢ = (rᵢ²/p) × (hᵢᵢ/(1-hᵢᵢ))

Dᵢ > 1: ponto muito influente (investigar)
```

**DFBETAS:**
```
Mudança em β ao remover observação i
```

### **7.3 Métricas de Qualidade**

**R² (Coeficiente de Determinação):**
```
R² = 1 - RSS/TSS

0 ≤ R² ≤ 1
```

**R² Ajustado:**
```
R²ₐⱼ = 1 - (1-R²)×(n-1)/(n-p-1)

Penaliza modelos com muitos parâmetros
```

**AIC (Akaike Information Criterion):**
```
AIC = n×ln(RSS/n) + 2p

Menor AIC = melhor modelo
```

**BIC (Bayesian Information Criterion):**
```
BIC = n×ln(RSS/n) + p×ln(n)

Penaliza mais modelos complexos que AIC
```

---

## **8. 🧮 Considerações Computacionais**

### **8.1 Complexidade**

**Método Direto (Inversão):**
```
Complexidade: O(p³) + O(p²n)

Gargalo: inversão de (XᵀX)
```

**Decomposição QR:**
```
X = QR

β̂ = R⁻¹QᵀY

Complexidade: O(p²n)
Mais estável numericamente
```

**Decomposição SVD:**
```
X = UΣVᵀ

β̂ = VΣ⁺UᵀY

onde Σ⁺ é pseudo-inversa

Mais estável, detecta multicolinearidade
```

### **8.2 Problemas Numéricos**

**Multicolinearidade:**
```
Colunas de X altamente correlacionadas

Problema: (XᵀX) quase singular
Solução: Ridge, PCA, remover features
```

**Número de Condição:**
```
κ = σₘₐₓ / σₘᵢₙ

κ > 100: mal-condicionado
κ > 1000: muito mal-condicionado
```

**Rank Deficiency:**
```
rank(X) < p (colunas linearmente dependentes)

Solução: Pseudo-inversa (SVD)
```

### **8.3 Grandes Datasets**

**Gradiente Descendente Estocástico (SGD):**
```
Para cada mini-batch:
  β ← β - α×∇RSS

Complexidade por iteração: O(batch_size × p)
```

**Online Learning:**
```
Atualiza β incrementalmente com novos dados
Não precisa armazenar todos os dados
```

---

## **9. 🧮 Exercícios Resolvidos**

### **Exercício 1: OLS Manual**
**Dados:** Ajustar y = β₀ + β₁x
```
x: [0, 1, 2]
y: [1, 2, 4]
```

**Solução:**
```
X = [1  0]    Y = [1]
    [1  1]        [2]
    [1  2]        [4]

XᵀX = [3  3]    XᵀY = [7]
      [3  5]          [10]

det = 15 - 9 = 6

(XᵀX)⁻¹ = (1/6)[5  -3] = [5/6   -1/2]
               [-3  3]    [-1/2   1/2]

β̂ = [5/6   -1/2] × [7]  = [35/6 - 5]   = [5/6]   ≈ [0.83]
    [-1/2   1/2]   [10]    [-7/2 + 5]     [3/2]     [1.50]

Modelo: y = 0.83 + 1.50x
```

### **Exercício 2: Resíduos e R²**
**Continuando Exercício 1:**

**Predições:**
```
ŷ₀ = 0.83 + 1.50×0 = 0.83
ŷ₁ = 0.83 + 1.50×1 = 2.33
ŷ₂ = 0.83 + 1.50×2 = 3.83
```

**Resíduos:**
```
e₀ = 1 - 0.83 = 0.17
e₁ = 2 - 2.33 = -0.33
e₂ = 4 - 3.83 = 0.17

RSS = 0.17² + 0.33² + 0.17² = 0.167
```

**R²:**
```
ȳ = 7/3 ≈ 2.33

TSS = (1-2.33)² + (2-2.33)² + (4-2.33)²
    = 1.78 + 0.11 + 2.78 = 4.67

R² = 1 - 0.167/4.67 = 1 - 0.036 = 0.964 = 96.4%
```

---

## **10. 💻 Implementação em Python**

```python
import numpy as np
from scipy.linalg import lstsq
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

# Dados
X = np.array([[1, 0], [1, 1], [1, 2]])
y = np.array([1, 2, 4])

# 1. Solução Manual (OLS)
XtX = X.T @ X
Xty = X.T @ y
beta_manual = np.linalg.inv(XtX) @ Xty
print(f"OLS Manual: {beta_manual}")

# 2. Usando scipy.linalg.lstsq
beta_scipy, residuals, rank, s = lstsq(X, y)
print(f"OLS scipy: {beta_scipy}")

# 3. Usando sklearn
model = LinearRegression()
model.fit(X[:, 1:], y)  # sem coluna de 1s
print(f"OLS sklearn: intercept={model.intercept_}, coef={model.coef_}")

# 4. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X[:, 1:], y)
print(f"Ridge: intercept={ridge.intercept_}, coef={ridge.coef_}")

# 5. Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X[:, 1:], y)
print(f"Lasso: intercept={lasso.intercept_}, coef={lasso.coef_}")

# 6. Análise de Resíduos
y_pred = X @ beta_manual
residuals = y - y_pred
RSS = np.sum(residuals**2)
TSS = np.sum((y - y.mean())**2)
R2 = 1 - RSS/TSS

print(f"\nAnálise:")
print(f"Resíduos: {residuals}")
print(f"RSS: {RSS:.4f}")
print(f"R²: {R2:.4f}")

# 7. Visualização
plt.scatter(X[:, 1], y, label='Dados', s=100)
x_plot = np.linspace(0, 2, 100)
y_plot = beta_manual[0] + beta_manual[1]*x_plot
plt.plot(x_plot, y_plot, 'r-', label='Ajuste OLS')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Regressão Linear: y = {beta_manual[0]:.2f} + {beta_manual[1]:.2f}x')
plt.grid(True)
plt.show()
```

---

## **11. 🔗 Recursos Adicionais**

### **Livros Recomendados**
- **Linear Regression Analysis** - Seber & Lee
- **Applied Linear Regression** - Weisberg
- **Matrix Computations** - Golub & Van Loan
- **Numerical Recipes** - Press et al.

### **Bibliotecas Python**
```python
# Básico
import numpy as np
from scipy import linalg, optimize

# Machine Learning
from sklearn.linear_model import (
    LinearRegression,  # OLS
    Ridge,            # L2
    Lasso,            # L1
    ElasticNet,       # L1 + L2
    HuberRegressor    # Robusto
)

# Estatística
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, WLS
```

### **Ferramentas Online**
- [Wolfram Alpha](https://www.wolframalpha.com/)
- [Matrix Calculator](https://matrixcalc.org/)
- [Desmos Regression](https://www.desmos.com/calculator)

---

**Voltar para:** [Estatística](../README.md) | [Notebooks](../../README.md)

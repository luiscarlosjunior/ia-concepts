# Ajuste de Curva 📈

**Ajuste de curva** (curve fitting) é o processo de encontrar uma função matemática que melhor representa a relação entre variáveis observadas. É fundamental em ciência de dados, modelagem estatística e aprendizado de máquina.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 O Que É Ajuste de Curva?**

Dado um conjunto de pontos de dados (x, y), queremos encontrar uma função f(x) tal que:
```
y ≈ f(x)
```

**Objetivo:**
> Encontrar a função que **minimize o erro** entre valores observados e preditos.

**Tipos de Ajuste:**
- **Regressão:** Ajuste estatístico com ruído
- **Interpolação:** Passa exatamente pelos pontos
- **Suavização:** Compromisso entre ambos

### **1.2 Por Que Ajustar Curvas?**

**Aplicações:**
- ✅ **Predição:** Estimar valores futuros
- ✅ **Modelagem:** Descrever relações físicas/naturais
- ✅ **Compressão:** Representar muitos pontos com poucos parâmetros
- ✅ **Análise:** Identificar tendências e padrões
- ✅ **Interpolação:** Estimar valores entre pontos conhecidos

---

## **2. 📊 Regressão Linear Simples**

### **2.1 Definição**

Ajusta uma **reta** aos dados:
```
y = β₀ + β₁x + ε

onde:
• β₀: intercepto (coeficiente linear)
• β₁: inclinação (coeficiente angular)
• ε: erro aleatório
```

**Forma Estimada:**
```
ŷ = b₀ + b₁x

onde:
• ŷ (y-chapéu): valor predito
• b₀, b₁: estimativas dos parâmetros
```

### **2.2 Método dos Mínimos Quadrados Ordinários (OLS)**

Minimiza a **soma dos quadrados dos resíduos (RSS)**:
```
RSS = Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - b₀ - b₁xᵢ)²
```

**Solução Analítica:**
```
b₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
   = Cov(X,Y) / Var(X)

b₀ = ȳ - b₁x̄
```

### **2.3 Exemplo Prático**

**Dados:** Horas de estudo vs. Nota
```
X (horas): [1, 2, 3, 4, 5]
Y (nota):  [50, 60, 65, 80, 85]
```

**Cálculo:**
```
Passo 1: Médias
x̄ = 3, ȳ = 68

Passo 2: Calcular b₁
Numerador: (1-3)(50-68) + (2-3)(60-68) + ... = 70
Denominador: (1-3)² + (2-3)² + ... = 10

b₁ = 70/10 = 7

Passo 3: Calcular b₀
b₀ = 68 - 7×3 = 68 - 21 = 47

Equação: ŷ = 47 + 7x
```

**Interpretação:**
```
• Nota inicial (sem estudo): 47
• Cada hora adicional aumenta nota em 7 pontos
```

### **2.4 Propriedades**

**Propriedade 1: Reta Passa Pela Média**
```
A reta sempre passa pelo ponto (x̄, ȳ)
```

**Propriedade 2: Resíduos**
```
• Soma dos resíduos = 0
• Resíduos não correlacionados com X (OLS)
```

**Propriedade 3: Decomposição da Variância**
```
TSS = RSS + ESS

onde:
• TSS: Variação total = Σ(yᵢ - ȳ)²
• RSS: Variação residual = Σ(yᵢ - ŷᵢ)²
• ESS: Variação explicada = Σ(ŷᵢ - ȳ)²
```

### **2.5 Coeficiente de Determinação (R²)**

Mede a **proporção da variância explicada** pelo modelo.

**Fórmula:**
```
R² = 1 - (RSS / TSS) = ESS / TSS

0 ≤ R² ≤ 1
```

**Interpretação:**
```
R² = 0.0:   Modelo não explica nada (linha horizontal)
R² = 0.5:   Modelo explica 50% da variância
R² = 1.0:   Ajuste perfeito (todos pontos na reta)
```

**Classificação:**
```
R² < 0.3:     Ajuste fraco
0.3 ≤ R² < 0.7: Ajuste moderado
R² ≥ 0.7:     Ajuste forte
```

**Exemplo:**
```
TSS = 1000
RSS = 200

R² = 1 - 200/1000 = 0.8 = 80%

Interpretação: Modelo explica 80% da variação nas notas!
```

### **2.6 Hipóteses do Modelo Linear**

**Para inferência válida:**
1. **Linearidade:** Relação é linear
2. **Independência:** Observações independentes
3. **Homocedasticidade:** Variância constante dos erros
4. **Normalidade:** Erros seguem distribuição normal
5. **Sem multicolinearidade:** (regressão múltipla)

---

## **3. 📈 Regressão Linear Múltipla**

### **3.1 Definição**

Modelo com **múltiplas variáveis preditoras**:
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

**Forma Matricial:**
```
Y = Xβ + ε

onde:
• Y: vetor n×1 de respostas
• X: matriz n×(p+1) de preditores
• β: vetor (p+1)×1 de coeficientes
• ε: vetor n×1 de erros
```

### **3.2 Solução dos Mínimos Quadrados**

**Equação Normal:**
```
β̂ = (XᵀX)⁻¹XᵀY
```

**Interpretação dos Coeficientes:**
```
βⱼ: mudança em Y para aumento unitário em xⱼ,
    mantendo todas outras variáveis constantes
```

### **3.3 Exemplo**

**Predizer preço de casa:**
```
Preço = β₀ + β₁×Área + β₂×Quartos + β₃×Idade
```

**Resultados Hipotéticos:**
```
Preço = 50000 + 1000×Área + 20000×Quartos - 500×Idade

Interpretação:
• Casa base: R$ 50.000
• Cada m² adicional: +R$ 1.000
• Cada quarto adicional: +R$ 20.000
• Cada ano mais velha: -R$ 500
```

### **3.4 R² Ajustado**

Penaliza adição de variáveis irrelevantes:
```
R²ₐⱼ = 1 - (1-R²)×(n-1)/(n-p-1)

onde:
• n: número de observações
• p: número de preditores
```

**Por que usar?**
> R² sempre aumenta ao adicionar variáveis, mesmo irrelevantes. R²ₐⱼ só aumenta se nova variável melhora significativamente o modelo.

---

## **4. 🔄 Regressão Polinomial**

### **4.1 Definição**

Ajusta **polinômios** aos dados:
```
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε
```

**Graus Comuns:**
- **Grau 1:** Linear (reta)
- **Grau 2:** Quadrático (parábola)
- **Grau 3:** Cúbico
- **Grau n:** Polinômio de ordem n

### **4.2 Quando Usar?**

**Indicadores de Não-Linearidade:**
```
• Gráfico de resíduos mostra padrão
• Relação claramente curva
• Conhecimento do domínio sugere não-linearidade
```

**Exemplo Visual:**
```
Linear (ruim):          Quadrático (melhor):
    ●                       ●
   ●                       ●  ╱╲
  ●  /                    ●  ╱  ╲
 ●  /                    ●  ╱    ╲
●  /                    ●  ╱      ╲
```

### **4.3 Escolha do Grau**

**Trade-off:**
```
Grau baixo → Underfitting (subajuste)
Grau alto  → Overfitting (sobreajuste)
```

**Métodos de Seleção:**
1. **Validação Cruzada:** Erro em dados de teste
2. **Critérios de Informação:** AIC, BIC
3. **Conhecimento do Domínio:** Base teórica
4. **Análise de Resíduos:** Padrões residuais

**Exemplo:**
```
Dados com 10 pontos:
• Grau 1: R² = 0.70  ← Pode estar subajustado
• Grau 2: R² = 0.92  ← Bom equilíbrio
• Grau 3: R² = 0.94  ← Melhora marginal
• Grau 9: R² = 1.00  ← Passa por todos, mas overfit!
```

### **4.4 Implementação**

**Transformação:**
```python
# Criar features polinomiais
X_poly = [x, x², x³, ...]

# Depois aplicar regressão linear
y = β₀ + β₁×X_poly[:, 0] + β₂×X_poly[:, 1] + ...
```

**Nota:** Regressão polinomial é **linear nos parâmetros**, então usamos mínimos quadrados lineares!

### **4.5 Problemas Potenciais**

**Oscilações de Runge:**
```
Polinômios de alto grau podem oscilar
violentamente entre pontos de dados
```

**Extrapolação Perigosa:**
```
Polinômios podem divergir rapidamente
fora do intervalo de dados observados
```

**Multicolinearidade:**
```
x, x², x³, ... são altamente correlacionados
Solução: Usar polinômios ortogonais
```

---

## **5. 🔧 Interpolação**

### **5.1 Diferença: Interpolação vs. Regressão**

| **Aspecto** | **Interpolação** | **Regressão** |
|-------------|------------------|---------------|
| **Passa pelos pontos** | ✅ Exato | ❌ Aproximado |
| **Considera ruído** | ❌ Não | ✅ Sim |
| **Suavização** | Não | Sim |
| **Graus de liberdade** | n-1 (n pontos) | p (parâmetros) |
| **Uso** | Dados sem ruído | Dados com ruído |

### **5.2 Interpolação Linear por Partes**

Conecta pontos consecutivos com **segmentos de reta**.

**Vantagens:**
- ✅ Simples e rápido
- ✅ Sempre funciona
- ✅ Não oscila

**Desvantagens:**
- ❌ Não diferenciável nos nós
- ❌ Aparência "quebrada"

### **5.3 Interpolação de Lagrange**

**Fórmula:**
```
P(x) = Σ yᵢ × Lᵢ(x)

onde:
Lᵢ(x) = ∏ (x - xⱼ) / (xᵢ - xⱼ)  para j≠i
        j
```

**Características:**
- Polinômio de grau n-1 para n pontos
- Passa exatamente por todos os pontos
- Pode oscilar para muitos pontos

**Exemplo (2 pontos):**
```
Pontos: (1, 2), (3, 8)

L₁(x) = (x - 3) / (1 - 3) = (x - 3) / (-2)
L₂(x) = (x - 1) / (3 - 1) = (x - 1) / 2

P(x) = 2×L₁(x) + 8×L₂(x)
     = 2×(x-3)/(-2) + 8×(x-1)/2
     = -(x-3) + 4(x-1)
     = -x + 3 + 4x - 4
     = 3x - 1

Verificação:
P(1) = 3×1 - 1 = 2 ✓
P(3) = 3×3 - 1 = 8 ✓
```

### **5.4 Splines Cúbicos**

**Polinômio cúbico por partes** que é suave nos nós.

**Propriedades:**
- Cúbico entre cada par de pontos consecutivos
- Função contínua (C⁰)
- Primeira derivada contínua (C¹)
- Segunda derivada contínua (C²)

**Vantagens:**
- ✅ **Suave** (C² contínuo)
- ✅ Não oscila como polinômios de alto grau
- ✅ Flexível e preciso
- ✅ Padrão em gráficos e CAD

**Tipos:**
- **Natural Spline:** Segunda derivada = 0 nas extremidades
- **Clamped Spline:** Primeira derivada especificada nas extremidades
- **Not-a-Knot:** Terceira derivada contínua em x₂ e xₙ₋₁

### **5.5 Aplicações de Interpolação**

- **Gráficos:** Suavização de curvas
- **Animação:** Interpolação de keyframes
- **Processamento de Sinais:** Reamostragem
- **CAD/CAM:** Design de curvas suaves
- **Geofísica:** Interpolação de dados espaciais

---

## **6. 📊 Avaliação de Modelos**

### **6.1 Métricas de Erro**

**Mean Squared Error (MSE):**
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

Penaliza erros grandes quadraticamente
```

**Root Mean Squared Error (RMSE):**
```
RMSE = √MSE

Mesma unidade que Y, mais interpretável
```

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|

Menos sensível a outliers que MSE
```

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (100/n) × Σ|yᵢ - ŷᵢ| / |yᵢ|

Erro relativo em percentual
```

**Comparação:**
```
Dados: y = [100, 110, 120]
Predições: ŷ = [98, 115, 118]

Erros: [-2, 5, -2]

MSE = (4 + 25 + 4) / 3 = 11
RMSE = √11 ≈ 3.32
MAE = (2 + 5 + 2) / 3 = 3
MAPE = (2/100 + 5/110 + 2/120) × 100/3 ≈ 2.1%
```

### **6.2 Análise de Resíduos**

**Resíduos:**
```
eᵢ = yᵢ - ŷᵢ
```

**Gráficos Diagnósticos:**

**1. Resíduos vs. Valores Ajustados**
```
Ideal: pontos aleatórios em torno de zero
Problema: padrão indica não-linearidade
```

**2. Q-Q Plot**
```
Ideal: pontos na diagonal
Problema: desvios indicam não-normalidade
```

**3. Resíduos vs. Leverage**
```
Identifica pontos influentes
```

### **6.3 Validação**

**Validação Holdout:**
```
• 70-80% dados de treino
• 20-30% dados de teste
• Avaliar em dados não vistos
```

**Validação Cruzada K-Fold:**
```
• Dividir dados em K partes
• Treinar em K-1, testar em 1
• Repetir K vezes
• Média dos erros
```

**Leave-One-Out (LOO):**
```
• K = n (cada ponto é fold)
• Máxima utilização dos dados
• Computacionalmente caro
```

---

## **7. 🎯 Modelos Não-Lineares**

### **7.1 Regressão Não-Linear**

Quando a relação não é linear nos **parâmetros**:
```
y = f(x, β) + ε

onde f não é linear em β
```

**Exemplos:**
```
Exponencial:  y = β₀ × e^(β₁x)
Logística:    y = L / (1 + e^(-k(x-x₀)))
Potência:     y = β₀ × x^β₁
```

### **7.2 Método de Otimização**

Não há solução analítica, usa **otimização iterativa**:

**Métodos Comuns:**
- **Gradiente Descendente**
- **Levenberg-Marquardt**
- **Gauss-Newton**
- **Trust Region**

**Processo:**
```
1. Chute inicial para β
2. Calcular erro (RSS)
3. Atualizar β para reduzir erro
4. Repetir até convergência
```

### **7.3 Exemplo: Crescimento Exponencial**

**Modelo:**
```
y = a × e^(bx)
```

**Linearização (truque):**
```
log(y) = log(a) + bx

Fica linear em log(y)!
Aplicar regressão linear em log(y)
```

**Dados:**
```
x: [0, 1, 2, 3, 4]
y: [1, 2.7, 7.4, 20.1, 54.6]
```

**Solução:**
```
log(y): [0, 1, 2, 3, 4]  (aproximadamente)

Regressão: log(y) = 0 + 1×x
Portanto: y = e^0 × e^(1×x) = e^x

Verificação:
e^0 = 1 ✓
e^1 ≈ 2.7 ✓
e^2 ≈ 7.4 ✓
...
```

---

## **8. 🚀 Técnicas Avançadas**

### **8.1 Regularização**

Adiciona **penalidade** aos coeficientes para prevenir overfitting.

**Ridge Regression (L2):**
```
Minimizar: RSS + λ × Σβⱼ²

Encolhe coeficientes suavemente
```

**Lasso Regression (L1):**
```
Minimizar: RSS + λ × Σ|βⱼ|

Força alguns coeficientes a zero (seleção de features)
```

**Elastic Net:**
```
Minimizar: RSS + λ₁×Σ|βⱼ| + λ₂×Σβⱼ²

Combinação de Ridge e Lasso
```

### **8.2 Regressão Robusta**

Menos sensível a **outliers**.

**Métodos:**
- **Least Absolute Deviations (LAD):** Minimiza MAE
- **Huber Regression:** Híbrido quadrático/absoluto
- **RANSAC:** Amostra aleatória de consenso
- **Theil-Sen:** Mediana das inclinações

### **8.3 Regressão Não-Paramétrica**

Não assume forma funcional específica.

**Métodos:**
- **Loess/Lowess:** Regressão local ponderada
- **Spline Smoothing:** Splines com regularização
- **Kernel Regression:** Média ponderada local
- **Gaussian Process Regression:** Distribuição sobre funções

---

## **9. 🧮 Exercícios Resolvidos**

### **Exercício 1: Regressão Linear Simples**
**Dados:** X = [1, 2, 3, 4], Y = [2, 4, 5, 4]

**Solução:**
```
x̄ = 2.5, ȳ = 3.75

b₁ = [(1-2.5)(2-3.75) + ... ] / [(1-2.5)² + ...]
   = 5.5 / 5 = 1.1

b₀ = 3.75 - 1.1×2.5 = 1.0

Equação: ŷ = 1.0 + 1.1x

Predição para x=5:
ŷ = 1.0 + 1.1×5 = 6.5
```

### **Exercício 2: R²**
**Problema:** Calcular R² para:
```
Y observado: [2, 4, 5, 4]
Ŷ predito:   [2.1, 3.2, 4.3, 5.4]
```

**Solução:**
```
ȳ = 3.75

TSS = (2-3.75)² + (4-3.75)² + (5-3.75)² + (4-3.75)²
    = 3.0625 + 0.0625 + 1.5625 + 0.0625 = 4.75

RSS = (2-2.1)² + (4-3.2)² + (5-4.3)² + (4-5.4)²
    = 0.01 + 0.64 + 0.49 + 1.96 = 3.1

R² = 1 - 3.1/4.75 = 1 - 0.653 = 0.347 ≈ 34.7%
```

---

## **10. 💻 Implementação em Python**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import CubicSpline

# Dados de exemplo
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# 1. Regressão Linear
model_linear = LinearRegression()
model_linear.fit(X, y)
y_pred_linear = model_linear.predict(X)

print(f"Coeficientes: β₀={model_linear.intercept_:.2f}, β₁={model_linear.coef_[0]:.2f}")
print(f"R² Linear: {r2_score(y, y_pred_linear):.3f}")

# 2. Regressão Polinomial (grau 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression()
model_poly.fit(X_poly, y)
y_pred_poly = model_poly.predict(X_poly)

print(f"R² Polinomial: {r2_score(y, y_pred_poly):.3f}")

# 3. Interpolação com Spline Cúbico
cs = CubicSpline(X.ravel(), y)
X_smooth = np.linspace(1, 5, 100)
y_smooth = cs(X_smooth)

# 4. Visualização
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X, y, label='Dados')
plt.plot(X, y_pred_linear, 'r-', label='Linear')
plt.legend()
plt.title('Regressão Linear')

plt.subplot(1, 3, 2)
plt.scatter(X, y, label='Dados')
plt.plot(X, y_pred_poly, 'g-', label='Polinomial (grau 2)')
plt.legend()
plt.title('Regressão Polinomial')

plt.subplot(1, 3, 3)
plt.scatter(X, y, label='Dados')
plt.plot(X_smooth, y_smooth, 'b-', label='Spline Cúbico')
plt.legend()
plt.title('Interpolação Spline')

plt.tight_layout()
plt.show()
```

---

## **11. 🔗 Recursos Adicionais**

### **Livros Recomendados**
- **Introduction to Statistical Learning** - James et al.
- **The Elements of Statistical Learning** - Hastie, Tibshirani & Friedman
- **Numerical Methods** - Press et al.
- **Applied Regression Analysis** - Draper & Smith

### **Ferramentas Online**
- [Curve Fitting Tool](https://www.desmos.com/calculator)
- [Wolfram Alpha](https://www.wolframalpha.com/)
- [GeoGebra](https://www.geogebra.org/)

### **Bibliotecas Python**
```python
# Regressão
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.api import OLS

# Interpolação
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline

# Otimização
from scipy.optimize import curve_fit, least_squares
```

---

**Voltar para:** [Estatística](../README.md) | [Notebooks](../../README.md)

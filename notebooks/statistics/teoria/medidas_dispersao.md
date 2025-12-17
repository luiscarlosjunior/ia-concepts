# Medidas de Dispersão 📏

As **medidas de dispersão** (ou variabilidade) quantificam o quanto os dados estão espalhados em relação à medida de tendência central. Enquanto média, mediana e moda nos dizem "onde" os dados estão, as medidas de dispersão nos dizem "quão espalhados" eles estão.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 Por Que Medir Dispersão?**

Considere dois conjuntos de dados com a **mesma média** (50):

```
Grupo A: [48, 49, 50, 51, 52]     → Baixa dispersão
Grupo B: [10, 30, 50, 70, 90]     → Alta dispersão

Ambos: Média = 50
```

**Importância:**
- ✅ Complementa medidas de tendência central
- ✅ Indica confiabilidade da média
- ✅ Fundamental para inferência estatística
- ✅ Crucial para análise de risco e controle de qualidade
- ✅ Base para muitos algoritmos de ML

---

## **2. 📊 Amplitude (Range)**

### **2.1 Definição**

A **amplitude** é a diferença entre o maior e o menor valor.

**Fórmula:**
```
R = Xₘₐₓ - Xₘᵢₙ
```

**Exemplo:**
```
Temperaturas: [15, 18, 20, 22, 25]

R = 25 - 15 = 10°C
```

### **2.2 Vantagens e Desvantagens**

**✅ Vantagens:**
- Extremamente fácil de calcular
- Intuitiva e compreensível
- Útil para controle de qualidade rápido

**❌ Desvantagens:**
- **Extremamente sensível a outliers**
- Usa apenas 2 valores (ignora todos os outros)
- Aumenta com tamanho da amostra
- Não tem boas propriedades estatísticas

**Exemplo de Problema:**
```
Dados originais: [10, 11, 12, 13, 14]
R = 14 - 10 = 4

Com outlier: [10, 11, 12, 13, 100]
R = 100 - 10 = 90  ← Mudança drástica!
```

### **2.3 Aplicações**

- **Meteorologia:** Variação de temperatura diária
- **Finanças:** Preço máximo e mínimo de ação
- **Controle de Qualidade:** Tolerância de medições
- **Estatística Descritiva:** Visão inicial rápida

---

## **3. 📈 Variância**

### **3.1 Definição**

A **variância** mede a dispersão média dos dados em relação à média, usando o quadrado das distâncias.

**Variância Populacional (σ²):**
```
σ² = Σ(xᵢ - μ)² / N

onde:
• σ² (sigma ao quadrado): variância populacional
• μ: média populacional
• N: tamanho da população
```

**Variância Amostral (s²):**
```
s² = Σ(xᵢ - x̄)² / (n-1)

onde:
• s²: variância amostral
• x̄: média amostral
• n: tamanho da amostra
• (n-1): correção de Bessel (graus de liberdade)
```

**Por que (n-1)?**
> A correção de Bessel torna s² um **estimador não-viesado** de σ². Usando n subestimaria a variância populacional.

### **3.2 Cálculo Passo a Passo**

**Dados:** [2, 4, 6, 8, 10]

```
Passo 1: Calcular a média
x̄ = (2 + 4 + 6 + 8 + 10) / 5 = 30/5 = 6

Passo 2: Calcular desvios
(2-6) = -4
(4-6) = -2
(6-6) = 0
(8-6) = 2
(10-6) = 4

Passo 3: Elevar ao quadrado
(-4)² = 16
(-2)² = 4
(0)² = 0
(2)² = 4
(4)² = 16

Passo 4: Somar
Σ(xᵢ - x̄)² = 16 + 4 + 0 + 4 + 16 = 40

Passo 5: Dividir por (n-1)
s² = 40 / (5-1) = 40/4 = 10
```

### **3.3 Propriedades Matemáticas**

#### **Propriedade 1: Fórmula Alternativa**
```
σ² = E[X²] - (E[X])²
   = Média dos quadrados - Quadrado da média
```

**Exemplo:**
```
Dados: [2, 4, 6, 8, 10]

E[X] = 6
E[X²] = (4 + 16 + 36 + 64 + 100)/5 = 44

σ² = 44 - 6² = 44 - 36 = 8  (para população)
```

#### **Propriedade 2: Transformações Lineares**
```
Se Y = aX + b, então:
Var(Y) = a² × Var(X)

Nota: o termo constante b não afeta a variância!
```

**Exemplo:**
```
Converter Celsius para Fahrenheit: F = 1.8C + 32

Se Var(C) = 25:
Var(F) = 1.8² × 25 = 3.24 × 25 = 81
```

#### **Propriedade 3: Variância de Soma**
```
Para variáveis independentes:
Var(X + Y) = Var(X) + Var(Y)
Var(X - Y) = Var(X) + Var(Y)  (note que é soma!)
```

### **3.4 Interpretação**

**Unidades:**
- Variância está em **unidades ao quadrado**
- Se dados em metros, variância em metros²
- Dificulta interpretação direta

**Exemplo:**
```
Alturas (cm): [160, 165, 170, 175, 180]
Variância ≈ 62.5 cm²  ← O que significa 62.5 cm²?
```

### **3.5 Aplicações**

- **Estatística:** Base para testes de hipóteses
- **Finanças:** Medida de risco (volatilidade)
- **Machine Learning:** Regularização, feature selection
- **Controle de Qualidade:** Análise de processo

---

## **4. 📏 Desvio Padrão**

### **4.1 Definição**

O **desvio padrão** é a raiz quadrada da variância, trazendo a medida de volta às unidades originais.

**Fórmula:**
```
σ = √σ²     (populacional)
s = √s²     (amostral)
```

**Exemplo:**
```
Alturas (cm): [160, 165, 170, 175, 180]
Variância = 62.5 cm²
Desvio Padrão = √62.5 ≈ 7.9 cm  ← Interpretável!

Interpretação: Em média, as alturas desviam 7.9 cm da média.
```

### **4.2 Interpretação com Regra Empírica**

Para distribuições **aproximadamente normais**:

**Regra 68-95-99.7:**
```
• μ ± 1σ contém aproximadamente 68% dos dados
• μ ± 2σ contém aproximadamente 95% dos dados
• μ ± 3σ contém aproximadamente 99.7% dos dados
```

**Visualização:**
```
        │       68%
        │   ├─────────┤
        │     95%
        │ ├───────────────┤
        │      99.7%
        │├─────────────────────┤
    ────┼─────────────────────────
       μ-3σ  μ-2σ  μ-σ  μ  μ+σ  μ+2σ  μ+3σ
```

**Exemplo:**
```
Altura média = 170 cm, σ = 8 cm

• 68% das pessoas: entre 162 cm e 178 cm
• 95% das pessoas: entre 154 cm e 186 cm
• 99.7% das pessoas: entre 146 cm e 194 cm
```

### **4.3 Vantagens e Desvantagens**

**✅ Vantagens:**
- **Mesma unidade dos dados**
- Interpretação intuitiva
- Propriedades matemáticas bem definidas
- Amplamente usado em estatística

**❌ Desvantagens:**
- Sensível a outliers
- Não adequado para distribuições assimétricas
- Pode ser influenciado por valores extremos

### **4.4 Aplicações**

- **Finanças:** Volatilidade de ativos
- **Controle de Qualidade:** Six Sigma (6σ)
- **Padronização:** Z-scores
- **Machine Learning:** Normalização de features

---

## **5. 🎯 Coeficiente de Variação**

### **5.1 Definição**

O **coeficiente de variação (CV)** é a razão entre desvio padrão e média, expressa em porcentagem.

**Fórmula:**
```
CV = (σ / μ) × 100%    (populacional)
CV = (s / x̄) × 100%    (amostral)
```

### **5.2 Por Que Usar CV?**

**Problema:** Comparar dispersões em escalas diferentes

```
Grupo A (pesos em kg): x̄ = 70, s = 5
Grupo B (alturas em cm): x̄ = 170, s = 8

Questão: Qual grupo é mais disperso?
s não permite comparação direta (unidades diferentes)!
```

**Solução:** Coeficiente de Variação
```
CV_A = (5/70) × 100% = 7.14%
CV_B = (8/170) × 100% = 4.71%

Conclusão: Pesos são mais dispersos relativamente!
```

### **5.3 Interpretação**

**Classificação Geral:**
```
CV < 10%:    Baixa dispersão (dados homogêneos)
10% ≤ CV < 20%: Dispersão média
20% ≤ CV < 30%: Dispersão alta
CV ≥ 30%:    Dispersão muito alta (dados heterogêneos)
```

**Vantagens:**
- ✅ **Adimensional** (sem unidades)
- ✅ Permite comparação entre diferentes escalas
- ✅ Útil para avaliar precisão relativa

**Limitações:**
- ❌ Não definido quando média = 0
- ❌ Problemático para dados com valores negativos
- ❌ Sensível a outliers

### **5.4 Exemplo Prático**

**Comparando Precisão de Medições:**
```
Equipamento A:
• Mede distâncias curtas: μ = 10 cm, σ = 0.5 cm
• CV = (0.5/10) × 100% = 5%

Equipamento B:
• Mede distâncias longas: μ = 1000 cm, σ = 20 cm
• CV = (20/1000) × 100% = 2%

Conclusão: Equipamento B é mais preciso relativamente!
```

### **5.5 Aplicações**

- **Controle de Qualidade:** Comparar processos
- **Finanças:** Comparar risco de diferentes ativos
- **Medicina:** Variabilidade de medições clínicas
- **Pesquisa:** Avaliar consistência de experimentos

---

## **6. 📊 Quartis e Amplitude Interquartil (IQR)**

### **6.1 Quartis**

**Definição:**
Quartis dividem dados ordenados em **quatro partes iguais**.

```
Q₁ (1º Quartil): 25% dos dados
Q₂ (2º Quartil): 50% dos dados (= Mediana)
Q₃ (3º Quartil): 75% dos dados
```

**Cálculo:**
```
Dados ordenados: [1, 2, 3, 4, 5, 6, 7, 8, 9]

Posição Q₁ = 0.25 × (n+1) = 0.25 × 10 = 2.5
→ Q₁ = (2 + 3) / 2 = 2.5

Q₂ = 5  (mediana)

Posição Q₃ = 0.75 × (n+1) = 0.75 × 10 = 7.5
→ Q₃ = (7 + 8) / 2 = 7.5
```

### **6.2 Amplitude Interquartil (IQR)**

**Definição:**
```
IQR = Q₃ - Q₁
```

**Interpretação:**
- Contém os **50% centrais** dos dados
- **Robusta a outliers** (não afetada por valores extremos)

**Exemplo:**
```
Dados: [1, 2, 3, 4, 5, 6, 7, 8, 100]

Q₁ = 2.5
Q₃ = 7.5
IQR = 7.5 - 2.5 = 5

Nota: O outlier 100 não afeta o IQR!

Comparação:
• Desvio Padrão ≈ 31.8 (fortemente afetado por 100)
• IQR = 5 (robusto)
```

### **6.3 Boxplot (Diagrama de Caixa)**

Representação visual usando quartis:

```
        outlier
           ○
           │
     ┌─────┴─────┐
     │           │
─────┤     │     ├─────
  mín│  Q₁ Q₂ Q₃ │máx
     │           │
     └───────────┘
     
Elementos:
• Caixa: Q₁ a Q₃ (IQR)
• Linha central: Mediana (Q₂)
• Whiskers (bigodes): 1.5×IQR de Q₁ e Q₃
• Pontos fora: Outliers
```

### **6.4 Detecção de Outliers com IQR**

**Método de Tukey:**
```
Lower Fence = Q₁ - 1.5 × IQR
Upper Fence = Q₃ + 1.5 × IQR

Outliers: valores fora de [Lower Fence, Upper Fence]
```

**Exemplo:**
```
Q₁ = 25, Q₃ = 75, IQR = 50

Lower Fence = 25 - 1.5×50 = 25 - 75 = -50
Upper Fence = 75 + 1.5×50 = 75 + 75 = 150

Outliers: valores < -50 ou > 150
```

### **6.5 Vantagens do IQR**

**✅ Comparado ao Desvio Padrão:**
- Robusto a outliers
- Apropriado para distribuições assimétricas
- Não assume normalidade

**Aplicações:**
- Análise exploratória de dados
- Detecção de outliers
- Dados com distribuições não-normais
- Controle de qualidade robusto

---

## **7. 🌐 Medidas Multivariadas**

### **7.1 Covariância**

Mede como duas variáveis **variam juntas**.

**Fórmula:**
```
Cov(X,Y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / (n-1)
```

**Interpretação:**
```
Cov(X,Y) > 0:  X e Y tendem a aumentar juntas
Cov(X,Y) = 0:  Sem relação linear
Cov(X,Y) < 0:  Quando X aumenta, Y diminui
```

**Problema:** Unidades da covariância dependem das unidades de X e Y.

**Exemplo:**
```
X = Horas de estudo: [1, 2, 3, 4, 5]
Y = Nota: [50, 60, 70, 80, 90]

Cov(X,Y) = 25 "horas×pontos"  ← Difícil interpretar!
```

### **7.2 Correlação de Pearson**

Versão **normalizada** da covariância.

**Fórmula:**
```
ρ(X,Y) = Cov(X,Y) / (σₓ × σᵧ)

-1 ≤ ρ ≤ 1
```

**Interpretação:**
```
ρ = +1:  Correlação linear positiva perfeita
ρ = 0:   Sem correlação linear
ρ = -1:  Correlação linear negativa perfeita
```

**Classificação:**
```
|ρ| < 0.3:    Correlação fraca
0.3 ≤ |ρ| < 0.7: Correlação moderada
|ρ| ≥ 0.7:    Correlação forte
```

**Exemplo:**
```
Continuando exemplo anterior:

σₓ = 1.58
σᵧ = 15.81

ρ = 25 / (1.58 × 15.81) = 25 / 25 = 1.0

Conclusão: Correlação perfeita!
```

### **7.3 Distância de Mahalanobis**

Medida de distância que considera **covariância** entre variáveis.

**Fórmula:**
```
D = √((x - μ)ᵀ Σ⁻¹ (x - μ))

onde:
• x: vetor de observação
• μ: vetor de médias
• Σ: matriz de covariância
```

**Por que usar?**
- Considera correlações entre variáveis
- Unidades independentes
- Útil para detecção de outliers multivariados

**Aplicações:**
- Detecção de anomalias multivariadas
- Classificação (Análise Discriminante)
- Teste de normalidade multivariada

---

## **8. 📊 Comparação das Medidas**

| **Medida** | **Robusta a Outliers** | **Unidades** | **Uso Principal** |
|------------|----------------------|--------------|-------------------|
| **Amplitude** | ❌ Não | Mesmas dos dados | Visão rápida |
| **Variância** | ❌ Não | Quadrado das originais | Base matemática |
| **Desvio Padrão** | ❌ Não | Mesmas dos dados | Dispersão interpretável |
| **CV** | ❌ Não | Adimensional (%) | Comparações relativas |
| **IQR** | ✅ Sim | Mesmas dos dados | Dados assimétricos |

---

## **9. 🎓 Aplicações em Machine Learning**

### **9.1 Feature Scaling**

**Padronização (Z-score):**
```python
z = (x - μ) / σ

# Resultado: média = 0, desvio padrão = 1
```

**Quando usar:** Algoritmos sensíveis a escala (KNN, SVM, Redes Neurais)

### **9.2 Feature Selection**

**Baixa Variância = Feature Pouco Informativa**
```python
# Remover features com variância < threshold
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
X_new = selector.fit_transform(X)
```

### **9.3 Detecção de Anomalias**

**Método Z-score:**
```python
z_scores = (data - mean) / std
outliers = data[abs(z_scores) > 3]
```

**Método IQR:**
```python
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
```

### **9.4 Regularização**

**Ridge Regression:** Penaliza alta variância dos coeficientes
```python
# Minimiza: RSS + α × Σβ²
```

---

## **10. 🧮 Exercícios Resolvidos**

### **Exercício 1: Cálculo Completo**
**Problema:** Calcule todas as medidas de dispersão para: [2, 4, 6, 8, 10]

**Solução:**
```
Média: x̄ = 6

Amplitude:
R = 10 - 2 = 8

Variância:
s² = [(2-6)² + (4-6)² + (6-6)² + (8-6)² + (10-6)²] / 4
   = [16 + 4 + 0 + 4 + 16] / 4
   = 40 / 4 = 10

Desvio Padrão:
s = √10 ≈ 3.16

Coeficiente de Variação:
CV = (3.16 / 6) × 100% ≈ 52.7%

Quartis:
Q₁ = 4, Q₃ = 8
IQR = 8 - 4 = 4
```

### **Exercício 2: Comparação**
**Problema:** Qual grupo é mais homogêneo?
```
Grupo A: x̄ = 100, s = 15
Grupo B: x̄ = 50, s = 10
```

**Solução:**
```
CV_A = (15/100) × 100% = 15%
CV_B = (10/50) × 100% = 20%

Conclusão: Grupo A é mais homogêneo (menor CV)
```

### **Exercício 3: Outliers**
**Problema:** Detecte outliers usando IQR: [10, 12, 14, 15, 16, 18, 20, 25, 100]

**Solução:**
```
Q₁ = 13, Q₃ = 22.5
IQR = 22.5 - 13 = 9.5

Lower Fence = 13 - 1.5×9.5 = -1.25
Upper Fence = 22.5 + 1.5×9.5 = 36.75

Outliers: 100 (> 36.75)
```

---

## **11. 💻 Implementação em Python**

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Dados de exemplo
dados = np.array([2, 4, 6, 8, 10])

# Medidas de Dispersão
amplitude = np.ptp(dados)  # Peak to peak
variancia = np.var(dados, ddof=1)  # ddof=1 para amostra
desvio_padrao = np.std(dados, ddof=1)
cv = (desvio_padrao / np.mean(dados)) * 100

# Quartis e IQR
Q1 = np.percentile(dados, 25)
Q3 = np.percentile(dados, 75)
IQR = Q3 - Q1

# Detecção de outliers
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers = dados[(dados < lower_fence) | (dados > upper_fence)]

# Resultados
print(f"Amplitude: {amplitude}")
print(f"Variância: {variancia:.2f}")
print(f"Desvio Padrão: {desvio_padrao:.2f}")
print(f"CV: {cv:.2f}%")
print(f"IQR: {IQR}")
print(f"Outliers: {outliers}")

# Boxplot
plt.boxplot(dados)
plt.title("Boxplot dos Dados")
plt.show()
```

---

## **12. 🔗 Recursos Adicionais**

### **Livros Recomendados**
- **Estatística Básica** - Bussab & Morettin
- **Statistics for Data Science** - James et al.
- **Practical Statistics for Data Scientists** - Bruce & Bruce

### **Ferramentas Online**
- [StatKey](http://www.lock5stat.com/statkey/)
- [GeoGebra](https://www.geogebra.org/) - Visualizações
- [Desmos](https://www.desmos.com/calculator) - Calculadora

### **Bibliotecas Python**
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
```

---

**Voltar para:** [Estatística](../README.md) | [Notebooks](../../README.md)

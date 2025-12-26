# Medidas de Tendência Central 📊

As **medidas de tendência central** são valores que representam o "centro" ou "valor típico" de um conjunto de dados. Elas resumem uma distribuição em um único valor representativo, facilitando a compreensão e comparação de conjuntos de dados.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 O Que São Medidas de Tendência Central?**

São **estatísticas descritivas** que indicam onde os dados tendem a se concentrar. As três principais medidas são:

- **Média (Mean):** Valor médio aritmético
- **Mediana (Median):** Valor central quando dados ordenados
- **Moda (Mode):** Valor mais frequente

**Por que são importantes?**
- ✅ Resumem grandes conjuntos de dados em um único número
- ✅ Permitem comparações rápidas entre grupos
- ✅ Fundamentais para análise estatística e aprendizado de máquina
- ✅ Base para outras medidas estatísticas

---

## **2. 📈 Média Aritmética**

### **2.1 Definição**

A **média aritmética** (ou simplesmente média) é a soma de todos os valores dividida pelo número de observações.

**Fórmula (População):**
```
μ = Σxᵢ / N = (x₁ + x₂ + ... + xₙ) / N

onde:
• μ (mu): média populacional
• N: tamanho da população
• xᵢ: cada valor individual
```

**Fórmula (Amostra):**
```
x̄ = Σxᵢ / n = (x₁ + x₂ + ... + xₙ) / n

onde:
• x̄ (x-barra): média amostral
• n: tamanho da amostra
```

### **2.2 Exemplo Prático**

**Dados:** Notas de um aluno: 7, 8, 6, 9, 7

```
x̄ = (7 + 8 + 6 + 9 + 7) / 5
  = 37 / 5
  = 7.4
```

### **2.3 Propriedades Matemáticas**

#### **Propriedade 1: Linearidade**
```
Se Y = aX + b, então:
E[Y] = a×E[X] + b

Exemplo:
• Converter Celsius para Fahrenheit: F = 1.8C + 32
• Se média em Celsius = 20°C
• Média em Fahrenheit = 1.8×20 + 32 = 68°F
```

#### **Propriedade 2: Soma dos Desvios é Zero**
```
Σ(xᵢ - x̄) = 0

A soma das distâncias dos pontos à média sempre é zero!
```

#### **Propriedade 3: Minimização do Erro Quadrático**
```
A média minimiza: Σ(xᵢ - c)²

Ou seja, x̄ é o valor que minimiza a soma dos quadrados das distâncias.
```

### **2.4 Vantagens e Desvantagens**

**✅ Vantagens:**
- Usa todos os dados
- Propriedades matemáticas bem definidas
- Base para muitos métodos estatísticos
- Facilmente interpretável

**❌ Desvantagens:**
- **Sensível a outliers** (valores extremos)
- Pode não representar bem dados assimétricos
- Não existe para distribuições sem momento finito

**Exemplo de Sensibilidade a Outliers:**
```
Salários (em R$ mil):
Grupo A: 3, 3.5, 4, 4.2, 4.5
Média A = 3.84

Grupo B: 3, 3.5, 4, 4.2, 100  (CEO ganha muito!)
Média B = 22.94  ← NÃO representa bem o grupo!
```

### **2.5 Aplicações**

- **Educação:** Média de notas de turma
- **Economia:** PIB per capita, salário médio
- **Meteorologia:** Temperatura média
- **Machine Learning:** Normalização de dados, inicialização de pesos

---

## **3. 📏 Mediana**

### **3.1 Definição**

A **mediana** é o valor que divide o conjunto de dados ordenados em duas partes iguais: 50% dos dados estão abaixo e 50% acima.

**Cálculo:**

**Para n ímpar:**
```
Mediana = x₍ₙ₊₁₎/₂

Exemplo: [1, 3, 5, 7, 9]
n = 5
Mediana = x₍₅₊₁₎/₂ = x₃ = 5
```

**Para n par:**
```
Mediana = (x₍ₙ/₂₎ + x₍ₙ/₂₊₁₎) / 2

Exemplo: [1, 3, 5, 7, 9, 11]
n = 6
Mediana = (x₃ + x₄) / 2 = (5 + 7) / 2 = 6
```

### **3.2 Exemplo Prático**

**Salários (em R$ mil):** 3, 3.5, 4, 4.2, 100

```
Passo 1: Ordenar (já está ordenado)
Passo 2: Encontrar posição central
n = 5 (ímpar)
Posição = (5+1)/2 = 3

Mediana = 4
```

**Comparação:**
```
Média = 22.94    ← Puxada pelo outlier
Mediana = 4.0    ← Representa melhor o grupo típico
```

### **3.3 Propriedades**

#### **Robustez a Outliers**
```
A mediana não é afetada por valores extremos!

Dados: [1, 2, 3, 4, 5]
Mediana = 3

Dados com outlier: [1, 2, 3, 4, 1000]
Mediana = 3  (permanece igual!)
```

#### **Minimização do Erro Absoluto**
```
A mediana minimiza: Σ|xᵢ - c|

Ou seja, minimiza a soma das distâncias absolutas.
```

### **3.4 Quartis e Percentis**

A mediana é um caso especial de **quantil**.

**Quartis:**
```
Q₁ (1º Quartil): 25% dos dados
Q₂ (2º Quartil): 50% dos dados = Mediana
Q₃ (3º Quartil): 75% dos dados
```

**Percentis:**
```
P₁₀: 10% dos dados estão abaixo
P₅₀: 50% dos dados estão abaixo = Mediana
P₉₀: 90% dos dados estão abaixo
```

**Exemplo:**
```
Notas: [2, 3, 4, 5, 6, 7, 8, 9, 10]

Q₁ = 4    (25% das notas ≤ 4)
Q₂ = 6    (50% das notas ≤ 6) = Mediana
Q₃ = 8    (75% das notas ≤ 8)
```

### **3.5 Vantagens e Desvantagens**

**✅ Vantagens:**
- **Robusta a outliers**
- Sempre existe e é única
- Apropriada para dados ordinais
- Melhor para distribuições assimétricas

**❌ Desvantagens:**
- Não usa todos os dados
- Propriedades matemáticas menos convenientes
- Difícil calcular para dados agrupados
- Não adequada para operações algébricas

### **3.6 Aplicações**

- **Economia:** Renda mediana (melhor que média)
- **Imóveis:** Preço mediano de casas
- **Medicina:** Tempo mediano de sobrevivência
- **Processamento de Imagens:** Filtro de mediana para remover ruído

---

## **4. 🎯 Moda**

### **4.1 Definição**

A **moda** é o valor que ocorre com **maior frequência** no conjunto de dados.

**Características:**
- Pode não existir (distribuição uniforme)
- Pode ter múltiplas modas (bimodal, multimodal)
- Única medida apropriada para dados nominais

### **4.2 Tipos de Distribuições**

**Unimodal:** Uma moda
```
Dados: [1, 2, 2, 2, 3, 4, 5]
Moda = 2
```

**Bimodal:** Duas modas
```
Dados: [1, 2, 2, 2, 3, 4, 4, 4, 5]
Modas = 2 e 4
```

**Multimodal:** Mais de duas modas
```
Dados: [1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5]
Modas = 1, 3, 5
```

**Amodal:** Sem moda
```
Dados: [1, 2, 3, 4, 5]
Não há moda (todos igualmente frequentes)
```

### **4.3 Exemplo Prático**

**Cores de carros vendidos:**
```
Preto: 45
Branco: 38
Prata: 32
Vermelho: 25
Azul: 15

Moda = Preto (cor mais vendida)
```

### **4.4 Moda para Dados Contínuos**

Para dados contínuos, usamos **classes** e encontramos a **classe modal**.

**Exemplo: Alturas (cm)**
```
150-160: 5 pessoas
160-170: 12 pessoas  ← Classe modal
170-180: 8 pessoas
180-190: 3 pessoas

Moda ≈ 165 (centro da classe modal)
```

### **4.5 Vantagens e Desvantagens**

**✅ Vantagens:**
- Única medida para dados nominais
- Não afetada por outliers
- Fácil de entender e calcular
- Útil para identificar valores típicos em negócios

**❌ Desvantagens:**
- Pode não existir ou não ser única
- Ignora a maioria dos dados
- Não tem boas propriedades matemáticas
- Instável em amostras pequenas

### **4.6 Aplicações**

- **Varejo:** Tamanho de roupa mais vendido
- **Marketing:** Produto mais popular
- **Dados Categóricos:** Categoria mais comum
- **Análise de Falhas:** Tipo de falha mais frequente

---

## **5. 🔄 Outras Medidas de Tendência Central**

### **5.1 Média Ponderada**

Cada valor tem um **peso** diferente.

**Fórmula:**
```
x̄w = Σ(wᵢ × xᵢ) / Σwᵢ

onde wᵢ são os pesos
```

**Exemplo: Cálculo de Média Final**
```
Provas:
P1 = 7  (peso 2)
P2 = 8  (peso 3)
P3 = 6  (peso 5)

Média = (2×7 + 3×8 + 5×6) / (2+3+5)
      = (14 + 24 + 30) / 10
      = 68 / 10
      = 6.8
```

**Aplicações:**
- Índices econômicos (inflação, bolsa)
- Médias escolares com pesos
- Estimativas com diferentes níveis de confiança

### **5.2 Média Geométrica**

Usada para **taxas de crescimento** e **proporções**.

**Fórmula:**
```
MG = ⁿ√(x₁ × x₂ × ... × xₙ) = (∏xᵢ)^(1/n)
```

**Exemplo: Taxa de Crescimento**
```
Crescimento anual de vendas:
Ano 1: +10% → 1.10
Ano 2: +20% → 1.20
Ano 3: -5%  → 0.95

MG = ³√(1.10 × 1.20 × 0.95)
   = ³√1.254
   ≈ 1.078

Taxa média anual = 7.8%
```

**Propriedade Importante:**
```
MG ≤ MA (Média Aritmética)

Igualdade ocorre apenas quando todos os valores são iguais.
```

**Aplicações:**
- Finanças: retorno médio de investimentos
- Biologia: taxas de crescimento populacional
- Geometria: lado médio de formas geométricas

### **5.3 Média Harmônica**

Usada para **médias de taxas** e **velocidades**.

**Fórmula:**
```
MH = n / Σ(1/xᵢ)
```

**Exemplo: Velocidade Média**
```
Viagem de 100 km:
• Ida: 50 km a 100 km/h
• Volta: 50 km a 50 km/h

Velocidade média (ERRADO usar média aritmética):
(100 + 50)/2 = 75 km/h ✗

Velocidade média (CORRETO usar média harmônica):
MH = 2 / (1/100 + 1/50)
   = 2 / (0.01 + 0.02)
   = 2 / 0.03
   ≈ 66.67 km/h ✓
```

**Relação entre as Médias:**
```
MH ≤ MG ≤ MA

(Desigualdade das Médias)
```

**Aplicações:**
- Física: velocidade média
- Finanças: P/E ratio médio
- Computação: throughput médio

### **5.4 Média Truncada (Trimmed Mean)**

Remove **outliers** antes de calcular a média.

**Procedimento:**
```
1. Ordenar os dados
2. Remover k% dos valores extremos (ambos os lados)
3. Calcular média dos valores restantes
```

**Exemplo: Média Truncada a 10%**
```
Dados: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
n = 10

Remover 10% de cada extremo:
• Remove 1 valor menor: 1
• Remove 1 valor maior: 100

Dados truncados: [2, 3, 4, 5, 6, 7, 8, 9]

Média truncada = (2+3+4+5+6+7+8+9) / 8 = 5.5

Comparação:
• Média original = 14.5
• Média truncada = 5.5  ← Mais robusta!
```

**Aplicações:**
- Olimpíadas: notas de juízes (remove máxima e mínima)
- Economia: taxas de juros médias
- Pesquisa: remove respostas extremas

---

## **6. 📊 Comparação e Escolha da Medida**

### **6.1 Relação entre Média, Mediana e Moda**

**Distribuição Simétrica:**
```
Média = Mediana = Moda

    │     ╱─╲
    │    ╱   ╲
    │   ╱     ╲
    │──────────────
       ↑
     Todas iguais
```

**Distribuição Assimétrica à Direita (Positiva):**
```
Moda < Mediana < Média

    │╲
    │ ╲
    │  ╲____
    │──────────────
      ↑  ↑   ↑
     Mo Me  Ma
```

**Distribuição Assimétrica à Esquerda (Negativa):**
```
Média < Mediana < Moda

    │      ╱
    │     ╱
    │____╱
    │──────────────
      ↑   ↑  ↑
     Ma  Me Mo
```

### **6.2 Guia de Decisão**

| **Situação** | **Melhor Medida** | **Motivo** |
|--------------|-------------------|------------|
| Distribuição simétrica sem outliers | **Média** | Usa todos os dados, propriedades matemáticas |
| Presença de outliers | **Mediana** | Robusta a valores extremos |
| Dados ordinais | **Mediana** | Não requer aritmética |
| Dados nominais | **Moda** | Única aplicável |
| Renda/salários | **Mediana** | Distribuição assimétrica |
| Preferência do consumidor | **Moda** | Valor mais comum |
| Taxas de crescimento | **Média Geométrica** | Multiplica fatores |
| Velocidades/taxas | **Média Harmônica** | Recíprocos |

### **6.3 Exemplo Completo**

**Salários de uma empresa (em R$ mil):**
```
[2.5, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5, 5.0, 15.0]
```

**Análise:**
```
Média = 4.87
Mediana = (3.8 + 4.0) / 2 = 3.9
Moda = Não há (todos únicos)

Interpretação:
• Média = R$ 4.870 (puxada pelo salário de R$ 15k)
• Mediana = R$ 3.900 (representa melhor o "trabalhador típico")
• Recomendação: Usar MEDIANA para reportar "salário típico"
```

---

## **7. 🎓 Aplicações em Machine Learning**

### **7.1 Pré-processamento de Dados**

**Normalização usando Média:**
```python
# Z-score normalization
z = (x - μ) / σ

# Resultado: média = 0, desvio padrão = 1
```

**Imputação de Valores Faltantes:**
```python
# Estratégias:
missing_value = mean(data)    # Média
missing_value = median(data)  # Mediana (mais robusta)
missing_value = mode(data)    # Moda (dados categóricos)
```

### **7.2 Detecção de Outliers**

**Método IQR (Interquartile Range):**
```
IQR = Q₃ - Q₁
Lower_bound = Q₁ - 1.5 × IQR
Upper_bound = Q₃ + 1.5 × IQR

Outliers: valores fora de [Lower_bound, Upper_bound]
```

### **7.3 Inicialização de Modelos**

**K-Means Clustering:**
```python
# Inicializa centroides usando média de subconjuntos
centroids = [mean(subset) for subset in random_subsets]
```

**Redes Neurais:**
```python
# Inicialização Xavier/Glorot usa média = 0
weights ~ N(0, σ²)
```

---

## **8. 🧮 Exercícios Resolvidos**

### **Exercício 1: Cálculo Básico**
**Problema:** Calcule média, mediana e moda dos dados: [2, 4, 4, 5, 7, 9]

**Solução:**
```
Média:
x̄ = (2 + 4 + 4 + 5 + 7 + 9) / 6 = 31/6 ≈ 5.17

Mediana:
Dados ordenados: [2, 4, 4, 5, 7, 9]
n = 6 (par)
Mediana = (4 + 5) / 2 = 4.5

Moda:
Valor mais frequente = 4 (aparece 2 vezes)
```

### **Exercício 2: Comparação**
**Problema:** Compare as medidas para: [10, 20, 30, 40, 1000]

**Solução:**
```
Média = (10 + 20 + 30 + 40 + 1000) / 5 = 220
Mediana = 30 (valor central)
Moda = Não há

Interpretação:
• Média fortemente influenciada por 1000
• Mediana representa melhor o conjunto típico
• Use MEDIANA para reportar valor central
```

### **Exercício 3: Média Ponderada**
**Problema:** Calcule nota final com: Prova1=8(peso 3), Prova2=6(peso 2), Trabalho=9(peso 1)

**Solução:**
```
Média = (8×3 + 6×2 + 9×1) / (3+2+1)
      = (24 + 12 + 9) / 6
      = 45 / 6
      = 7.5
```

---

## **9. 💻 Implementação em Python**

```python
import numpy as np
from scipy import stats

# Dados de exemplo
dados = [2, 4, 4, 5, 7, 9, 100]

# Média
media = np.mean(dados)
print(f"Média: {media:.2f}")

# Mediana
mediana = np.median(dados)
print(f"Mediana: {mediana:.2f}")

# Moda
moda = stats.mode(dados, keepdims=True)
print(f"Moda: {moda.mode[0]}")

# Média Truncada (10%)
media_truncada = stats.trim_mean(dados, 0.1)
print(f"Média Truncada: {media_truncada:.2f}")

# Comparação
print("\nComparação:")
print(f"Média:           {media:.2f}  ← Puxada pelo outlier")
print(f"Mediana:         {mediana:.2f}  ← Robusta")
print(f"Média Truncada:  {media_truncada:.2f}  ← Compromisso")
```

---

## **10. 🔗 Recursos Adicionais**

### **Livros Recomendados**
- **Estatística Básica** - Bussab & Morettin
- **Statistics** - Freedman, Pisani & Purves
- **Think Stats** - Allen Downey
- **Practical Statistics for Data Scientists** - Bruce & Bruce

### **Ferramentas Online**
- [StatKey](http://www.lock5stat.com/statkey/) - Calculadora estatística
- [Khan Academy](https://www.khanacademy.org/) - Tutoriais
- [Wolfram Alpha](https://www.wolframalpha.com/) - Cálculos

### **Bibliotecas Python**
```python
import numpy as np           # Operações básicas
import pandas as pd          # DataFrames
from scipy import stats      # Estatística avançada
import matplotlib.pyplot as plt  # Visualização
import seaborn as sns        # Gráficos estatísticos
```

---

**Voltar para:** [Estatística](../README.md) | [Notebooks](../../README.md)

# Métodos Numéricos - Sistemas Lineares 🔢

A resolução de **sistemas de equações lineares** é um problema fundamental em matemática computacional, engenharia, ciência de dados e aprendizado de máquina. Este documento apresenta os principais métodos numéricos para resolver sistemas lineares Ax = b.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 O Problema**

**Sistema de Equações Lineares:**
```
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ = b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ = b₂
  ⋮       ⋮             ⋮      ⋮
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ = bₘ
```

**Forma Matricial:**
```
Ax = b

onde:
• A: matriz m×n de coeficientes
• x: vetor n×1 de incógnitas (solução)
• b: vetor m×1 de termos independentes
```

**Exemplo:**
```
2x + 3y = 8
4x - y = 2

Em forma matricial:
[2   3] [x]   [8]
[4  -1] [y] = [2]
```

### **1.2 Tipos de Sistemas**

**Por Dimensão:**
```
m = n: Sistema quadrado (n equações, n incógnitas)
m > n: Sistema sobredeterminado (mais equações que incógnitas)
m < n: Sistema subdeterminado (menos equações que incógnitas)
```

**Por Solução:**
```
• Consistente: tem pelo menos uma solução
  - Determinado: solução única
  - Indeterminado: infinitas soluções
• Inconsistente: sem solução
```

**Condições de Existência (Sistema Quadrado):**
```
det(A) ≠ 0: Solução única (sistema não-singular)
det(A) = 0: Sem solução ou infinitas soluções (singular)
```

### **1.3 Por Que Métodos Numéricos?**

**Limitações Analíticas:**
- Sistemas grandes (n > 1000)
- Matrizes esparsas
- Soluções aproximadas suficientes
- Eficiência computacional

**Classificação dos Métodos:**
1. **Métodos Diretos:** Solução exata (em aritmética exata)
   - Eliminação de Gauss, Decomposição LU, QR
2. **Métodos Iterativos:** Sequência convergente
   - Jacobi, Gauss-Seidel, Gradiente Conjugado

---

## **2. 📊 Métodos Diretos**

### **2.1 Eliminação de Gauss**

**Princípio:** Transformar A em matriz triangular superior.

**Processo:**
```
[A|b] → [Triangular Superior|b']

Depois resolver por substituição retroativa
```

**Algoritmo:**
```
Para k = 1 até n-1:
  Para i = k+1 até n:
    multiplicador mᵢₖ = aᵢₖ / aₖₖ
    Para j = k até n:
      aᵢⱼ = aᵢⱼ - mᵢₖ × aₖⱼ
    bᵢ = bᵢ - mᵢₖ × bₖ
```

**Exemplo:**
```
Sistema:
x + 2y + z = 4
2x + y + z = 3
x + y + 2z = 5

Matriz aumentada:
[1  2  1 | 4]
[2  1  1 | 3]
[1  1  2 | 5]

Passo 1: Eliminar x da linha 2 e 3
m₂₁ = 2/1 = 2
m₃₁ = 1/1 = 1

[1  2   1  | 4]
[0 -3  -1  |-5]  (L2 - 2×L1)
[0 -1   1  | 1]  (L3 - L1)

Passo 2: Eliminar y da linha 3
m₃₂ = -1/(-3) = 1/3

[1  2    1   | 4]
[0 -3   -1   |-5]
[0  0   2/3  | 2/3]  (L3 - (1/3)×L2)

Substituição Retroativa:
z = (2/3)/(2/3) = 1
y = (-5 - (-1)×1)/(-3) = 4/3
x = (4 - 2×(4/3) - 1) = 1/3

Solução: x = 1/3, y = 4/3, z = 1
```

**Complexidade:**
```
Eliminação: O(n³/3)
Substituição: O(n²)
Total: O(n³)
```

**Limitações:**
- Pivô zero causa divisão por zero
- Pivôs pequenos causam instabilidade numérica
- **Solução:** Pivoteamento parcial ou completo

### **2.2 Pivoteamento**

**Pivoteamento Parcial:**
```
Em cada etapa k, escolher pivô aₖₖ com maior |aᵢₖ|
(trocar linhas se necessário)
```

**Pivoteamento Completo:**
```
Escolher pivô com maior |aᵢⱼ| em toda submatriz
(trocar linhas e colunas)
```

**Por que é importante:**
```
Sem pivoteamento:
[10⁻⁴  1] [x]   [1]
[1     1] [y] = [2]

Com arredamento:
x = 0 (errado!)

Com pivoteamento:
[1     1] [y]   [2]
[10⁻⁴  1] [x] = [1]

x ≈ 1, y ≈ 1 (correto!)
```

### **2.3 Decomposição LU**

**Princípio:** Fatorar A = LU
```
L: matriz triangular inferior (Lower)
U: matriz triangular superior (Upper)
```

**Vantagem:** Resolver múltiplos sistemas com mesma A:
```
Ax = b₁  →  LUx = b₁
Ax = b₂  →  LUx = b₂
...

1. Fatorar A = LU uma vez: O(n³)
2. Para cada bᵢ: resolver Ly = bᵢ e Ux = y: O(n²)
```

**Algoritmo de Doolittle:**
```
L tem diagonal de 1s
U é obtida pela eliminação de Gauss

Para k = 1 até n:
  uₖⱼ = aₖⱼ - Σ(lₖₚ×uₚⱼ)  (j = k até n)
  lᵢₖ = (aᵢₖ - Σ(lᵢₚ×uₚₖ))/uₖₖ  (i = k+1 até n)
```

**Exemplo:**
```
A = [2  1  1]
    [4  3  3]
    [8  7  9]

L = [1    0    0]
    [2    1    0]
    [4    3    1]

U = [2  1  1]
    [0  1  1]
    [0  0  2]

Verificação: L×U = A ✓
```

**Variantes:**
- **Crout:** U tem diagonal de 1s
- **Cholesky:** Para matrizes simétricas positivas definidas (A = LLᵀ)

### **2.4 Decomposição QR**

**Princípio:** Fatorar A = QR
```
Q: matriz ortogonal (QᵀQ = I)
R: matriz triangular superior
```

**Vantagens:**
- **Mais estável numericamente** que LU
- Útil para problemas de mínimos quadrados
- Funciona para matrizes retangulares

**Métodos:**

**1. Gram-Schmidt:**
```
Ortogonaliza colunas de A sequencialmente

qᵢ = aᵢ - Σ(aᵢᵀqⱼ)qⱼ  (j < i)
qᵢ = qᵢ / ||qᵢ||
```

**2. Reflexões de Householder:**
```
Usa matrizes de reflexão para zerar elementos abaixo da diagonal

Hᵢ = I - 2vᵢvᵢᵀ

Mais estável que Gram-Schmidt
```

**3. Rotações de Givens:**
```
Rotaciona pares de elementos para zerar um por vez

Útil para matrizes esparsas
```

**Aplicação em Mínimos Quadrados:**
```
Minimizar ||Ax - b||²

Solução: x = (AᵀA)⁻¹Aᵀb

Com QR: A = QR
x = R⁻¹Qᵀb

Vantagem: Não precisa calcular AᵀA (melhor condicionamento)
```

### **2.5 Decomposição SVD**

**Singular Value Decomposition:**
```
A = UΣVᵀ

onde:
• U: m×m ortogonal (vetores singulares à esquerda)
• Σ: m×n diagonal (valores singulares σᵢ ≥ 0)
• V: n×n ortogonal (vetores singulares à direita)
```

**Solução de Mínimos Quadrados:**
```
x = VΣ⁺Uᵀb

onde Σ⁺ é pseudo-inversa:
Σ⁺ᵢᵢ = 1/σᵢ  se σᵢ ≠ 0
      = 0     se σᵢ = 0
```

**Vantagens:**
- **Mais estável** de todos os métodos
- Funciona para qualquer matriz
- Detecta rank deficiency
- Análise de condicionamento

**Aplicações:**
- PCA (Análise de Componentes Principais)
- Compressão de imagens
- Recomendação (Matrix Factorization)
- Redução de dimensionalidade

---

## **3. 🔄 Métodos Iterativos**

### **3.1 Quando Usar Métodos Iterativos?**

**Vantagens:**
- Eficientes para matrizes **grandes e esparsas**
- Não requerem armazenar matriz completa
- Podem parar com aproximação suficiente
- Paralelizáveis

**Desvantagens:**
- Podem não convergir
- Convergência pode ser lenta
- Requerem bom chute inicial

### **3.2 Método de Jacobi**

**Decomposição:** A = D + L + U
```
D: diagonal de A
L: triangular inferior estrita (abaixo da diagonal)
U: triangular superior estrita (acima da diagonal)
```

**Iteração:**
```
Dx⁽ᵏ⁺¹⁾ = b - (L + U)x⁽ᵏ⁾

ou elemento a elemento:

xᵢ⁽ᵏ⁺¹⁾ = (bᵢ - Σaᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ  (j ≠ i)
```

**Exemplo:**
```
Sistema:
4x + y = 15
x + 3y = 14

Iterações:
x⁽⁰⁾ = [0, 0]

x⁽¹⁾ = [(15 - 0)/4, (14 - 0)/3] = [3.75, 4.67]

x⁽²⁾ = [(15 - 4.67)/4, (14 - 3.75)/3] = [2.58, 3.42]

x⁽³⁾ = [(15 - 3.42)/4, (14 - 2.58)/3] = [2.90, 3.81]

...converge para x = 3, y = 4
```

**Convergência:**
```
Condição suficiente: A é estritamente diagonal dominante

|aᵢᵢ| > Σ|aᵢⱼ|  (j ≠ i) para todo i
```

### **3.3 Método de Gauss-Seidel**

**Melhoria:** Usa valores **já atualizados** na mesma iteração.

**Iteração:**
```
xᵢ⁽ᵏ⁺¹⁾ = (bᵢ - Σaᵢⱼxⱼ⁽ᵏ⁺¹⁾ - Σaᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ
              j<i           j>i

Usa x⁽ᵏ⁺¹⁾ para j < i (já calculados)
Usa x⁽ᵏ⁾ para j > i (ainda não calculados)
```

**Forma Matricial:**
```
(D + L)x⁽ᵏ⁺¹⁾ = b - Ux⁽ᵏ⁾
```

**Exemplo (mesmo sistema):**
```
x⁽⁰⁾ = [0, 0]

x₁⁽¹⁾ = (15 - 0)/4 = 3.75
x₂⁽¹⁾ = (14 - 3.75)/3 = 3.42  (usa x₁⁽¹⁾!)

x₁⁽²⁾ = (15 - 3.42)/4 = 2.90
x₂⁽²⁾ = (14 - 2.90)/3 = 3.70

x₁⁽³⁾ = (15 - 3.70)/4 = 2.83
x₂⁽³⁾ = (14 - 2.83)/3 = 3.72

...converge mais rápido que Jacobi!
```

**Convergência:**
```
• Mesma condição de Jacobi (diagonal dominante)
• Geralmente converge mais rápido que Jacobi
• Pode convergir quando Jacobi não converge
```

### **3.4 Método SOR (Successive Over-Relaxation)**

**Melhoria:** Adiciona fator de **relaxamento** ω.

**Iteração:**
```
x̃ᵢ = (bᵢ - Σaᵢⱼxⱼ⁽ᵏ⁺¹⁾ - Σaᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ
     j<i           j>i

xᵢ⁽ᵏ⁺¹⁾ = ω×x̃ᵢ + (1-ω)×xᵢ⁽ᵏ⁾

onde:
• ω = 1: Gauss-Seidel
• 1 < ω < 2: Over-relaxation (acelera)
• 0 < ω < 1: Under-relaxation (estabiliza)
```

**Escolha de ω:**
```
• Teórica: Depende de propriedades de A
• Prática: Experimentação (típico: 1.0-1.5)
• Ótimo: ω_ótimo ≈ 2/(1 + √(1-ρ²))
  onde ρ é raio espectral de Gauss-Seidel
```

### **3.5 Gradiente Conjugado**

**Para:** Sistemas simétricos positivos definidos (SPD).

**Princípio:** Minimizar função quadrática
```
f(x) = (1/2)xᵀAx - bᵀx

∇f(x) = Ax - b = 0  →  Ax = b
```

**Algoritmo:**
```
x⁽⁰⁾ = chute inicial
r⁽⁰⁾ = b - Ax⁽⁰⁾  (resíduo)
p⁽⁰⁾ = r⁽⁰⁾  (direção)

Para k = 0, 1, 2, ...
  αₖ = (r⁽ᵏ⁾ᵀr⁽ᵏ⁾) / (p⁽ᵏ⁾ᵀAp⁽ᵏ⁾)
  x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + αₖp⁽ᵏ⁾
  r⁽ᵏ⁺¹⁾ = r⁽ᵏ⁾ - αₖAp⁽ᵏ⁾
  βₖ = (r⁽ᵏ⁺¹⁾ᵀr⁽ᵏ⁺¹⁾) / (r⁽ᵏ⁾ᵀr⁽ᵏ⁾)
  p⁽ᵏ⁺¹⁾ = r⁽ᵏ⁺¹⁾ + βₖp⁽ᵏ⁾
```

**Propriedades:**
- **Teoricamente:** Converge em no máximo n iterações
- **Praticamente:** Aproximação boa em muito menos iterações
- **Eficiente:** Apenas produto matriz-vetor por iteração
- **Ideal:** Para matrizes grandes e esparsas

**Pré-condicionamento:**
```
Resolver M⁻¹Ax = M⁻¹b

onde M é matriz de pré-condicionamento:
• M ≈ A (aproxima A)
• M fácil de inverter
• Melhora condicionamento

Exemplo: M = diagonal de A
```

---

## **4. 📊 Análise de Erros e Condicionamento**

### **4.1 Tipos de Erro**

**Erro de Arredondamento:**
```
Computadores têm precisão finita
Operações introduzem pequenos erros
Erros se acumulam
```

**Erro de Truncamento:**
```
Métodos iterativos param antes da convergência
```

**Erro Total:**
```
||x_exato - x_computado||
```

### **4.2 Número de Condição**

**Definição:**
```
κ(A) = ||A|| × ||A⁻¹||

Para norma-2:
κ₂(A) = σₘₐₓ / σₘᵢₙ
```

**Interpretação:**
```
κ(A) ≈ 1:     Bem-condicionado
κ(A) ≈ 10³:   Mal-condicionado
κ(A) ≈ 10⁶:   Muito mal-condicionado
```

**Amplificação de Erro:**
```
Perturbação em b:
||Δx|| / ||x|| ≤ κ(A) × (||Δb|| / ||b||)

Se κ(A) = 10⁶ e erro em b é 10⁻⁸:
Erro em x pode ser 10⁻²!
```

**Exemplo:**
```
A = [1    1  ]    κ(A) ≈ 4 (bem-condicionado)
    [0  0.01]

B = [1    1   ]   κ(B) ≈ 4×10⁴ (mal-condicionado)
    [0  10⁻⁵]
```

### **4.3 Normas Vetoriais e Matriciais**

**Normas Vetoriais:**
```
||x||₁ = Σ|xᵢ|               (norma-1)
||x||₂ = √(Σxᵢ²)             (norma-2, euclidiana)
||x||∞ = max|xᵢ|             (norma-infinito)
```

**Normas Matriciais:**
```
||A||₁ = max_j Σ|aᵢⱼ|        (máximo da soma das colunas)
||A||₂ = √(λₘₐₓ(AᵀA))        (maior valor singular)
||A||∞ = max_i Σ|aᵢⱼ|        (máximo da soma das linhas)
||A||F = √(ΣΣaᵢⱼ²)           (Frobenius)
```

### **4.4 Critérios de Parada**

**Para Métodos Iterativos:**

**1. Resíduo:**
```
||Ax⁽ᵏ⁾ - b|| < tol
```

**2. Mudança Relativa:**
```
||x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾|| / ||x⁽ᵏ⁺¹⁾|| < tol
```

**3. Combinado:**
```
(||Ax⁽ᵏ⁾ - b|| < tol₁) E (||x⁽ᵏ⁺¹⁾ - x⁽ᵏ⁾|| < tol₂)
```

**4. Número Máximo de Iterações:**
```
k > k_max
```

---

## **5. 🎯 Matrizes Especiais**

### **5.1 Matrizes Esparsas**

**Definição:** Maioria dos elementos são zero.

**Armazenamento Eficiente:**

**COO (Coordinate):**
```
Armazena: (linha, coluna, valor) para elementos não-zeros
```

**CSR (Compressed Sparse Row):**
```
Armazena: valores, índices de colunas, ponteiros de linhas
Eficiente para operações por linha
```

**CSC (Compressed Sparse Column):**
```
Similar a CSR, mas por coluna
Eficiente para operações por coluna
```

**Exemplo:**
```
A = [1  0  0  2]
    [0  3  0  0]
    [4  0  5  0]
    [0  6  0  7]

COO:
rows = [0, 0, 1, 2, 2, 3, 3]
cols = [0, 3, 1, 0, 2, 1, 3]
vals = [1, 2, 3, 4, 5, 6, 7]
```

**Métodos Especiais:**
- Eliminação de Gauss com fill-in mínimo
- Métodos iterativos (muito eficientes!)
- Fatoração Cholesky esparsa

### **5.2 Matrizes Banda**

**Definição:** Elementos não-zeros concentrados perto da diagonal.

**Largura de Banda:**
```
b = max(|i-j|) para aᵢⱼ ≠ 0
```

**Exemplos:**
- **Tridiagonal:** b = 1
- **Pentadiagonal:** b = 2

**Vantagem:** Algoritmos O(n×b²) em vez de O(n³).

**Aplicações:**
- Diferenças finitas (EDPs)
- Splines
- Séries temporais (AR, MA)

### **5.3 Matrizes Simétricas Positivas Definidas**

**Propriedades:**
- A = Aᵀ (simétrica)
- xᵀAx > 0 para todo x ≠ 0 (positiva definida)
- Autovalores positivos
- Decomposição de Cholesky existe

**Fatoração de Cholesky:**
```
A = LLᵀ

onde L é triangular inferior

Vantagens:
• Metade do custo de LU
• Numericamente estável
• Única (com diagonal de L positiva)
```

**Algoritmo:**
```
Para k = 1 até n:
  lₖₖ = √(aₖₖ - Σlₖⱼ²)  (j < k)
  lᵢₖ = (aᵢₖ - Σlᵢⱼlₖⱼ) / lₖₖ  (i > k, j < k)
```

**Aplicações:**
- Mínimos quadrados
- Estatística (matrizes de covariância)
- Otimização
- Processos gaussianos

---

## **6. 🚀 Aplicações**

### **6.1 Regressão Linear**

**Problema:** Minimizar ||Ax - b||²

**Equações Normais:**
```
AᵀAx = Aᵀb

Resolver com:
• Decomposição de Cholesky (se AᵀA é SPD)
• Decomposição QR (mais estável)
• SVD (mais robusto)
```

### **6.2 Redes Elétricas**

**Lei de Kirchhoff:** Soma de correntes em nó = 0

**Sistema:**
```
Gv = i

onde:
• G: matriz de condutância
• v: tensões nos nós
• i: correntes injetadas
```

**Características:**
- G é esparsa (conexões locais)
- G é simétrica
- G é positiva definida

### **6.3 Diferenças Finitas (EDPs)**

**Equação de Laplace:** ∇²u = 0

**Discretização:**
```
(uᵢ₊₁,ⱼ - 2uᵢ,ⱼ + uᵢ₋₁,ⱼ)/h² + 
(uᵢ,ⱼ₊₁ - 2uᵢ,ⱼ + uᵢ,ⱼ₋₁)/h² = 0
```

**Sistema Linear:**
```
Au = f

onde A é esparsa e banda
```

### **6.4 PageRank (Google)**

**Problema:** Calcular importância de páginas web.

**Sistema:**
```
(I - αP)x = (1-α)e/n

onde:
• P: matriz de transição (esparsa!)
• α: damping factor (≈ 0.85)
• x: vetor de PageRank
```

**Solução:** Método iterativo (Power Method)

### **6.5 Machine Learning**

**Treinamento de Redes Neurais:**
```
Hessiano Δw = -∇L

onde H é matriz Hessiana
Resolver para Δw (atualização de pesos)
```

**Métodos:**
- Gradiente Conjugado
- L-BFGS (quasi-Newton)
- Adam (gradiente adaptativo)

---

## **7. 🧮 Exercícios Resolvidos**

### **Exercício 1: Eliminação de Gauss**
**Sistema:**
```
x + y = 3
2x + 3y = 8
```

**Solução:**
```
[1  1 | 3]
[2  3 | 8]

Eliminar x da L2:
[1  1 | 3]
[0  1 | 2]  (L2 - 2×L1)

Substituição:
y = 2
x = 3 - 2 = 1

Solução: x = 1, y = 2
```

### **Exercício 2: Jacobi**
**Sistema:**
```
4x + y = 15
x + 3y = 14
```

**Solução:**
```
x⁽⁰⁾ = [0, 0]

Iteração 1:
x⁽¹⁾ = (15 - 0)/4 = 3.75
y⁽¹⁾ = (14 - 0)/3 = 4.67

Iteração 2:
x⁽²⁾ = (15 - 4.67)/4 = 2.58
y⁽²⁾ = (14 - 3.75)/3 = 3.42

...continua até convergir para x = 3, y = 4
```

### **Exercício 3: Condicionamento**
**Calcular κ(A) para:**
```
A = [1  2]
    [2  4.001]

Valores singulares:
σ₁ ≈ 4.5
σ₂ ≈ 0.001

κ(A) = σ₁/σ₂ ≈ 4500

Mal-condicionado! Pequenos erros se amplificam.
```

---

## **8. 💻 Implementação em Python**

```python
import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, cg
import matplotlib.pyplot as plt

# Sistema de exemplo
A = np.array([[4, 1, 0],
              [1, 3, 1],
              [0, 1, 2]], dtype=float)
b = np.array([15, 14, 8], dtype=float)

print("Sistema Ax = b:")
print(f"A =\n{A}")
print(f"b = {b}")

# 1. Eliminação de Gauss (direto)
x_gauss = linalg.solve(A, b)
print(f"\n1. Gauss: x = {x_gauss}")

# 2. Decomposição LU
P, L, U = linalg.lu(A)
print(f"\n2. LU Decomposition:")
print(f"L =\n{L}")
print(f"U =\n{U}")
y = linalg.solve(L, P @ b)
x_lu = linalg.solve(U, y)
print(f"Solução: x = {x_lu}")

# 3. Decomposição QR
Q, R = linalg.qr(A)
x_qr = linalg.solve(R, Q.T @ b)
print(f"\n3. QR: x = {x_qr}")

# 4. Decomposição SVD
U_svd, s, Vt = linalg.svd(A)
x_svd = Vt.T @ np.diag(1/s) @ U_svd.T @ b
print(f"\n4. SVD: x = {x_svd}")

# 5. Cholesky (A é SPD)
L_chol = linalg.cholesky(A, lower=True)
y_chol = linalg.solve(L_chol, b)
x_chol = linalg.solve(L_chol.T, y_chol)
print(f"\n5. Cholesky: x = {x_chol}")

# 6. Método de Jacobi
def jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    
    for i in range(max_iter):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    return x, max_iter

x_jacobi, iters = jacobi(A, b)
print(f"\n6. Jacobi: x = {x_jacobi} (iterações: {iters})")

# 7. Método de Gauss-Seidel
def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - A[i,:i] @ x[:i] - A[i,i+1:] @ x_old[i+1:]) / A[i,i]
        if np.linalg.norm(x - x_old) < tol:
            return x, k+1
        x = x
    return x, max_iter

x_gs, iters_gs = gauss_seidel(A, b)
print(f"\n7. Gauss-Seidel: x = {x_gs} (iterações: {iters_gs})")

# 8. Gradiente Conjugado
x_cg, info = cg(A, b, tol=1e-6)
print(f"\n8. Gradiente Conjugado: x = {x_cg}")

# 9. Análise de Condicionamento
cond = np.linalg.cond(A)
print(f"\n9. Número de Condição: κ(A) = {cond:.2f}")

# 10. Verificação
residual = b - A @ x_gauss
print(f"\n10. Resíduo: ||Ax - b|| = {np.linalg.norm(residual):.2e}")

# 11. Matriz Esparsa
A_sparse = csr_matrix(A)
x_sparse = spsolve(A_sparse, b)
print(f"\n11. Esparsa: x = {x_sparse}")

# 12. Visualização da Convergência
def convergence_plot():
    x_true = linalg.solve(A, b)
    errors_jacobi = []
    errors_gs = []
    
    x_j = np.zeros(len(b))
    x_g = np.zeros(len(b))
    
    for i in range(30):
        # Jacobi
        D = np.diag(A)
        R = A - np.diagflat(D)
        x_j = (b - R @ x_j) / D
        errors_jacobi.append(np.linalg.norm(x_j - x_true))
        
        # Gauss-Seidel
        for j in range(len(b)):
            x_g[j] = (b[j] - A[j,:j] @ x_g[:j] - A[j,j+1:] @ x_g[j+1:]) / A[j,j]
        errors_gs.append(np.linalg.norm(x_g - x_true))
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(errors_jacobi, 'o-', label='Jacobi')
    plt.semilogy(errors_gs, 's-', label='Gauss-Seidel')
    plt.xlabel('Iteração')
    plt.ylabel('Erro ||x - x_true||')
    plt.title('Convergência dos Métodos Iterativos')
    plt.legend()
    plt.grid(True)
    plt.show()

convergence_plot()
```

---

## **9. 🔗 Recursos Adicionais**

### **Livros Recomendados**
- **Numerical Linear Algebra** - Trefethen & Bau
- **Matrix Computations** - Golub & Van Loan
- **Numerical Analysis** - Burden & Faires
- **Applied Numerical Linear Algebra** - Demmel

### **Cursos Online**
- MIT 18.06 - Linear Algebra (Gilbert Strang)
- Stanford CS 205A - Mathematical Methods
- Coursera - Numerical Methods

### **Bibliotecas Python**
```python
# Básico
import numpy as np
import scipy.linalg

# Esparso
from scipy.sparse import csr_matrix, linalg as sparse_linalg
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab

# Especializadas
import pyamg  # Multigrid algébrico
import petsc4py  # PETSc (High Performance)
```

### **Ferramentas**
- [Matrix Calculator](https://matrixcalc.org/)
- [Wolfram Alpha](https://www.wolframalpha.com/)
- [Octave/MATLAB](https://www.gnu.org/software/octave/)

---

**Voltar para:** [Estatística](../README.md) | [Notebooks](../../README.md)

# Probabilidade Básica 🎲

A **teoria de probabilidade** é o ramo da matemática que estuda fenômenos aleatórios e quantifica a incerteza. É fundamental para estatística, ciência de dados, aprendizado de máquina e inteligência artificial.

---

## **1. 🎯 Fundamentos Teóricos**

### **1.1 Conceitos Fundamentais**

#### **Experimento Aleatório**
Um **experimento aleatório** é um processo que:
- Pode ser repetido sob as mesmas condições
- Tem resultados possíveis bem definidos
- O resultado específico não pode ser previsto com certeza

**Exemplos:**
- 🎲 Lançar um dado
- 🪙 Jogar uma moeda
- 🎴 Tirar uma carta de um baralho
- 🌡️ Medir a temperatura em um dia aleatório

#### **Espaço Amostral (Ω)**
Conjunto de **todos os resultados possíveis** de um experimento aleatório.

**Exemplos:**
```
Lançamento de moeda:    Ω = {Cara, Coroa}
Lançamento de dado:     Ω = {1, 2, 3, 4, 5, 6}
Soma de dois dados:     Ω = {2, 3, 4, ..., 12}
```

#### **Evento (E)**
Um **evento** é qualquer subconjunto do espaço amostral.

**Exemplos:**
```
E₁ = "Obter número par no dado"     = {2, 4, 6}
E₂ = "Obter número maior que 4"    = {5, 6}
E₃ = "Obter cara na moeda"         = {Cara}
```

**Tipos de Eventos:**
- **Evento Simples:** Contém apenas um resultado (ex: {3})
- **Evento Composto:** Contém múltiplos resultados (ex: {2, 4, 6})
- **Evento Certo:** É o próprio espaço amostral (Ω)
- **Evento Impossível:** É o conjunto vazio (∅)

### **1.2 Definição de Probabilidade**

A probabilidade de um evento E, denotada por P(E), é um número que satisfaz:

**Axiomas de Kolmogorov:**
1. **Não-negatividade:** P(E) ≥ 0 para todo evento E
2. **Normalização:** P(Ω) = 1
3. **Aditividade:** Se E₁ e E₂ são mutuamente exclusivos, então:
   ```
   P(E₁ ∪ E₂) = P(E₁) + P(E₂)
   ```

**Propriedades Derivadas:**
```
• P(∅) = 0
• P(Eᶜ) = 1 - P(E)  (probabilidade do complementar)
• 0 ≤ P(E) ≤ 1
• Se E₁ ⊆ E₂, então P(E₁) ≤ P(E₂)
```

---

## **2. 📊 Abordagens para Calcular Probabilidade**

### **2.1 Probabilidade Clássica (A Priori)**

Usada quando todos os resultados são **igualmente prováveis**.

**Fórmula:**
```
P(E) = Número de resultados favoráveis a E
       ─────────────────────────────────────
       Número total de resultados possíveis

P(E) = |E|
       ───
       |Ω|
```

**Exemplo 1: Lançamento de Dado**
```
P("obter 4") = 1/6 ≈ 0.1667 = 16.67%

P("obter número par") = P({2,4,6}) = 3/6 = 1/2 = 50%

P("obter número ≤ 4") = P({1,2,3,4}) = 4/6 = 2/3 ≈ 66.67%
```

**Exemplo 2: Baralho de 52 Cartas**
```
P("tirar um Ás") = 4/52 = 1/13 ≈ 7.69%

P("tirar uma carta de copas") = 13/52 = 1/4 = 25%

P("tirar uma figura") = 12/52 = 3/13 ≈ 23.08%
```

### **2.2 Probabilidade Frequentista (Empírica)**

Baseia-se na **frequência relativa** observada em experimentos repetidos.

**Fórmula:**
```
P(E) = lim   Número de vezes que E ocorreu
       n→∞   ─────────────────────────────
             Número total de experimentos

P(E) ≈ frequência relativa = nₑ/n
```

**Lei dos Grandes Números:**
> À medida que o número de experimentos aumenta, a frequência relativa converge para a probabilidade verdadeira.

**Exemplo: Lançamento de Moeda**
```python
# Simulação de lançamentos de moeda
n = 10:        P(Cara) ≈ 0.60  (60%)
n = 100:       P(Cara) ≈ 0.52  (52%)
n = 1,000:     P(Cara) ≈ 0.505 (50.5%)
n = 1,000,000: P(Cara) ≈ 0.500001 (≈50%)
```

**Aplicações:**
- Controle de qualidade industrial
- Análise de dados históricos
- Testes A/B em marketing
- Simulações Monte Carlo

### **2.3 Probabilidade Subjetiva (Bayesiana)**

Representa o **grau de crença** pessoal sobre a ocorrência de um evento.

**Características:**
- Baseada em conhecimento prévio
- Pode ser atualizada com novas evidências
- Varia entre diferentes observadores

**Exemplo:**
```
"Qual a probabilidade de chover amanhã?"
- Meteorologista: 70% (baseado em modelos)
- Leigo: 30% (baseado em observação do céu)
```

---

## **3. 🔧 Operações com Eventos**

### **3.1 União de Eventos (E₁ ∪ E₂)**

Evento que ocorre quando **pelo menos um** dos eventos ocorre.

**Regra da Adição:**
```
P(E₁ ∪ E₂) = P(E₁) + P(E₂) - P(E₁ ∩ E₂)
```

**Caso Especial (eventos mutuamente exclusivos):**
```
Se E₁ ∩ E₂ = ∅, então:
P(E₁ ∪ E₂) = P(E₁) + P(E₂)
```

**Exemplo:**
```
Dado de 6 faces:
E₁ = "número par" = {2, 4, 6}
E₂ = "número ≤ 3" = {1, 2, 3}

E₁ ∪ E₂ = {1, 2, 3, 4, 6}
P(E₁ ∪ E₂) = P(E₁) + P(E₂) - P(E₁ ∩ E₂)
           = 3/6 + 3/6 - 1/6
           = 5/6 ≈ 83.33%
```

### **3.2 Interseção de Eventos (E₁ ∩ E₂)**

Evento que ocorre quando **ambos** os eventos ocorrem simultaneamente.

**Exemplo:**
```
E₁ ∩ E₂ = {2}  (número que é par E menor ou igual a 3)
P(E₁ ∩ E₂) = 1/6 ≈ 16.67%
```

### **3.3 Complemento de Evento (Eᶜ)**

Evento que ocorre quando E **não ocorre**.

**Fórmula:**
```
P(Eᶜ) = 1 - P(E)
```

**Exemplo:**
```
E = "obter número par"
Eᶜ = "obter número ímpar"
P(Eᶜ) = 1 - 3/6 = 1/2 = 50%
```

---

## **4. 🎲 Probabilidade Condicional e Independência**

### **4.1 Probabilidade Condicional**

Probabilidade de um evento **dado que outro já ocorreu**.

**Definição:**
```
P(A|B) = P(A ∩ B)
         ─────────
           P(B)

Lê-se: "Probabilidade de A dado B"
```

**Exemplo:**
```
Dois lançamentos de dado:
A = "soma é 8"
B = "primeiro dado é 3"

P(A|B) = P(soma é 8 | primeiro é 3)
       = P({3,5}) / P(primeiro é 3)
       = (1/36) / (1/6)
       = 1/6 ≈ 16.67%
```

### **4.2 Teorema de Bayes**

Fundamental para **inferência estatística** e **aprendizado de máquina**.

**Fórmula:**
```
P(A|B) = P(B|A) × P(A)
         ─────────────
             P(B)

Onde:
• P(A|B): Probabilidade a posteriori
• P(B|A): Verossimilhança
• P(A):   Probabilidade a priori
• P(B):   Evidência (normalização)
```

**Forma Expandida:**
```
P(A|B) = P(B|A) × P(A)
         ──────────────────────────────────
         P(B|A)×P(A) + P(B|Aᶜ)×P(Aᶜ)
```

**Exemplo: Teste Médico**
```
D = "pessoa tem doença"
+ = "teste positivo"

Dados:
P(D) = 0.01          (1% da população tem a doença)
P(+|D) = 0.95        (sensibilidade: 95%)
P(+|Dᶜ) = 0.05       (taxa de falso positivo: 5%)

Pergunta: Se o teste é positivo, qual a probabilidade de ter a doença?

P(D|+) = P(+|D) × P(D)
         ────────────────────────────────────
         P(+|D)×P(D) + P(+|Dᶜ)×P(Dᶜ)

       = 0.95 × 0.01
         ─────────────────────────────────
         0.95×0.01 + 0.05×0.99

       = 0.0095
         ────────
         0.0590

       ≈ 0.161 = 16.1%
```

**Interpretação:** Mesmo com teste positivo, a probabilidade de ter a doença é apenas 16.1%! Isso ocorre porque a doença é rara.

### **4.3 Independência de Eventos**

Dois eventos são **independentes** se a ocorrência de um não afeta a probabilidade do outro.

**Definição Matemática:**
```
A e B são independentes se e somente se:

P(A ∩ B) = P(A) × P(B)

Equivalentemente:
P(A|B) = P(A)
P(B|A) = P(B)
```

**Exemplo de Eventos Independentes:**
```
Lançamento de dois dados:
A = "primeiro dado é 4"
B = "segundo dado é 5"

P(A) = 1/6
P(B) = 1/6
P(A ∩ B) = 1/36 = (1/6) × (1/6) ✓
```

**Exemplo de Eventos Dependentes:**
```
Tirar duas cartas sem reposição:
A = "primeira é Ás"
B = "segunda é Ás"

P(A) = 4/52
P(B|A) = 3/51 ≠ 4/52
Portanto, A e B são dependentes
```

---

## **5. 📈 Distribuições de Probabilidade Discretas**

### **5.1 Distribuição Uniforme Discreta**

Todos os resultados têm a **mesma probabilidade**.

**Função de Probabilidade:**
```
P(X = xᵢ) = 1/n

onde n é o número de valores possíveis
```

**Exemplo:**
- Lançamento de dado justo
- Escolha aleatória de um número de loteria

**Propriedades:**
```
Média: μ = (a + b)/2
Variância: σ² = (n² - 1)/12
```

### **5.2 Distribuição Binomial**

Número de **sucessos** em n **tentativas independentes** com probabilidade p.

**Função de Probabilidade:**
```
P(X = k) = C(n,k) × pᵏ × (1-p)ⁿ⁻ᵏ

onde:
• n = número de tentativas
• k = número de sucessos
• p = probabilidade de sucesso em cada tentativa
• C(n,k) = n! / (k!(n-k)!)  (combinação)
```

**Notação:** X ~ Binomial(n, p)

**Exemplo:**
```
Lançar moeda 10 vezes, qual a probabilidade de 7 caras?

n = 10, k = 7, p = 0.5

P(X = 7) = C(10,7) × 0.5⁷ × 0.5³
         = 120 × 0.5¹⁰
         ≈ 0.117 = 11.7%
```

**Propriedades:**
```
Média: μ = n × p
Variância: σ² = n × p × (1-p)
Desvio Padrão: σ = √(n × p × (1-p))
```

**Aplicações:**
- Controle de qualidade (itens defeituosos)
- Testes A/B (conversões)
- Pesquisas de opinião (respostas sim/não)

### **5.3 Distribuição de Poisson**

Número de **eventos raros** em um intervalo fixo de tempo ou espaço.

**Função de Probabilidade:**
```
P(X = k) = (λᵏ × e⁻λ) / k!

onde:
• λ = taxa média de ocorrências
• k = número de ocorrências
• e ≈ 2.71828
```

**Notação:** X ~ Poisson(λ)

**Exemplo:**
```
Média de 3 chamadas por hora (λ = 3)
Probabilidade de 5 chamadas em uma hora?

P(X = 5) = (3⁵ × e⁻³) / 5!
         = (243 × 0.0498) / 120
         ≈ 0.101 = 10.1%
```

**Propriedades:**
```
Média: μ = λ
Variância: σ² = λ
Desvio Padrão: σ = √λ
```

**Aplicações:**
- Número de chamadas em call center
- Chegadas de clientes em fila
- Erros de digitação por página
- Acidentes de trânsito por dia

### **5.4 Distribuição Geométrica**

Número de **tentativas até o primeiro sucesso**.

**Função de Probabilidade:**
```
P(X = k) = (1-p)ᵏ⁻¹ × p

onde:
• p = probabilidade de sucesso
• k = número de tentativas até sucesso (k ≥ 1)
```

**Notação:** X ~ Geométrica(p)

**Exemplo:**
```
Probabilidade de acertar = 0.2
Quantas tentativas até acertar?

P(X = 1) = 0.2 = 20%           (acerta na primeira)
P(X = 2) = 0.8 × 0.2 = 16%     (acerta na segunda)
P(X = 3) = 0.8² × 0.2 = 12.8%  (acerta na terceira)
```

**Propriedades:**
```
Média: μ = 1/p
Variância: σ² = (1-p)/p²
```

**Propriedade da Falta de Memória:**
```
P(X > n + k | X > n) = P(X > k)
```

---

## **6. 🎮 Aplicações Práticas**

### **6.1 Jogos de Azar**

**Probabilidade em Loteria:**
```
Mega-Sena (6 números de 60):
P(ganhar) = 1 / C(60,6)
          = 1 / 50,063,860
          ≈ 0.00000002 = 0.000002%
```

**Probabilidade no Poker:**
```
Royal Flush:
P = 4 / C(52,5)
  = 4 / 2,598,960
  ≈ 0.000154%
```

### **6.2 Simulação Monte Carlo**

Técnica que usa **amostragem aleatória** para resolver problemas numéricos.

**Aplicações:**
- Precificação de opções financeiras
- Análise de risco
- Física computacional
- Otimização estocástica

**Exemplo: Estimando π**
```python
import random

def estimar_pi(n_pontos):
    dentro_circulo = 0
    for _ in range(n_pontos):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            dentro_circulo += 1
    
    pi_estimado = 4 * dentro_circulo / n_pontos
    return pi_estimado

# Com 1 milhão de pontos
pi_approx = estimar_pi(1_000_000)
# Resultado ≈ 3.141...
```

### **6.3 Testes de Hipóteses**

Probabilidade é fundamental em **inferência estatística**.

**Valor-p (p-value):**
```
Probabilidade de observar dados tão extremos quanto os observados,
assumindo que a hipótese nula é verdadeira.

Se p-value < α (nível de significância), rejeita H₀
```

**Exemplo:**
```
H₀: Moeda é justa (p = 0.5)
Observamos: 65 caras em 100 lançamentos

p-value ≈ 0.003
Se α = 0.05, rejeitamos H₀
Conclusão: Evidência de que a moeda não é justa
```

### **6.4 Machine Learning**

**Classificadores Probabilísticos:**
- **Naive Bayes:** Usa teorema de Bayes
- **Regressão Logística:** Modela P(Y=1|X)
- **Redes Bayesianas:** Grafos de dependências probabilísticas

**Exemplo: Filtro de Spam**
```
P(Spam | palavras) = P(palavras | Spam) × P(Spam)
                     ──────────────────────────────
                            P(palavras)
```

---

## **7. 📚 Conceitos Avançados**

### **7.1 Esperança (Valor Esperado)**

Média ponderada de todos os valores possíveis.

**Definição:**
```
E[X] = Σ xᵢ × P(X = xᵢ)
```

**Propriedades:**
```
• E[aX + b] = a×E[X] + b
• E[X + Y] = E[X] + E[Y]
• Se X e Y independentes: E[X×Y] = E[X]×E[Y]
```

### **7.2 Variância**

Medida de **dispersão** da distribuição.

**Definição:**
```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
```

**Propriedades:**
```
• Var(aX + b) = a² × Var(X)
• Se X e Y independentes: Var(X+Y) = Var(X) + Var(Y)
```

### **7.3 Covariância e Correlação**

**Covariância:**
```
Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] = E[XY] - E[X]E[Y]
```

**Coeficiente de Correlação:**
```
ρ(X,Y) = Cov(X,Y) / (σₓ × σᵧ)

-1 ≤ ρ ≤ 1
```

---

## **8. 🧮 Exercícios Resolvidos**

### **Exercício 1: Probabilidade Clássica**
**Problema:** Em uma urna há 5 bolas vermelhas, 3 azuis e 2 verdes. Qual a probabilidade de retirar uma bola azul?

**Solução:**
```
Total de bolas = 5 + 3 + 2 = 10
Bolas azuis = 3

P(azul) = 3/10 = 0.3 = 30%
```

### **Exercício 2: Probabilidade Condicional**
**Problema:** Em uma escola, 60% dos alunos jogam futebol, 40% jogam basquete, e 25% jogam ambos. Qual a probabilidade de um aluno jogar basquete dado que joga futebol?

**Solução:**
```
F = "joga futebol"
B = "joga basquete"

P(F) = 0.60
P(B) = 0.40
P(F ∩ B) = 0.25

P(B|F) = P(F ∩ B) / P(F)
       = 0.25 / 0.60
       ≈ 0.417 = 41.7%
```

### **Exercício 3: Teorema de Bayes**
**Problema:** Uma fábrica tem 3 máquinas. Máquina A produz 50% das peças (2% defeituosas), B produz 30% (3% defeituosas), C produz 20% (5% defeituosas). Uma peça é selecionada e está defeituosa. Qual a probabilidade de ser da máquina A?

**Solução:**
```
P(A) = 0.50,  P(D|A) = 0.02
P(B) = 0.30,  P(D|B) = 0.03
P(C) = 0.20,  P(D|C) = 0.05

P(D) = P(D|A)P(A) + P(D|B)P(B) + P(D|C)P(C)
     = 0.02×0.50 + 0.03×0.30 + 0.05×0.20
     = 0.010 + 0.009 + 0.010
     = 0.029

P(A|D) = P(D|A)×P(A) / P(D)
       = 0.02×0.50 / 0.029
       = 0.010 / 0.029
       ≈ 0.345 = 34.5%
```

---

## **9. 🔗 Recursos Adicionais**

### **Livros Recomendados**
- **Introduction to Probability** - Bertsekas & Tsitsiklis (MIT)
- **Probabilidade: Aplicações à Estatística** - Paul Meyer
- **A First Course in Probability** - Sheldon Ross
- **Probabilidade e Estatística** - Magalhães & Lima

### **Ferramentas Online**
- [Wolfram Alpha](https://www.wolframalpha.com/) - Cálculos de probabilidade
- [Seeing Theory](https://seeing-theory.brown.edu/) - Visualizações interativas
- [Khan Academy](https://www.khanacademy.org/) - Cursos gratuitos

### **Bibliotecas Python**
```python
import random          # Geração de números aleatórios
import numpy as np     # Operações numéricas
from scipy import stats # Distribuições de probabilidade
import matplotlib.pyplot as plt  # Visualizações
```

---

**Voltar para:** [Estatística](../README.md) | [Notebooks](../../README.md)

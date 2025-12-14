# Análise de Confiabilidade e Eventos Raros

Esta seção cobre métodos avançados para análise de confiabilidade estrutural, estimação de probabilidades de eventos raros e quantificação de incerteza em sistemas complexos.

## 📚 Métodos Disponíveis

### [BUS com Subset Simulation (SUS)](bus_subset_simulation.md)

BUS (Bayesian Updating with Structural reliability methods) combinado com Subset Simulation é um método avançado para estimação de probabilidades de eventos raros e atualização bayesiana de modelos.

**Principais Características:**
- ⚡ Eficiência computacional para eventos raros (Pf ~ 10⁻⁶)
- 🎯 Decomposição em eventos intermediários
- 🔄 MCMC condicional para amostragem
- 🧮 Integração com inferência bayesiana

**Quando Usar:**
- Análise de confiabilidade estrutural
- Probabilidades muito pequenas (Pf < 10⁻³)
- Função de desempenho cara de avaliar
- Atualização bayesiana com dados limitados
- Necessidade de amostras da região de falha

**Comparação com Outros Métodos:**

| Método | Eficiência | Aplicabilidade | Complexidade |
|--------|-----------|----------------|--------------|
| **Monte Carlo Simples** | ❌ Baixa | ✅ Universal | ✅ Simples |
| **FORM/SORM** | ✅ Alta | ⚠️ Limitada | ⚠️ Média |
| **Importance Sampling** | ⚠️ Variável | ✅ Boa | ⚠️ Média |
| **Subset Simulation** | ✅✅ Muito Alta | ✅ Universal | ⚠️ Média |
| **Cross-Entropy Method** | ✅ Alta | ✅ Boa | ✅ Simples |

## 🎯 Conceitos Fundamentais

### **Função de Desempenho (Performance Function)**
```
g(X) = [expressão das variáveis aleatórias]

g(X) > 0  → Estado seguro
g(X) ≤ 0  → Estado de falha

Pf = P(g(X) ≤ 0)
```

### **Problema de Eventos Raros**
Para Pf = 10⁻⁶ com Monte Carlo simples:
- Necessário: ~10⁹ amostras
- Tempo: Impraticável para funções caras

### **Solução: Subset Simulation**
Decomposição em níveis intermediários:
```
F = F₁ ⊃ F₂ ⊃ ... ⊃ Fₘ

P(F) = P(F₁) × ∏ P(Fᵢ₊₁|Fᵢ)
       ≈ p₀ᵐ  (para m níveis)
```

## 📊 Aplicações Práticas

### 🏗️ **Engenharia Estrutural**
- Análise de confiabilidade de edifícios
- Avaliação de riscos sísmicos
- Fadiga e fratura de materiais
- Segurança de pontes e barragens

### ⚡ **Engenharia Nuclear**
- Análise probabilística de segurança (PSA)
- Estimação de risco de acidentes
- Confiabilidade de sistemas de segurança

### ✈️ **Engenharia Aeroespacial**
- Confiabilidade de sistemas de voo
- Análise de missão
- Tolerância a falhas

### 🏭 **Sistemas Industriais**
- Confiabilidade de processos
- Análise de risco operacional
- Manutenção preditiva

## 🔧 Ferramentas e Implementações

### **Algoritmos Principais**
1. **Subset Simulation:** Estimação de Pf
2. **BUS (Bayesian Updating):** Atualização de modelos
3. **Modified Metropolis-Hastings:** MCMC condicional

### **Componentes Chave**
- Amostragem adaptativa
- Cadeias de Markov condicionais
- Decomposição em níveis
- Estimação de evidência bayesiana

## 📖 Recursos Adicionais

### **Literatura Fundamental**
1. **Au & Beck (2001):** "Estimation of small failure probabilities in high dimensions by subset simulation"
2. **Straub & Papaioannou (2015):** "Bayesian updating with structural reliability methods"
3. **Papaioannou et al. (2015):** "MCMC algorithms for subset simulation"

### **Software e Bibliotecas**
```python
# Bibliotecas Python

# UQpy: Uncertainty Quantification with Python
from UQpy import SubsetSimulation

# OpenCOSSAN: Computational platform for reliability
import opencossan

# PyRe: Python Reliability
from pyre import subset_simulation
```

### **Links Úteis**
- **ERA Group (TU München):** Software e tutoriais
- **UQpy:** https://github.com/SURGroup/UQpy
- **Tutorial:** https://arxiv.org/abs/1505.03506

## 🎓 Tópicos Avançados

### **Extensões e Variantes**
- Adaptive Subset Simulation
- Subset Simulation com modelos substitutos
- Multi-level Subset Simulation
- Parallelização de SUS

### **Integração com Outras Técnicas**
- SUS + Metamodelos (GP, Kriging)
- SUS + Otimização
- SUS + Análise de sensibilidade
- SUS + Machine Learning

## 🔗 Métodos Relacionados

- **FORM/SORM:** First/Second Order Reliability Methods
- **Importance Sampling:** Amostragem por importância
- **Line Sampling:** Amostragem linear
- **Cross-Entropy Method:** [Ver documentação](../optimization/cross_entropy_method.md)

---

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

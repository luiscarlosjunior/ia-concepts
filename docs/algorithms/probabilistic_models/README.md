# Modelos Probabilísticos e Gráficos

Esta seção aborda modelos probabilísticos gráficos, incluindo redes bayesianas, cadeias de Markov e modelos temporais.

## 📚 Modelos Disponíveis

### [Redes Bayesianas Dinâmicas (Dynamic Bayesian Networks - DBN)](dynamic_bayesian_networks.md)

Redes Bayesianas Dinâmicas são modelos probabilísticos gráficos que representam processos estocásticos temporais, estendendo Redes Bayesianas para modelar sistemas que evoluem no tempo.

**Principais Características:**
- 🔄 Modelagem de processos temporais
- 📊 Representação gráfica de dependências causais
- 🎯 Inferência rigorosa sob incerteza
- 🧩 Estrutura modular e interpretável

**Quando Usar:**
- Dados sequenciais/temporais
- Necessidade de quantificar incerteza temporal
- Estrutura causal conhecida ou a descobrir
- Integração de múltiplas fontes de informação

**Modelos Especiais (Casos de DBN):**
- **Hidden Markov Models (HMM):** Estados discretos
- **Kalman Filter:** Estados contínuos, dinâmica linear
- **Particle Filter:** Modelos não-lineares gerais
- **Conditional Random Fields (CRF):** Modelos discriminativos

## 📊 Tarefas de Inferência

### 1. **Filtragem (Filtering)**
Estimar estado atual dado observações passadas:
```
P(Xₜ | Y₁, ..., Yₜ)
```
**Aplicação:** Rastreamento em tempo real

### 2. **Predição (Prediction)**
Prever estados futuros:
```
P(Xₜ₊ₖ | Y₁, ..., Yₜ)
```
**Aplicação:** Previsão de séries temporais

### 3. **Suavização (Smoothing)**
Estimar estados passados com todas observações:
```
P(Xₜ | Y₁, ..., Yₜ)  para t ≤ T
```
**Aplicação:** Análise retrospectiva

### 4. **Viterbi (Most Likely Path)**
Encontrar sequência de estados mais provável:
```
argmax P(X₁, ..., Xₜ | Y₁, ..., Yₜ)
```
**Aplicação:** Reconhecimento de padrões

## 🎯 Aplicações Práticas

### 🗣️ **Reconhecimento de Fala**
- HMMs para modelar fonemas
- Transições entre estados fonéticos
- Observações: características acústicas (MFCCs)

### 🧬 **Bioinformática**
- Predição de estrutura de proteínas
- Modelagem de sequências genéticas
- Alinhamento de sequências

### 💰 **Finanças**
- Detecção de regimes de mercado
- Modelagem de volatilidade
- Previsão de séries financeiras

### 🤖 **Robótica**
- Localização e mapeamento (SLAM)
- Rastreamento de objetos
- Navegação autônoma

## 🔗 Conceitos Relacionados

- **Markov Chains:** Processos sem observações
- **Redes Bayesianas Estáticas:** Sem dimensão temporal
- **Processos Gaussianos:** Modelos não-paramétricos contínuos
- **Recurrent Neural Networks (RNN):** Abordagem de deep learning

## 📖 Recursos Adicionais

- **Livro de Referência:** *Probabilistic Graphical Models* (Koller & Friedman)
- **Tutorial HMM:** "A Tutorial on Hidden Markov Models" (Rabiner, 1989)
- **Bibliotecas Python:** hmmlearn, pomegranate, pyro, pgmpy

## 🎓 Tópicos Avançados

- Aprendizado de estrutura de redes bayesianas
- Inferência variacional em DBNs
- DBNs com variáveis contínuas e discretas mistas
- Redes bayesianas hierárquicas

---

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

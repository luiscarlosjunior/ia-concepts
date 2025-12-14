# Aprendizado Estatístico e Machine Learning

Esta seção contém documentação sobre métodos de aprendizado de máquina com fundamentação estatística rigorosa, incluindo métodos bayesianos e não-paramétricos.

## 📚 Algoritmos Disponíveis

### [Regressão de Processo Gaussiano (Gaussian Process Regression - GPR)](gaussian_process_regression.md)

Regressão de Processo Gaussiano é um método não-paramétrico de aprendizado de máquina que fornece não apenas predições, mas também quantificação completa de incerteza.

**Principais Características:**
- 📊 Distribuição de probabilidade sobre funções
- 🎯 Quantificação natural de incerteza
- 🧮 Fundamentação bayesiana rigorosa
- 🔧 Flexível através da escolha de kernels

**Quando Usar:**
- Dados pequenos a médios (n < 10,000)
- Necessidade de quantificar incerteza
- Otimização bayesiana
- Modelagem científica interpretável
- Incorporar conhecimento prévio

**Aplicações:**
- Otimização Bayesiana
- Interpolação e suavização de dados
- Análise de sensibilidade
- Modelagem de sistemas complexos
- Calibração de modelos

**Kernels Disponíveis:**
- **RBF (Squared Exponential):** Funções suaves
- **Matérn:** Controle de suavidade
- **Linear:** Relações lineares
- **Periódico:** Padrões repetitivos
- **Composições:** Combinações de kernels

## 🔗 Métodos Relacionados

- **Kriging:** Equivalente em geoestatística
- **Spline Smoothing:** Suavização de dados
- **Support Vector Machines (SVM):** Kernels similares
- **Bayesian Neural Networks:** Alternativa para dados grandes

## 📖 Recursos Adicionais

- **Livro de Referência:** *Gaussian Processes for Machine Learning* (Rasmussen & Williams)
- **Bibliotecas Python:** GPy, scikit-learn, GPflow, GPyTorch
- **Visualizador Interativo:** [GP Playground](https://chi-feng.github.io/gp-demo/)

## 🎯 Tópicos Avançados

- GP Esparsos para escalabilidade
- GP Multi-tarefa
- Deep Gaussian Processes
- GP para classificação
- GP em séries temporais

---

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

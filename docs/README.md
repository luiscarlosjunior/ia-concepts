# Documentação - IA Concepts

Esta pasta contém a documentação detalhada dos algoritmos e conceitos implementados no projeto.

## 📁 Organização

### `/algorithms` - Documentação de Algoritmos
Documentação técnica completa dos algoritmos implementados, organizados por categoria:

#### `/greedy` - Algoritmos Gulosos
- **[Hill Climbing](algorithms/greedy/hill_climbing.md)**: Algoritmo de busca local para otimização

#### `/metaheuristics` - Metaheurísticas 
- **[Simulated Annealing](algorithms/metaheuristics/simulated_annealing.md)**: Algoritmo inspirado no processo de recozimento
- **[Tabu Search](algorithms/metaheuristics/tabu_search.md)**: Algoritmo com memória para evitar ciclos

#### `/optimization` - [Algoritmos de Otimização](algorithms/optimization/)
- **[Cross-Entropy Method (CE)](algorithms/optimization/cross_entropy_method.md)**: Método de otimização estocástica baseado em amostragem adaptativa e teoria da informação

#### `/statistical_learning` - [Aprendizado Estatístico](algorithms/statistical_learning/)
- **[Gaussian Process Regression (GPR)](algorithms/statistical_learning/gaussian_process_regression.md)**: Regressão não-paramétrica com quantificação de incerteza

#### `/probabilistic_models` - [Modelos Probabilísticos](algorithms/probabilistic_models/)
- **[Dynamic Bayesian Networks (DBN)](algorithms/probabilistic_models/dynamic_bayesian_networks.md)**: Modelos gráficos probabilísticos para processos temporais

#### `/reliability_analysis` - [Análise de Confiabilidade](algorithms/reliability_analysis/)
- **[BUS com Subset Simulation (SUS)](algorithms/reliability_analysis/bus_subset_simulation.md)**: Método para estimação de eventos raros e atualização bayesiana

### `/images` - Recursos Visuais
Diagramas, fluxogramas e imagens utilizados na documentação.

## 📖 Como Usar a Documentação

1. **Leitura Sequencial**: Comece pelos algoritmos mais simples (greedy) e avance para os mais complexos
2. **Referência Rápida**: Use os links diretos para conceitos específicos
3. **Implementação**: Cada documento inclui links para o código fonte correspondente
4. **Exercícios**: Documentos incluem exercícios práticos para aprendizado

## 🎯 Estrutura dos Documentos

Cada documento de algoritmo segue a estrutura:
- **Conceitos Fundamentais**: Teoria e princípios
- **Como Funciona**: Explicação do algoritmo passo a passo
- **Vantagens e Desvantagens**: Análise crítica
- **Implementação**: Detalhes técnicos e código
- **Exemplos Práticos**: Casos de uso reais
- **Exercícios**: Atividades para prática

## 🔗 Links Úteis

- **Notebooks**: [../notebooks/](../notebooks/) - Implementações práticas
- **Código Fonte**: [../src/algorithms/](../src/algorithms/) - Implementações dos algoritmos
- **API**: [../src/api/](../src/api/) - Interface para execução via REST
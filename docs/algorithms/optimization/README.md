# Algoritmos de Otimização

Esta seção contém documentação detalhada sobre algoritmos de otimização, incluindo métodos estocásticos, heurísticos e baseados em amostragem.

## 📚 Algoritmos Disponíveis

### [Método de Entropia Cruzada (Cross-Entropy Method - CE)](cross_entropy_method.md)

O Método de Entropia Cruzada é um algoritmo de otimização estocástica e simulação de eventos raros baseado em amostragem adaptativa.

**Principais Características:**
- 🎯 Otimização baseada em amostragem adaptativa
- 📊 Minimiza divergência de Kullback-Leibler
- 🔄 Aprende distribuição ótima iterativamente
- 🌐 Aplicável a problemas contínuos e discretos

**Quando Usar:**
- Otimização de funções objetivas ruidosas
- Estimação de probabilidades de eventos raros
- Problemas combinatoriais (TSP, agendamento)
- Quando gradientes não estão disponíveis

**Aplicações:**
- Problema do Caixeiro Viajante (TSP)
- Ajuste de hiperparâmetros em ML
- Otimização de redes neurais
- Estimação de eventos raros

## 🔗 Algoritmos Relacionados

- **Simulated Annealing:** [docs/algorithms/metaheuristics/simulated_annealing.md](../metaheuristics/simulated_annealing.md)
- **Algoritmos Genéticos:** Busca baseada em população
- **Particle Swarm Optimization (PSO):** Otimização por enxame

## 📖 Recursos Adicionais

- **Tutorial Interativo:** Implementações práticas em Python
- **Notebooks:** Exemplos de uso em `/notebooks/algorithms/`
- **Código Fonte:** Implementações em `/src/algorithms/optimization/`

## 🎯 Próximos Passos

1. Leia a documentação completa de cada algoritmo
2. Execute os exemplos fornecidos
3. Adapte os algoritmos para seu problema específico
4. Explore variantes e extensões

---

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

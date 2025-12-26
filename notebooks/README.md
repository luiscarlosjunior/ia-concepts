# Notebooks - IA Concepts

Esta pasta contém Jupyter notebooks organizados por categoria temática.

## 📁 Organização

### `/algorithms` - Algoritmos de IA
- **Otimização por Enxame**: 
  - `particle_swarm_optimization.ipynb` - PSO
  - `ant_colony_optimization.ipynb` - Colônia de Formigas
- **Algoritmos Evolutivos**:
  - `algoritmo_genetico.ipynb` - Algoritmos Genéticos
  - `algoritmo_genetico_operadores.ipynb` - Operadores de transição
  - `differential_evolution.ipynb` - Evolução Diferencial
- **Busca**: 
  - `busca_largura_bfs.ipynb` - Busca em Largura (BFS)
  - `busca_profundidade_dfs.ipynb` - Busca em Profundidade (DFS)
  - `buscas_informadas.ipynb` - Buscas heurísticas
- **Clustering**: 
  - `clustering_k_means.ipynb` - Algoritmo K-means
- **Outros**: 
  - `automato_celular_game_of_life.ipynb` - Jogo da Vida

### `/data_science` - Ciência de Dados
- **Análise de Dados**: Pandas, processamento de dados
- **Mineração de Dados**: Análise de sentimentos
- **Aplicações Específicas**: Binding de anticorpos

### `/statistics` - Estatística e Métodos Numéricos
- **Estatística Descritiva**: 
  - `medidas_tendencia_central.ipynb` - Média, mediana, moda
  - `medidas_dispersao.ipynb` - Variância, desvio padrão
- **Probabilidade**: 
  - `probabilidade_basica.ipynb` - Conceitos fundamentais
- **Métodos Numéricos**: 
  - `metodos_numericos_mmq.ipynb` - Mínimos Quadrados
  - `metodos_numericos_sistemas_lineares.ipynb` - Sistemas lineares
  - `ajuste_de_curva.ipynb` - Ajuste e regressão

**Nota**: A documentação teórica foi movida para `/docs/statistics/teoria/`

### `/visualization` - Visualização e Ferramentas
- **Matplotlib**: Introdução e tutoriais
- **SciPy**: Computação científica

## 🚀 Como Usar

```bash
# Instalar dependências
poetry install

# Executar Jupyter Lab
jupyter lab notebooks/

# Ou Jupyter Notebook tradicional
jupyter notebook notebooks/
```

## 📋 Convenções

- Notebooks seguem nomenclatura **snake_case**
- Cada notebook inclui documentação em português
- Células bem comentadas e organizadas
- Exemplos práticos com dados reais quando possível
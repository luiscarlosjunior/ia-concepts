# IA Concepts

Este projeto tem como objetivo explorar e demonstrar conceitos fundamentais de Inteligência Artificial (IA). 
Através de exemplos práticos e teóricos, abordamos diversas áreas da IA, incluindo:

## 📚 Conteúdos

### 1. **Algoritmos de Otimização**
   - **Algoritmos Gulosos (Greedy)**: Hill Climbing, Gradiente Descendente
   - **Metaheurísticas**: Simulated Annealing, Tabu Search, Algoritmos Genéticos
   - **Otimização por Enxame**: PSO (Particle Swarm Optimization), ACO (Ant Colony)

### 2. **Algoritmos de Busca**
   - Busca em Largura (BFS)
   - Busca em Profundidade (DFS)
   - Buscas Informadas (A*, Heurísticas)

### 3. **Aprendizado de Máquina**
   - Algoritmos de Clustering (K-means)
   - Métodos Numéricos e Estatísticos
   - Análise de Dados com Pandas

### 4. **Processamento de Dados**
   - Mineração de Dados
   - Análise de Sentimentos
   - Medidas Estatísticas (Tendência Central, Dispersão)

### 5. **Ferramentas e Bibliotecas**
   - Matplotlib para Visualização
   - SciPy para Computação Científica
   - NumPy para Computação Numérica

## 🗂️ Estrutura do Projeto

- **`/notebooks`**: Jupyter notebooks organizados por categoria
  - `/algorithms`: Algoritmos de IA (otimização, busca, clustering)
  - `/data_science`: Ciência de dados e mineração
  - `/statistics`: Estatística e métodos numéricos  
  - `/visualization`: Visualização e ferramentas
- **`/src`**: Código fonte organizado por algoritmos e funcionalidades
  - `/algorithms/greedy`: Algoritmos gulosos e de gradiente
  - `/algorithms/metaheuristics`: Metaheurísticas (Simulated Annealing, Tabu Search)
  - `/api`: API REST para execução de algoritmos
  - `/services`: Serviços auxiliares
  - `/utils`: Utilitários e funções auxiliares
  - `main.py`: Entry point da API REST
  - `cli.py`: Interface de linha de comando
- **`/docs`**: Documentação detalhada sobre cada conceito e algoritmo
  - `/algorithms`: Documentação de algoritmos organizados por tipo
  - `/statistics`: Teoria estatística e métodos numéricos
- **`/datasets`**: Conjunto de dados utilizados nos exemplos
- **`/scripts`**: Scripts auxiliares para execução e automação
- **`/output`**: Resultados e outputs de execuções

## 🚀 Como Usar

### Executando Notebooks
```bash
# Instalar dependências
poetry install

# Executar Jupyter
jupyter notebook notebooks/
```

### Executando a API
```bash
# Iniciar servidor FastAPI
python src/main.py
```

### Executando a Interface CLI
```bash
# Executar interface de linha de comando
python src/cli.py
```

## 🤝 Como Contribuir

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nome-da-feature`)
3. Implemente suas alterações seguindo as convenções do projeto
4. Adicione testes quando aplicável
5. Faça commit das suas alterações (`git commit -m 'Adiciona nova feature'`)
6. Faça push para a branch (`git push origin feature/nome-da-feature`)
7. Abra um Pull Request com descrição detalhada

### 📋 Convenções do Projeto
- Use **snake_case** para nomes de arquivos
- Documente algoritmos em português no `/docs`
- Mantenha notebooks organizados e bem comentados
- Siga as boas práticas de código Python (PEP 8)

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📧 Contato

Para dúvidas ou sugestões, entre em contato pelo email: luissantos@uni9.pro.br

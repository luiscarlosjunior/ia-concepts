# Estatística e Métodos Numéricos 📊

Esta seção contém notebooks e documentação teórica sobre conceitos fundamentais de estatística, probabilidade e métodos numéricos aplicados à ciência de dados e inteligência artificial.

## 📚 Conteúdos Disponíveis

### 1. [Probabilidade Básica](teoria/probabilidade_basica.md)
Fundamentos de teoria de probabilidade, incluindo:
- **Conceitos Fundamentais:** Espaço amostral, eventos, axiomas da probabilidade
- **Probabilidade Clássica:** Experimentos aleatórios e equiprováveis
- **Probabilidade Frequentista:** Lei dos grandes números
- **Probabilidade Condicional:** Regra de Bayes e independência
- **Distribuições de Probabilidade Discretas:** Binomial, Poisson, Geométrica
- **Aplicações Práticas:** Jogos de azar, simulações Monte Carlo

**Notebook:** [`probabilidade_basica.ipynb`](probabilidade_basica.ipynb)

---

### 2. [Medidas de Tendência Central](teoria/medidas_tendencia_central.md)
Análise das principais medidas que representam o centro de uma distribuição:
- **Média Aritmética:** Propriedades, vantagens e limitações
- **Mediana:** Robustez a outliers e quando utilizá-la
- **Moda:** Identificação de valores mais frequentes
- **Média Ponderada:** Aplicações em diferentes contextos
- **Média Geométrica e Harmônica:** Casos especiais de uso
- **Comparação entre Medidas:** Quando usar cada uma

**Notebook:** [`medidas_tendencia_central.ipynb`](medidas_tendencia_central.ipynb)

---

### 3. [Medidas de Dispersão](teoria/medidas_dispersao.md)
Quantificação da variabilidade e espalhamento dos dados:
- **Amplitude (Range):** Medida mais simples de dispersão
- **Variância:** Quantificação matemática da dispersão
- **Desvio Padrão:** Interpretação na mesma unidade dos dados
- **Coeficiente de Variação:** Comparação de dispersão relativa
- **Quartis e Amplitude Interquartil (IQR):** Medidas robustas
- **Distância de Mahalanobis:** Dispersão multivariada

**Notebook:** [`medidas_de_dispersao.ipynb`](medidas_de_dispersao.ipynb)

---

### 4. [Ajuste de Curva](teoria/ajuste_de_curva.md)
Técnicas para modelar relações entre variáveis:
- **Regressão Linear Simples:** Modelo de duas variáveis
- **Regressão Linear Múltipla:** Múltiplas variáveis preditoras
- **Regressão Polinomial:** Modelagem de relações não-lineares
- **Interpolação vs. Regressão:** Diferenças fundamentais
- **Métodos de Interpolação:** Lagrange, Newton, Splines
- **Avaliação de Modelos:** R², MSE, RMSE, validação

**Notebook:** [`ajuste_de_curva.ipynb`](ajuste_de_curva.ipynb)

---

### 5. [Métodos Numéricos - Mínimos Quadrados (MMQ)](teoria/metodos_numericos_mmq.md)
Método fundamental para ajuste de modelos aos dados:
- **Princípio dos Mínimos Quadrados:** Minimização do erro quadrático
- **Formulação Matricial:** Solução analítica via álgebra linear
- **Mínimos Quadrados Ordinários (OLS):** Hipóteses e propriedades
- **Mínimos Quadrados Ponderados (WLS):** Variâncias heterogêneas
- **Mínimos Quadrados Não-Lineares:** Otimização iterativa
- **Regularização:** Ridge, Lasso e Elastic Net

**Notebook:** [`metodos_numericos_mmq.ipynb`](metodos_numericos_mmq.ipynb)

---

### 6. [Métodos Numéricos - Sistemas Lineares](teoria/metodos_numericos_sistemas_lineares.md)
Técnicas computacionais para resolver sistemas de equações lineares:
- **Métodos Diretos:** Eliminação de Gauss, decomposição LU
- **Decomposição QR:** Estabilidade numérica
- **Métodos Iterativos:** Jacobi, Gauss-Seidel, Gradiente Conjugado
- **Condicionamento de Matrizes:** Número de condição e estabilidade
- **Aplicações:** Regressão linear, redes de circuitos, análise estrutural
- **Complexidade Computacional:** Eficiência dos diferentes métodos

**Notebook:** [`metodos_numericos_sistemas_lineares.ipynb`](metodos_numericos_sistemas_lineares.ipynb)

---

## 🎯 Como Usar Este Material

### Notebooks Práticos
Os notebooks Jupyter contêm implementações práticas, exemplos executáveis e visualizações:
```bash
# Instalar dependências
poetry install

# Executar Jupyter
jupyter notebook
```

### Documentação Teórica
Cada tópico possui documentação teórica detalhada na pasta `teoria/` com:
- Fundamentos matemáticos
- Definições formais
- Exemplos ilustrativos
- Aplicações práticas
- Referências bibliográficas

---

## 🔗 Conceitos Relacionados

### Estatística Inferencial
- Testes de Hipóteses
- Intervalos de Confiança
- Análise de Variância (ANOVA)
- Correlação e Causalidade

### Aprendizado de Máquina
- Regressão Linear e Logística
- Validação Cruzada
- Viés e Variância
- Overfitting e Underfitting

### Otimização
- Gradiente Descendente
- Otimização Convexa
- Algoritmos Genéticos
- Recozimento Simulado

---

## 📖 Recursos Recomendados

### Livros
- **Estatística Básica** - Bussab & Morettin
- **Statistical Learning** - Hastie, Tibshirani & Friedman
- **Numerical Methods** - Press et al. (Numerical Recipes)
- **Pattern Recognition and Machine Learning** - Christopher Bishop

### Cursos Online
- Khan Academy - Estatística e Probabilidade
- Coursera - Estatística para Ciência de Dados
- MIT OpenCourseWare - Probability and Statistics

### Ferramentas Python
- **NumPy:** Computação numérica fundamental
- **SciPy:** Algoritmos científicos avançados
- **Pandas:** Análise e manipulação de dados
- **Matplotlib/Seaborn:** Visualização estatística
- **Scikit-learn:** Aprendizado de máquina

---

## 🤝 Contribuindo

Para adicionar novos conteúdos teóricos ou melhorar os existentes:

1. Mantenha a estrutura consistente com os documentos existentes
2. Inclua fundamentos matemáticos rigorosos
3. Adicione exemplos práticos e aplicações
4. Use visualizações e diagramas quando apropriado
5. Forneça referências bibliográficas

---

**Voltar para:** [Notebooks](../README.md) | [Documentação Principal](../../README.md)

# Algoritmos Gulosos

Os **Algoritmos Gulosos** (Greedy Algorithms) são uma classe de algoritmos que fazem escolhas localmente ótimas em cada etapa, com a esperança de encontrar um ótimo global. Estes algoritmos são fundamentais em ciência da computação e otimização, sendo amplamente utilizados em problemas de busca, otimização combinatória e teoria dos grafos.

![Greedy Algorithms Concept](../../images/greedy_algorithms_concept.png)

## 🎯 Fundamentos dos Algoritmos Gulosos

### **Princípios Básicos**

Os algoritmos gulosos compartilham características fundamentais que os distinguem de outras abordagens:

1. **Escolha Gulosa:** Em cada etapa, faz a escolha que parece melhor no momento
2. **Sem Retrocesso:** Uma vez feita, a escolha não é reconsiderada
3. **Eficiência:** Geralmente possuem complexidade de tempo polinomial
4. **Propriedade de Escolha Gulosa:** A escolha local ótima leva à solução global ótima
5. **Subestrutura Ótima:** Uma solução ótima contém soluções ótimas para subproblemas

### **Estrutura Geral de um Algoritmo Guloso**

```
🌱 1. INICIALIZAÇÃO
   └── Começar com uma solução vazia

🔄 2. ITERAÇÃO (enquanto houver elementos):
   ├── 🔍 SELEÇÃO
   │   └── Escolher o "melhor" elemento disponível
   │
   ├── ✅ VERIFICAÇÃO DE VIABILIDADE
   │   └── Verificar se o elemento pode ser adicionado
   │
   └── ➕ ADIÇÃO
       └── Adicionar elemento à solução parcial

🏆 3. RETORNAR solução construída
```

### **Quando um Algoritmo Guloso Funciona?**

Para que um algoritmo guloso produza a solução ótima, o problema deve ter duas propriedades:

#### **1. Propriedade de Escolha Gulosa**
- A escolha localmente ótima em cada etapa leva à solução globalmente ótima
- Podemos montar uma solução ótima fazendo escolhas localmente ótimas

#### **2. Subestrutura Ótima**
- Uma solução ótima para o problema contém soluções ótimas para subproblemas
- Se removermos uma escolha da solução ótima, o que resta é uma solução ótima para o subproblema correspondente

---

## 📚 Algoritmos Disponíveis

### 1. [**Hill Climbing**](hill_climbing.md)

Algoritmo de busca local que move iterativamente para soluções vizinhas melhores até alcançar um ótimo local.

**Principais Características:**
- 🏔️ Busca local em espaço de soluções
- ⬆️ Sempre move para vizinho melhor
- 🎯 Rápido mas pode ficar preso em ótimos locais
- 🔄 Várias variantes (simple, steepest-ascent, stochastic)

**Quando Usar:**
- Otimização rápida com recursos limitados
- Problemas onde "bom o suficiente" é aceitável
- Prototipagem de soluções
- Inicialização para algoritmos mais sofisticados

**Aplicações:**
- Otimização de funções
- Problema do caixeiro viajante
- Agendamento de tarefas
- Configuração de sistemas

---

### 2. [**Algoritmo de Dijkstra**](dijkstra.md)

Algoritmo para encontrar o caminho mais curto de um vértice fonte para todos os outros vértices em um grafo com pesos não-negativos.

**Principais Características:**
- 🗺️ Encontra caminhos mais curtos em grafos
- ➕ Funciona apenas com pesos não-negativos
- 📊 Usa fila de prioridade para eficiência
- ✅ Garante solução ótima

**Quando Usar:**
- Roteamento em redes
- Sistemas de navegação
- Grafos com pesos não-negativos
- Quando todos os caminhos são necessários

**Aplicações:**
- GPS e sistemas de navegação
- Roteamento de redes de computadores
- Planejamento de rotas de transporte
- Análise de redes sociais

---

### 3. [**Algoritmo de Kruskal**](kruskal.md)

Algoritmo para encontrar a árvore geradora mínima (Minimum Spanning Tree - MST) de um grafo conectado e ponderado.

**Principais Características:**
- 🌳 Constrói árvore geradora mínima
- 📈 Ordena arestas por peso
- 🔗 Usa estrutura Union-Find
- ⚡ Eficiente para grafos esparsos

**Quando Usar:**
- Design de redes com custo mínimo
- Grafos esparsos (poucas arestas)
- Problemas de clustering
- Conexão de pontos com custo mínimo

**Aplicações:**
- Design de redes (elétrica, água, comunicação)
- Cabeamento de redes de computadores
- Planejamento de circuitos
- Análise de clusters

---

### 4. [**Algoritmo de Prim**](prim.md)

Outro algoritmo para encontrar a árvore geradora mínima, que cresce a árvore a partir de um vértice inicial.

**Principais Características:**
- 🌱 Cresce árvore a partir de um vértice
- 🔄 Adiciona vértices um por vez
- 📊 Usa fila de prioridade
- ⚡ Eficiente para grafos densos

**Quando Usar:**
- Grafos densos (muitas arestas)
- Quando a árvore cresce naturalmente de um ponto
- Implementações com matriz de adjacência
- Problemas de conectividade mínima

**Aplicações:**
- Design de redes de telecomunicações
- Redes de distribuição
- Análise de imagens (segmentação)
- Problemas de aproximação

---

### 5. [**Codificação de Huffman**](huffman.md)

Algoritmo para compressão de dados sem perdas que cria códigos de comprimento variável baseados na frequência dos símbolos.

**Principais Características:**
- 🗜️ Compressão sem perdas
- 🌲 Constrói árvore binária ótima
- 📊 Usa frequências dos símbolos
- ✅ Código de prefixo (não ambíguo)

**Quando Usar:**
- Compressão de dados
- Transmissão eficiente de informação
- Codificação de símbolos
- Redução de armazenamento

**Aplicações:**
- Formatos de arquivo (ZIP, JPEG, MP3)
- Compressão de texto
- Transmissão de dados
- Codificação em telecomunicações

---

### 6. [**Seleção de Atividades**](activity_selection.md)

Algoritmo clássico para selecionar o máximo número de atividades compatíveis que não se sobrepõem no tempo.

**Principais Características:**
- ⏰ Agenda atividades sem sobreposição
- 📅 Ordena por tempo de término
- 🎯 Maximiza número de atividades
- 📝 Exemplo clássico de algoritmo guloso

**Quando Usar:**
- Agendamento de tarefas
- Alocação de recursos
- Planejamento de eventos
- Otimização de uso de salas/equipamentos

**Aplicações:**
- Agendamento de salas de reunião
- Alocação de CPU/processadores
- Programação de eventos
- Otimização de linha de produção

---

## 🔍 Comparação Entre Algoritmos Gulosos

| Algoritmo | Tipo de Problema | Complexidade | Garante Ótimo? | Estrutura de Dados |
|-----------|-----------------|--------------|----------------|-------------------|
| Hill Climbing | Otimização local | O(n × vizinhos) | ❌ | Variável |
| Dijkstra | Caminho mais curto | O((V+E) log V) | ✅ | Fila de prioridade |
| Kruskal | MST | O(E log E) | ✅ | Union-Find |
| Prim | MST | O(E log V) | ✅ | Fila de prioridade |
| Huffman | Codificação | O(n log n) | ✅ | Árvore binária |
| Activity Selection | Agendamento | O(n log n) | ✅ | Array ordenado |

---

## ⚖️ Vantagens e Limitações

### **✅ Vantagens dos Algoritmos Gulosos**

1. **Simplicidade:** Fáceis de entender e implementar
2. **Eficiência:** Geralmente muito rápidos (polinomiais)
3. **Uso de Memória:** Baixo consumo de memória
4. **Elegância:** Código limpo e intuitivo
5. **Base Teórica:** Bem estudados e documentados

### **❌ Limitações**

1. **Não Sempre Ótimos:** Nem sempre encontram a solução ótima global
2. **Dependência do Problema:** Requerem propriedades específicas
3. **Sem Retrocesso:** Não podem desfazer escolhas ruins
4. **Análise Necessária:** Precisa provar que funcionam para o problema
5. **Ótimos Locais:** Podem ficar presos em soluções subótimas

---

## 🎓 Comparação: Guloso vs Outras Técnicas

### **Algoritmos Gulosos vs Programação Dinâmica**

| Aspecto | Guloso | Programação Dinâmica |
|---------|---------|---------------------|
| Estratégia | Escolha local ótima | Examina todas as opções |
| Complexidade | Geralmente menor | Geralmente maior |
| Memória | Baixo uso | Pode usar muita memória |
| Garantia de ótimo | Apenas para problemas específicos | Sempre (se aplicável) |
| Exemplo | Dijkstra | Floyd-Warshall |

### **Algoritmos Gulosos vs Backtracking**

| Aspecto | Guloso | Backtracking |
|---------|---------|--------------|
| Busca | Sem retrocesso | Com retrocesso |
| Exploração | Uma opção por vez | Todas as opções |
| Velocidade | Rápido | Pode ser lento |
| Solução | Pode ser subótima | Sempre ótima |
| Exemplo | Activity Selection | N-Queens |

---

## 🛠️ Como Provar que um Algoritmo Guloso Funciona

### **Método 1: Greedy Stays Ahead**
Mostre que em cada etapa, a solução gulosa está "à frente" de qualquer outra solução:

```
Para toda solução ótima O e solução gulosa G:
  Após k etapas, G está pelo menos tão bem quanto O
```

### **Método 2: Exchange Argument**
Transforme uma solução ótima na solução gulosa através de trocas que não pioram a solução:

```
1. Comece com uma solução ótima O
2. Troque elementos de O para se parecer com G
3. Mostre que cada troca mantém ou melhora a otimalidade
4. Conclua que G é ótima
```

### **Método 3: Indução**
Prove por indução que a escolha gulosa leva à solução ótima:

```
Base: A primeira escolha gulosa está em alguma solução ótima
Passo: Se as primeiras k escolhas são ótimas, a (k+1)-ésima também é
```

---

## 📖 Exemplos de Problemas Gulosos Clássicos

### **Problemas que Algoritmos Gulosos Resolvem Otimamente:**

1. ✅ **Seleção de Atividades** - Ordena por fim e escolhe compatíveis
2. ✅ **Árvore Geradora Mínima** - Kruskal e Prim
3. ✅ **Caminho Mais Curto (pesos não-negativos)** - Dijkstra
4. ✅ **Código de Huffman** - Compressão ótima
5. ✅ **Problema da Mochila Fracionária** - Pode dividir itens

### **Problemas que Algoritmos Gulosos NÃO Resolvem Otimamente:**

1. ❌ **Problema da Mochila 0-1** - Precisa programação dinâmica
2. ❌ **Caminho Mais Longo** - NP-completo
3. ❌ **Problema do Caixeiro Viajante** - Guloso dá aproximação
4. ❌ **Coloração de Grafos** - Guloso pode usar mais cores
5. ❌ **Particionamento de Conjuntos** - NP-completo

---

## 💡 Heurísticas e Aproximações Gulosas

Mesmo quando não garantem otimalidade, algoritmos gulosos são valiosos como:

### **Heurísticas Rápidas**
- Fornecem soluções "boas o suficiente" rapidamente
- Úteis quando tempo é limitado
- Base para otimizações posteriores

### **Algoritmos de Aproximação**
- Garantem qualidade relativa à solução ótima
- Exemplo: TSP guloso tem razão de aproximação conhecida
- Trade-off entre tempo e qualidade

### **Componentes de Meta-heurísticas**
- Hill Climbing como busca local
- Parte de algoritmos genéticos
- Inicialização para simulated annealing

---

## 🎯 Estratégias de Design de Algoritmos Gulosos

### **1. Identificar a Escolha Gulosa**
```
❓ Qual escolha local é "melhor"?
   - Menor peso?
   - Maior valor?
   - Menor tempo de término?
   - Maior frequência?
```

### **2. Provar Propriedade de Escolha Gulosa**
```
✅ A escolha local ótima leva ao ótimo global?
   - Use greedy stays ahead
   - Use exchange argument
   - Use prova por indução
```

### **3. Demonstrar Subestrutura Ótima**
```
🔍 O problema pode ser dividido em subproblemas?
   - Solução ótima contém soluções ótimas?
   - Independência de subproblemas?
```

### **4. Desenvolver Algoritmo Recursivo**
```python
def guloso(problema):
    if problema é trivial:
        return solução_trivial
    
    escolha = fazer_escolha_gulosa(problema)
    subproblema = reduzir(problema, escolha)
    return combinar(escolha, guloso(subproblema))
```

### **5. Converter para Forma Iterativa**
```python
def guloso_iterativo(problema):
    solucao = []
    while problema não está resolvido:
        escolha = fazer_escolha_gulosa(problema)
        solucao.append(escolha)
        atualizar(problema, escolha)
    return solucao
```

---

## 📚 Recursos de Aprendizado

### **Livros Recomendados**

1. **"Introduction to Algorithms" (CLRS)** - Capítulo 16: Greedy Algorithms
2. **"Algorithm Design" (Kleinberg & Tardos)** - Capítulo 4
3. **"The Algorithm Design Manual" (Skiena)** - Greedy Algorithms
4. **"Algorithms" (Sedgewick & Wayne)** - Greedy Approaches

### **Recursos Online**

1. **Visualizações:**
   - VisuAlgo.net - Visualização de algoritmos gulosos
   - Algorithm Visualizer - Animações interativas
   - Graph Online - Visualização de algoritmos em grafos

2. **Prática:**
   - LeetCode - Tag "Greedy"
   - HackerRank - Greedy Algorithms
   - Codeforces - Problemas gulosos
   - AtCoder - Greedy problems

3. **Tutoriais:**
   - GeeksforGeeks - Greedy Algorithms
   - CP-Algorithms - Greedy methods
   - TopCoder Tutorials - Greedy is Good

---

## 🎯 Conclusão

Os Algoritmos Gulosos representam uma das estratégias mais elegantes e eficientes em ciência da computação. Suas características principais são:

### **🔑 Principais Aprendizados**

1. **Simplicidade é Poder:** Escolhas locais simples podem levar a soluções globais ótimas
2. **Não Universais:** Funcionam apenas para problemas com propriedades específicas
3. **Eficiência:** Quando aplicáveis, são extremamente rápidos
4. **Fundamento Teórico:** Requerem prova de correção
5. **Versatilidade:** Úteis mesmo quando não garantem otimalidade

### **💭 Pensamento Guloso**

O "pensamento guloso" vai além dos algoritmos - é uma filosofia de resolução de problemas:

> *"Faça a melhor escolha no momento e não olhe para trás. Se o problema tem as propriedades certas, você chegará ao melhor resultado."*

### **🚀 Próximos Passos**

1. **Estude** cada algoritmo individualmente através dos links acima
2. **Implemente** os algoritmos em sua linguagem favorita
3. **Pratique** em plataformas de programação competitiva
4. **Aprenda** a provar correção de algoritmos gulosos
5. **Explore** quando usar guloso vs outras técnicas

### **🌟 Reflexão Final**

Algoritmos gulosos nos ensinam que, com as condições certas, ser "guloso" (fazer sempre a escolha que parece melhor no momento) não apenas é aceitável, mas é a estratégia ótima. Entender quando e por que isso funciona é uma habilidade fundamental em algoritmos e otimização.

---

**Voltar para:** [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

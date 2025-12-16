# Algoritmo de Kruskal: Árvore Geradora Mínima

O Algoritmo de Kruskal é um algoritmo guloso clássico para encontrar a Árvore Geradora Mínima (Minimum Spanning Tree - MST) de um grafo conectado e ponderado. Desenvolvido por Joseph Kruskal em 1956, é amplamente utilizado em problemas de design de redes, clustering e otimização de conexões com custo mínimo.

![Kruskal Concept](../../images/kruskal_concept.png)

---

## **1. O Conceito de Árvore Geradora Mínima**

### **1.1 O Problema MST**

Dado um grafo conectado G = (V, E) onde:
- **V** é o conjunto de vértices
- **E** é o conjunto de arestas com pesos
- Grafo é não-direcionado e conectado

**Objetivo:** Encontrar um subconjunto de arestas T ⊆ E tal que:
1. T conecta todos os vértices (é uma árvore geradora)
2. A soma dos pesos das arestas em T é mínima

### **1.2 Propriedades de uma Árvore Geradora**

Uma árvore geradora de um grafo com V vértices tem sempre:
- **Exatamente V-1 arestas**
- **Conecta todos os vértices**
- **Não contém ciclos**
- **É única** se todos os pesos são distintos

### **1.3 Analogia com Redes**

Imagine que você precisa conectar várias cidades com cabos de fibra óptica:
- **Vértices** = Cidades
- **Arestas** = Possíveis rotas para os cabos
- **Pesos** = Custo de instalação de cada cabo
- **Objetivo** = Conectar todas as cidades com o menor custo total

O algoritmo de Kruskal encontra a configuração ótima de cabos!

---

## **2. Como Funciona o Algoritmo de Kruskal**

### **2.1 Estratégia Gulosa**

A escolha gulosa do Kruskal é:
> **"Sempre selecione a aresta de menor peso que não forma ciclo com as arestas já escolhidas"**

### **2.2 Passos do Algoritmo**

```
🚀 INICIALIZAÇÃO:
   ├── T ← conjunto vazio (árvore em construção)
   ├── Ordenar todas as arestas E por peso crescente
   └── Criar conjunto disjunto para cada vértice

🔄 ITERAÇÃO (para cada aresta em ordem crescente):
   │
   ├── Seja (u, v) a aresta atual
   │
   ├── ❓ VERIFICAÇÃO DE CICLO
   │   └── u e v estão em componentes diferentes?
   │
   ├── ✅ SE NÃO FORMA CICLO:
   │   ├── Adicionar (u, v) a T
   │   └── Unir componentes de u e v
   │
   └── ❌ SE FORMA CICLO:
       └── Descartar (u, v)

🏆 RETORNAR T (árvore geradora mínima)
```

### **2.3 Visualização Passo a Passo**

Considere o grafo:

```
        2         3
    A ─────── B ─────── C
    │    ╲    │    ╱    │
   6│     ╲5  │7  ╱8    │9
    │      ╲  │  ╱      │
    D ─────── E ─────── F
        1         4
```

**Arestas ordenadas por peso:**
1. (D,E): 1
2. (A,B): 2
3. (B,C): 3
4. (E,F): 4
5. (A,E): 5
6. (A,D): 6
7. (B,E): 7
8. (C,E): 8
9. (C,F): 9

| Passo | Aresta | Peso | Forma Ciclo? | Ação | Componentes |
|-------|--------|------|--------------|------|-------------|
| 0 | - | - | - | Inicializar | {A}, {B}, {C}, {D}, {E}, {F} |
| 1 | (D,E) | 1 | ❌ Não | ✅ Adicionar | {A}, {B}, {C}, {D,E}, {F} |
| 2 | (A,B) | 2 | ❌ Não | ✅ Adicionar | {A,B}, {C}, {D,E}, {F} |
| 3 | (B,C) | 3 | ❌ Não | ✅ Adicionar | {A,B,C}, {D,E}, {F} |
| 4 | (E,F) | 4 | ❌ Não | ✅ Adicionar | {A,B,C}, {D,E,F} |
| 5 | (A,E) | 5 | ❌ Não | ✅ Adicionar | {A,B,C,D,E,F} |
| 6 | (A,D) | 6 | ✅ Sim | ❌ Rejeitar | {A,B,C,D,E,F} |
| 7 | (B,E) | 7 | ✅ Sim | ❌ Rejeitar | {A,B,C,D,E,F} |

**MST Final:**
- Arestas: (D,E), (A,B), (B,C), (E,F), (A,E)
- Peso total: 1 + 2 + 3 + 4 + 5 = **15**
- Número de arestas: **5 = 6-1** ✅

---

## **3. Estrutura Union-Find (Disjoint Set)**

### **3.1 O Que É Union-Find?**

Union-Find é uma estrutura de dados fundamental para o Kruskal, que mantém uma coleção de conjuntos disjuntos e suporta duas operações eficientes:

1. **FIND(x):** Descobre a qual conjunto x pertence
2. **UNION(x, y):** Une os conjuntos que contêm x e y

### **3.2 Implementação com Path Compression e Union by Rank**

```python
class UnionFind:
    """
    Estrutura Union-Find otimizada.
    Complexidade: O(α(n)) ≈ O(1) amortizado
    onde α é a função inversa de Ackermann (cresce MUITO lentamente)
    """
    
    def __init__(self, n):
        """
        Inicializa n conjuntos disjuntos.
        
        Args:
            n: número de elementos (0 a n-1)
        """
        self.pai = list(range(n))  # Cada elemento é seu próprio pai
        self.rank = [0] * n        # Rank (profundidade aproximada)
        self.num_componentes = n
    
    def find(self, x):
        """
        Encontra o representante (raiz) do conjunto de x.
        Usa path compression para otimização.
        
        Args:
            x: elemento a buscar
        
        Returns:
            Representante do conjunto de x
        """
        if self.pai[x] != x:
            # Path compression: fazer x apontar diretamente para a raiz
            self.pai[x] = self.find(self.pai[x])
        return self.pai[x]
    
    def union(self, x, y):
        """
        Une os conjuntos que contêm x e y.
        Usa union by rank para manter árvore balanceada.
        
        Args:
            x, y: elementos a unir
        
        Returns:
            True se união foi feita, False se já estavam no mesmo conjunto
        """
        raiz_x = self.find(x)
        raiz_y = self.find(y)
        
        if raiz_x == raiz_y:
            return False  # Já estão no mesmo conjunto
        
        # Union by rank: anexar árvore menor à maior
        if self.rank[raiz_x] < self.rank[raiz_y]:
            self.pai[raiz_x] = raiz_y
        elif self.rank[raiz_x] > self.rank[raiz_y]:
            self.pai[raiz_y] = raiz_x
        else:
            self.pai[raiz_y] = raiz_x
            self.rank[raiz_x] += 1
        
        self.num_componentes -= 1
        return True
    
    def conectados(self, x, y):
        """Verifica se x e y estão no mesmo componente."""
        return self.find(x) == self.find(y)
    
    def num_componentes_conectados(self):
        """Retorna o número de componentes disjuntos."""
        return self.num_componentes
```

### **3.3 Exemplo de Uso do Union-Find**

```python
# Criar Union-Find com 6 elementos (0-5)
uf = UnionFind(6)

print(f"Componentes iniciais: {uf.num_componentes_conectados()}")  # 6

# Conectar elementos
uf.union(0, 1)  # Unir 0 e 1
uf.union(2, 3)  # Unir 2 e 3
print(f"Após 2 uniões: {uf.num_componentes_conectados()}")  # 4

# Verificar conexões
print(f"0 e 1 conectados? {uf.conectados(0, 1)}")  # True
print(f"0 e 2 conectados? {uf.conectados(0, 2)}")  # False

# Mais uniões
uf.union(1, 2)  # Une {0,1} com {2,3}
print(f"Após unir conjuntos: {uf.num_componentes_conectados()}")  # 3
print(f"0 e 3 conectados agora? {uf.conectados(0, 3)}")  # True
```

---

## **4. Implementação Completa**

### **4.1 Classe Grafo**

```python
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Aresta:
    """Representa uma aresta ponderada."""
    u: int
    v: int
    peso: float
    
    def __lt__(self, outra):
        """Permite ordenação por peso."""
        return self.peso < outra.peso
    
    def __repr__(self):
        return f"({self.u}--{self.v}: {self.peso})"

class Grafo:
    """Grafo não-direcionado ponderado para algoritmo de Kruskal."""
    
    def __init__(self, num_vertices: int):
        """
        Inicializa grafo com num_vertices vértices.
        
        Args:
            num_vertices: número de vértices (numerados de 0 a n-1)
        """
        self.V = num_vertices
        self.arestas: List[Aresta] = []
    
    def adicionar_aresta(self, u: int, v: int, peso: float):
        """
        Adiciona aresta não-direcionada ao grafo.
        
        Args:
            u, v: vértices da aresta
            peso: peso da aresta
        """
        self.arestas.append(Aresta(u, v, peso))
    
    def __repr__(self):
        return f"Grafo({self.V} vértices, {len(self.arestas)} arestas)"
```

### **4.2 Algoritmo de Kruskal**

```python
def kruskal(grafo: Grafo) -> Tuple[List[Aresta], float]:
    """
    Implementa o algoritmo de Kruskal para encontrar MST.
    
    Args:
        grafo: Grafo não-direcionado e conectado
    
    Returns:
        Tupla (mst_arestas, peso_total) onde:
        - mst_arestas: lista de arestas na MST
        - peso_total: soma dos pesos da MST
    
    Complexidade: O(E log E) = O(E log V)
    - Ordenação: O(E log E)
    - Loop: O(E × α(V)) ≈ O(E)
    """
    # Ordenar arestas por peso crescente
    arestas_ordenadas = sorted(grafo.arestas)
    
    # Inicializar Union-Find
    uf = UnionFind(grafo.V)
    
    # MST em construção
    mst = []
    peso_total = 0
    
    # Processar arestas em ordem crescente
    for aresta in arestas_ordenadas:
        # Verificar se adicionar esta aresta forma ciclo
        if not uf.conectados(aresta.u, aresta.v):
            # Não forma ciclo: adicionar à MST
            mst.append(aresta)
            peso_total += aresta.peso
            uf.union(aresta.u, aresta.v)
            
            # Otimização: parar se MST está completa
            if len(mst) == grafo.V - 1:
                break
    
    return mst, peso_total


def kruskal_verboso(grafo: Grafo) -> Tuple[List[Aresta], float]:
    """
    Versão verbosa do Kruskal para fins educacionais.
    Imprime cada passo do algoritmo.
    """
    print("=" * 60)
    print("ALGORITMO DE KRUSKAL - EXECUÇÃO PASSO A PASSO")
    print("=" * 60)
    print(f"\n📊 Grafo: {grafo.V} vértices, {len(grafo.arestas)} arestas")
    
    # Ordenar arestas
    arestas_ordenadas = sorted(grafo.arestas)
    print(f"\n📋 Arestas ordenadas por peso:")
    for i, aresta in enumerate(arestas_ordenadas, 1):
        print(f"   {i}. {aresta}")
    
    # Inicializar Union-Find
    uf = UnionFind(grafo.V)
    mst = []
    peso_total = 0
    
    print(f"\n🔄 Processando arestas:\n")
    
    # Processar arestas
    for i, aresta in enumerate(arestas_ordenadas, 1):
        # Verificar se forma ciclo
        if not uf.conectados(aresta.u, aresta.v):
            # Adicionar à MST
            mst.append(aresta)
            peso_total += aresta.peso
            uf.union(aresta.u, aresta.v)
            
            print(f"✅ Passo {i}: {aresta} - ADICIONADA")
            print(f"   Componentes restantes: {uf.num_componentes_conectados()}")
            print(f"   Peso acumulado: {peso_total}")
            
            if len(mst) == grafo.V - 1:
                print(f"\n🎉 MST completa! ({len(mst)} arestas)")
                break
        else:
            print(f"❌ Passo {i}: {aresta} - REJEITADA (formaria ciclo)")
    
    print(f"\n" + "=" * 60)
    print(f"🏆 RESULTADO FINAL")
    print(f"=" * 60)
    print(f"Arestas na MST:")
    for aresta in mst:
        print(f"   {aresta}")
    print(f"\n💰 Peso total da MST: {peso_total}")
    print(f"=" * 60)
    
    return mst, peso_total
```

### **4.3 Exemplo de Uso**

```python
# Criar grafo do exemplo anterior
g = Grafo(6)  # Vértices A=0, B=1, C=2, D=3, E=4, F=5

# Adicionar arestas
g.adicionar_aresta(0, 1, 2)   # A-B: 2
g.adicionar_aresta(1, 2, 3)   # B-C: 3
g.adicionar_aresta(0, 3, 6)   # A-D: 6
g.adicionar_aresta(0, 4, 5)   # A-E: 5
g.adicionar_aresta(1, 4, 7)   # B-E: 7
g.adicionar_aresta(2, 4, 8)   # C-E: 8
g.adicionar_aresta(2, 5, 9)   # C-F: 9
g.adicionar_aresta(3, 4, 1)   # D-E: 1
g.adicionar_aresta(4, 5, 4)   # E-F: 4

# Executar Kruskal
mst, peso = kruskal(g)

print(f"Árvore Geradora Mínima:")
for aresta in mst:
    print(f"  {aresta}")
print(f"Peso total: {peso}")

# Versão verbosa (para aprendizado)
print("\n" + "="*80)
print("VERSÃO DETALHADA:")
print("="*80)
kruskal_verboso(g)
```

**Saída esperada:**
```
Árvore Geradora Mínima:
  (3--4: 1)
  (0--1: 2)
  (1--2: 3)
  (4--5: 4)
  (0--4: 5)
Peso total: 15.0
```

---

## **5. Análise de Complexidade**

### **5.1 Complexidade de Tempo**

```
FASE 1: Ordenação de arestas
   └── O(E log E)

FASE 2: Loop principal (E iterações)
   ├── Verificar ciclo: O(α(V)) ≈ O(1)
   ├── Union: O(α(V)) ≈ O(1)
   └── Total fase 2: O(E × α(V)) ≈ O(E)

COMPLEXIDADE TOTAL: O(E log E)
```

**Observações:**
- Como E ≤ V² em um grafo simples, temos E log E ≤ E log V²= 2E log V
- Portanto: **O(E log E) = O(E log V)**
- A ordenação domina a complexidade
- α(V) é a função inversa de Ackermann (praticamente constante)

### **5.2 Complexidade de Espaço**

```
💾 MEMÓRIA:
   ├── Arestas ordenadas: O(E)
   ├── Union-Find: O(V)
   ├── MST: O(V) (V-1 arestas)
   └── Total: O(E + V)
```

### **5.3 Comparação: Kruskal vs Prim**

| Aspecto | Kruskal | Prim |
|---------|---------|------|
| **Complexidade** | O(E log E) | O((V+E) log V) com heap |
| **Tipo de grafo** | Melhor para esparsos | Melhor para densos |
| **Estrutura de dados** | Union-Find + ordenação | Fila de prioridade |
| **Abordagem** | Baseada em arestas | Baseada em vértices |
| **Paralelização** | Mais difícil | Mais difícil ainda |
| **Quando usar** | E << V² | E ≈ V² |

**Escolha prática:**
- **Grafo esparso (E = O(V)):** Kruskal é ligeiramente melhor
- **Grafo denso (E = O(V²)):** Prim pode ser melhor
- **Na prática:** Ambos são muito rápidos para grafos moderados

---

## **6. Prova de Correção**

### **6.1 Teorema: Kruskal Produz MST**

**Teorema:** O algoritmo de Kruskal produz uma árvore geradora mínima para qualquer grafo conectado e ponderado.

**Prova (por contradição):**

1. **Suponha** que Kruskal produz árvore T que não é mínima
2. **Seja** T* uma MST verdadeira
3. **Seja** e = (u,v) a primeira aresta que Kruskal adiciona a T mas não está em T*
4. **Ao adicionar e** a T*, forma-se um ciclo C (pois T* já era árvore)
5. **No ciclo C**, deve haver outra aresta e' = (x,y) que conecta os mesmos componentes que e conectava quando foi adicionada
6. **Kruskal escolheu e** antes de e', logo peso(e) ≤ peso(e')
7. **Substituir e' por e** em T* produz outra árvore geradora T'
8. **peso(T')** ≤ peso(T*), mas T* era mínima, logo peso(T') = peso(T*)
9. **Logo T'** também é MST e contém e
10. **Repetindo** o argumento para cada aresta, T é MST ✅

### **6.2 Propriedade de Corte (Cut Property)**

**Definição:** Um corte em um grafo é uma partição dos vértices em dois conjuntos S e V-S.

**Propriedade de Corte:**
> Se uma aresta e = (u,v) é a aresta de menor peso que cruza um corte (u ∈ S, v ∈ V-S), então e está em alguma MST.

**Como Kruskal usa isso:**
- Quando Kruskal seleciona uma aresta (u,v), os vértices u e v estão em componentes diferentes
- Esses componentes formam um corte
- A aresta (u,v) é a de menor peso cruzando esse corte (pois arestas estão ordenadas)
- Logo, (u,v) está em alguma MST ✅

### **6.3 Invariante de Loop**

**Invariante:** Ao final de cada iteração, as arestas escolhidas até o momento estão contidas em alguma MST do grafo.

**Prova:**
- **Base:** Conjunto vazio está em toda MST ✅
- **Passo:** Se arestas até agora estão em alguma MST T*, e adicionamos aresta e:
  - e é a menor aresta que não forma ciclo
  - Por propriedade de corte, e está em alguma MST
  - Se T* não contém e, podemos trocar uma aresta para incluir e (como na prova principal)
  - Logo, invariante mantida ✅

---

## **7. Aplicações Práticas**

### **7.1 🌐 Design de Redes**

```python
class DesignerRede:
    """
    Sistema para projetar redes de menor custo.
    Aplicável a: redes elétricas, água, telecomunicações, etc.
    """
    
    def __init__(self):
        self.locais = {}  # id -> (nome, coordenadas)
        self.custos_conexao = []  # Lista de (local1, local2, custo)
    
    def adicionar_local(self, id_local: int, nome: str, lat: float, lon: float):
        """Adiciona um local a ser conectado."""
        self.locais[id_local] = (nome, lat, lon)
    
    def calcular_custo_conexao(self, local1: int, local2: int, 
                               custo_por_km: float = 1000) -> float:
        """
        Calcula custo de conectar dois locais.
        Baseado em distância euclidiana × custo por km.
        """
        nome1, lat1, lon1 = self.locais[local1]
        nome2, lat2, lon2 = self.locais[local2]
        
        # Distância euclidiana simplificada (para demo)
        distancia = ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5
        custo = distancia * custo_por_km
        
        self.custos_conexao.append((local1, local2, custo))
        return custo
    
    def projetar_rede_minima(self) -> dict:
        """
        Projeta rede de conexões de custo mínimo usando Kruskal.
        
        Returns:
            Dicionário com informações da rede:
            - conexoes: lista de conexões
            - custo_total: custo total do projeto
            - economia: economia vs conectar tudo
        """
        # Criar grafo
        n = len(self.locais)
        grafo = Grafo(n)
        
        # Calcular todos os custos possíveis
        for i in range(n):
            for j in range(i + 1, n):
                custo = self.calcular_custo_conexao(i, j)
                grafo.adicionar_aresta(i, j, custo)
        
        # Executar Kruskal
        mst, custo_minimo = kruskal(grafo)
        
        # Calcular economia
        custo_total_possivel = sum(a.peso for a in grafo.arestas)
        economia = custo_total_possivel - custo_minimo
        percentual_economia = (economia / custo_total_possivel) * 100
        
        # Formatar resultado
        conexoes = []
        for aresta in mst:
            nome1 = self.locais[aresta.u][0]
            nome2 = self.locais[aresta.v][0]
            conexoes.append({
                'de': nome1,
                'para': nome2,
                'custo': aresta.peso
            })
        
        return {
            'conexoes': conexoes,
            'custo_total': custo_minimo,
            'economia': economia,
            'percentual_economia': percentual_economia
        }

# Exemplo de uso
designer = DesignerRede()

# Adicionar cidades
designer.adicionar_local(0, "São Paulo", -23.5505, -46.6333)
designer.adicionar_local(1, "Rio de Janeiro", -22.9068, -43.1729)
designer.adicionar_local(2, "Belo Horizonte", -19.9167, -43.9345)
designer.adicionar_local(3, "Brasília", -15.7939, -47.8828)
designer.adicionar_local(4, "Curitiba", -25.4290, -49.2671)

# Projetar rede
resultado = designer.projetar_rede_minima()

print("🌐 PROJETO DE REDE DE CUSTO MÍNIMO")
print("=" * 50)
print(f"\n📡 Conexões necessárias:")
for conexao in resultado['conexoes']:
    print(f"   {conexao['de']} ↔ {conexao['para']}: "
          f"R$ {conexao['custo']:,.2f}")
print(f"\n💰 Custo total: R$ {resultado['custo_total']:,.2f}")
print(f"💵 Economia: R$ {resultado['economia']:,.2f} "
      f"({resultado['percentual_economia']:.1f}%)")
```

### **7.2 🔌 Circuitos e VLSI**

```python
class ProjetadorCircuito:
    """Design de circuitos integrados com Kruskal."""
    
    def __init__(self):
        self.componentes = {}
        self.conexoes_necessarias = []
    
    def adicionar_componente(self, id_comp: int, tipo: str, x: int, y: int):
        """Adiciona componente ao circuito."""
        self.componentes[id_comp] = {
            'tipo': tipo,
            'posicao': (x, y)
        }
    
    def adicionar_requisito_conexao(self, comp1: int, comp2: int):
        """Define que dois componentes precisam estar conectados."""
        self.conexoes_necessarias.append((comp1, comp2))
    
    def calcular_comprimento_fio(self, comp1: int, comp2: int) -> float:
        """Calcula comprimento Manhattan para roteamento."""
        x1, y1 = self.componentes[comp1]['posicao']
        x2, y2 = self.componentes[comp2]['posicao']
        return abs(x1 - x2) + abs(y1 - y2)  # Distância Manhattan
    
    def otimizar_roteamento(self):
        """
        Otimiza roteamento de conexões minimizando comprimento total de fios.
        """
        n = len(self.componentes)
        grafo = Grafo(n)
        
        # Adicionar apenas conexões necessárias
        for comp1, comp2 in self.conexoes_necessarias:
            comprimento = self.calcular_comprimento_fio(comp1, comp2)
            grafo.adicionar_aresta(comp1, comp2, comprimento)
        
        # Se precisar conectar tudo, adicionar todas as arestas
        # (para net que precisa conectar múltiplos componentes)
        
        mst, comprimento_total = kruskal(grafo)
        
        return {
            'rotas': mst,
            'comprimento_total_fios': comprimento_total
        }
```

### **7.3 🌍 Clustering Hierárquico**

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

class ClusteringKruskal:
    """
    Clustering hierárquico usando abordagem de Kruskal reversa.
    """
    
    def __init__(self, dados: np.ndarray):
        """
        Args:
            dados: array de shape (n_amostras, n_features)
        """
        self.dados = dados
        self.n_amostras = dados.shape[0]
        
        # Calcular matriz de distâncias
        self.distancias = squareform(pdist(dados))
    
    def cluster_hierarquico(self, k: int):
        """
        Agrupa dados em k clusters usando MST.
        
        Algoritmo:
        1. Construir MST com Kruskal
        2. Remover k-1 arestas mais pesadas
        3. Componentes resultantes são os clusters
        
        Args:
            k: número de clusters desejado
        
        Returns:
            Array de labels (cluster de cada amostra)
        """
        # Criar grafo completo com distâncias
        grafo = Grafo(self.n_amostras)
        for i in range(self.n_amostras):
            for j in range(i + 1, self.n_amostras):
                grafo.adicionar_aresta(i, j, self.distancias[i, j])
        
        # Construir MST
        mst, _ = kruskal(grafo)
        
        # Ordenar arestas da MST por peso decrescente
        mst_ordenada = sorted(mst, key=lambda a: a.peso, reverse=True)
        
        # Remover k-1 arestas mais pesadas
        arestas_manter = mst_ordenada[k-1:]
        
        # Construir componentes finais
        uf = UnionFind(self.n_amostras)
        for aresta in arestas_manter:
            uf.union(aresta.u, aresta.v)
        
        # Atribuir labels
        labels = np.zeros(self.n_amostras, dtype=int)
        clusters = {}
        cluster_id = 0
        
        for i in range(self.n_amostras):
            raiz = uf.find(i)
            if raiz not in clusters:
                clusters[raiz] = cluster_id
                cluster_id += 1
            labels[i] = clusters[raiz]
        
        return labels

# Exemplo de uso
# Gerar dados de exemplo
np.random.seed(42)
dados = np.random.randn(100, 2)
dados[:30] += [5, 5]  # Cluster 1
dados[30:60] += [-5, -5]  # Cluster 2
dados[60:] += [5, -5]  # Cluster 3

clustering = ClusteringKruskal(dados)
labels = clustering.cluster_hierarquico(k=3)

print(f"Clusters atribuídos: {set(labels)}")
print(f"Distribuição: {[(i, sum(labels == i)) for i in set(labels)]}")
```

---

## **8. Variações e Extensões**

### **8.1 🌲 Floresta Geradora Mínima**

Para grafos desconectados:

```python
def kruskal_floresta(grafo: Grafo) -> Tuple[List[List[Aresta]], float]:
    """
    Adaptação para grafos desconectados.
    Retorna múltiplas MSTs (uma por componente).
    """
    arestas_ordenadas = sorted(grafo.arestas)
    uf = UnionFind(grafo.V)
    
    # Processar todas as arestas
    florestas = {i: [] for i in range(grafo.V)}
    peso_total = 0
    
    for aresta in arestas_ordenadas:
        if not uf.conectados(aresta.u, aresta.v):
            raiz = uf.find(aresta.u)
            florestas[raiz].append(aresta)
            peso_total += aresta.peso
            uf.union(aresta.u, aresta.v)
    
    # Filtrar componentes vazios e agrupar
    arvores = [arestas for arestas in florestas.values() if arestas]
    
    return arvores, peso_total
```

### **8.2 📊 Kruskal com Restrições**

```python
def kruskal_com_restricoes(grafo: Grafo, 
                          arestas_obrigatorias: List[Tuple[int, int]],
                          arestas_proibidas: List[Tuple[int, int]]) -> Tuple[List[Aresta], float]:
    """
    Kruskal com arestas que devem/não devem ser incluídas.
    
    Args:
        arestas_obrigatorias: arestas que DEVEM estar na MST
        arestas_proibidas: arestas que NÃO PODEM estar na MST
    """
    # Converter para conjunto para busca rápida
    obrigatorias = set(arestas_obrigatorias)
    proibidas = set(arestas_proibidas)
    
    uf = UnionFind(grafo.V)
    mst = []
    peso_total = 0
    
    # FASE 1: Adicionar arestas obrigatórias primeiro
    for aresta in grafo.arestas:
        par = (min(aresta.u, aresta.v), max(aresta.u, aresta.v))
        if par in obrigatorias:
            if not uf.conectados(aresta.u, aresta.v):
                mst.append(aresta)
                peso_total += aresta.peso
                uf.union(aresta.u, aresta.v)
    
    # FASE 2: Kruskal normal, evitando proibidas
    arestas_ordenadas = sorted(grafo.arestas)
    
    for aresta in arestas_ordenadas:
        par = (min(aresta.u, aresta.v), max(aresta.u, aresta.v))
        
        # Pular se proibida ou já processada
        if par in proibidas or par in obrigatorias:
            continue
        
        if not uf.conectados(aresta.u, aresta.v):
            mst.append(aresta)
            peso_total += aresta.peso
            uf.union(aresta.u, aresta.v)
            
            if len(mst) == grafo.V - 1:
                break
    
    return mst, peso_total
```

### **8.3 🎯 MST de Grau Limitado**

```python
def kruskal_grau_limitado(grafo: Grafo, grau_maximo: int) -> Tuple[List[Aresta], float]:
    """
    Variação que limita o grau máximo de cada vértice.
    Útil quando há restrições físicas de conexões.
    """
    arestas_ordenadas = sorted(grafo.arestas)
    uf = UnionFind(grafo.V)
    
    # Rastrear grau de cada vértice
    graus = [0] * grafo.V
    
    mst = []
    peso_total = 0
    
    for aresta in arestas_ordenadas:
        # Verificar restrições de grau
        if graus[aresta.u] >= grau_maximo or graus[aresta.v] >= grau_maximo:
            continue
        
        if not uf.conectados(aresta.u, aresta.v):
            mst.append(aresta)
            peso_total += aresta.peso
            uf.union(aresta.u, aresta.v)
            
            # Atualizar graus
            graus[aresta.u] += 1
            graus[aresta.v] += 1
            
            if len(mst) == grafo.V - 1:
                break
    
    return mst, peso_total
```

---

## **9. Exercícios Práticos**

### **9.1 🎯 Nível Básico**

#### **Exercício 1: Implementação Manual**
```python
"""
Implemente Kruskal sem usar a classe UnionFind pronta.
Use uma abordagem simples com listas para rastrear componentes.
"""

def kruskal_simples(grafo):
    # Seu código aqui
    # Dica: use uma lista onde componentes[i] = id do componente do vértice i
    pass
```

#### **Exercício 2: Verificação de MST**
```python
"""
Dado um grafo e uma suposta MST, verifique se ela é realmente mínima.
"""

def verificar_mst(grafo: Grafo, mst_candidata: List[Aresta]) -> bool:
    """
    Retorna True se mst_candidata é realmente uma MST de grafo.
    
    Verificações necessárias:
    1. É uma árvore geradora? (V-1 arestas, conecta todos os vértices)
    2. É mínima? (nenhuma aresta pode ser trocada por uma mais leve)
    """
    # Seu código aqui
    pass
```

### **9.2 🎯 Nível Intermediário**

#### **Exercício 3: Segunda Melhor MST**
```python
"""
Encontre a segunda melhor MST (a MST de segundo menor peso).
Algoritmo: Para cada aresta na MST, tente removê-la e encontrar nova MST.
"""

def segunda_melhor_mst(grafo: Grafo) -> Tuple[List[Aresta], float]:
    """
    Encontra a MST de segundo menor peso.
    
    Complexidade: O(E² log E) - pode ser melhorada para O(VE)
    """
    # Seu código aqui
    pass
```

#### **Exercício 4: MST Dinâmica**
```python
"""
Implemente estrutura que mantém MST e atualiza eficientemente
quando arestas são adicionadas/removidas.
"""

class MSTDinamica:
    def __init__(self, num_vertices: int):
        self.V = num_vertices
        self.arestas = []
        self.mst = []
        self.peso_mst = 0
    
    def adicionar_aresta(self, u: int, v: int, peso: float):
        """Adiciona aresta e atualiza MST se necessário."""
        # Seu código aqui
        pass
    
    def remover_aresta(self, u: int, v: int):
        """Remove aresta e recalcula MST se necessário."""
        # Seu código aqui
        pass
```

### **9.3 🎯 Nível Avançado**

#### **Exercício 5: MST Ótima com K Arestas Específicas**
```python
"""
Dado um conjunto de k arestas, encontre a MST de menor peso
que contém pelo menos k' dessas arestas (k' ≤ k).
"""

def mst_com_k_arestas_preferidas(grafo: Grafo, 
                                  arestas_preferidas: List[Tuple[int, int]],
                                  k_minimo: int) -> Tuple[List[Aresta], float]:
    """
    MST que tenta incluir o máximo possível das arestas preferidas.
    """
    # Desafio: balancear peso total vs número de arestas preferidas
    pass
```

#### **Exercício 6: Kruskal Paralelo**
```python
"""
Implemente versão paralela do Kruskal.
Desafio: Union-Find é inerentemente sequencial. Como paralelizar?
"""

import multiprocessing

def kruskal_paralelo(grafo: Grafo, num_processos: int = 4):
    """
    Paraleliza a construção da MST.
    
    Ideias:
    - Particionar arestas por faixa de peso
    - Processar partições em paralelo
    - Mesclar resultados
    """
    # Desafio avançado
    pass
```

---

## **10. Recursos e Referências**

### **10.1 📚 Leitura Fundamental**

1. **"Introduction to Algorithms" (CLRS)** - Capítulo 23
   - Prova completa de correção
   - Análise amortizada do Union-Find
   - Comparação Kruskal vs Prim

2. **"Algorithm Design" (Kleinberg & Tardos)** - Capítulo 4.5
   - Introdução intuitiva
   - Múltiplos exemplos práticos
   - Propriedade de corte explicada

3. **"The Design and Analysis of Computer Algorithms" (Aho, Hopcroft, Ullman)**
   - Análise clássica
   - Provas formais

### **10.2 🌐 Recursos Online**

**Visualizações:**
- VisuAlgo: https://visualgo.net/en/mst
- Algorithm Visualizer: Kruskal Animation
- Graph Online: MST Tools

**Tutoriais:**
- GeeksforGeeks: Kruskal's Algorithm
- CP-Algorithms: Minimum Spanning Tree
- Coursera: Algorithms on Graphs

### **10.3 🛠️ Bibliotecas**

```python
# NetworkX
import networkx as nx
G = nx.Graph()
G.add_weighted_edges_from([(0,1,2), (1,2,3), (0,2,4)])
mst = nx.minimum_spanning_tree(G, algorithm='kruskal')

# SciPy
from scipy.sparse.csgraph import minimum_spanning_tree
mst = minimum_spanning_tree(distance_matrix)

# graph-tool
import graph_tool.all as gt
tree = gt.min_spanning_tree(g)
```

---

## **11. 🎯 Conclusão**

O Algoritmo de Kruskal é um exemplo perfeito de como a estratégia gulosa pode produzir soluções ótimas quando aplicada ao problema certo.

### **🔑 Principais Aprendizados**

1. **Elegância da Abordagem:** Ordenar e processar arestas é surpreendentemente eficaz
2. **Importância do Union-Find:** Estrutura de dados crucial para eficiência
3. **Provas de Correção:** Propriedade de corte garante optimalidade
4. **Aplicabilidade Universal:** Útil em diversos domínios práticos
5. **Trade-offs:** Kruskal vs Prim dependem da densidade do grafo

### **💡 Quando Usar Kruskal**

| **✅ Use quando:** | **❌ Evite quando:** |
|-------------------|---------------------|
| Grafo esparso (E << V²) | Grafo muito denso |
| Arestas já ordenadas | Apenas algumas arestas relevantes |
| Implementação simples necessária | Grafos dinâmicos |
| Floresta geradora necessária | MST direcionada (não existe!) |

### **🚀 Próximos Passos**

1. **Implemente** do zero para entendimento profundo
2. **Compare** com algoritmo de Prim
3. **Estude** Union-Find em profundidade
4. **Explore** aplicações em sua área
5. **Pratique** problemas de competição

### **🌟 Reflexão Final**

Kruskal nos ensina que problemas complexos podem ter soluções surpreendentemente simples. Ordenar arestas e selecionar gulosa mente é tudo que precisamos para conectar qualquer conjunto de pontos com custo mínimo - uma ideia poderosa com aplicações em todo lugar, de redes de computadores a design de circuitos.

> *"Conecte os pontos de forma gulosa, e o resultado será ótimo!"*

---

**Voltar para:** [Documentação de Algoritmos Gulosos](README.md) | [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

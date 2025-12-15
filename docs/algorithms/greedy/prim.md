# Algoritmo de Prim: Árvore Geradora Mínima

O Algoritmo de Prim é outro algoritmo guloso clássico para encontrar a Árvore Geradora Mínima (MST) de um grafo conectado e ponderado. Desenvolvido por Robert C. Prim em 1957 (e redescoberto por Dijkstra em 1959), é particularmente eficiente para grafos densos e é amplamente utilizado em design de redes e problemas de conectividade.

![Prim Concept](../../images/prim_concept.png)

---

## **1. O Conceito do Algoritmo de Prim**

### **1.1 Diferença entre Prim e Kruskal**

Ambos resolvem o problema MST, mas com abordagens diferentes:

| Aspecto | Prim | Kruskal |
|---------|------|---------|
| **Estratégia** | Cresce árvore a partir de um vértice | Processa arestas globalmente |
| **Foco** | Baseado em vértices | Baseado em arestas |
| **Estrutura** | Fila de prioridade de vértices | Union-Find + ordenação |
| **Melhor para** | Grafos densos | Grafos esparsos |
| **Crescimento** | Sempre conectada | Pode ter múltiplos componentes |

### **1.2 Analogia com Crescimento de Árvore**

Imagine plantar uma árvore que cresce adicionando galhos:
- **Início:** Planta a semente (vértice inicial)
- **Crescimento:** Sempre adiciona o galho mais barato que estende a árvore
- **Processo:** A árvore permanece conectada em todas as etapas
- **Fim:** Quando todos os pontos estão conectados

### **1.3 Propriedade Gulosa**

A escolha gulosa do Prim é:
> **"Sempre adicione a aresta de menor peso que conecta um vértice na árvore a um vértice fora dela"**

---

## **2. Como Funciona o Algoritmo de Prim**

### **2.1 Estruturas de Dados**

```
📊 ESTRUTURAS:
   ├── key[v] : peso mínimo de aresta conectando v à árvore
   ├── parent[v] : vértice pai de v na MST
   ├── inMST[v] : v já está na MST?
   └── fila_prioridade : vértices ordenados por key[]
```

### **2.2 Passos do Algoritmo**

```
🚀 INICIALIZAÇÃO:
   ├── key[início] ← 0
   ├── key[v] ← ∞ para todos os outros vértices
   ├── parent[v] ← NULL para todos os vértices
   ├── inMST[v] ← False para todos os vértices
   └── Adicionar todos os vértices à fila de prioridade

🔄 ITERAÇÃO (enquanto fila não está vazia):
   │
   ├── 1️⃣ EXTRAÇÃO
   │   ├── u ← extrair vértice com menor key[]
   │   └── inMST[u] ← True
   │
   └── 2️⃣ ATUALIZAÇÃO DOS VIZINHOS
       └── Para cada vizinho v de u não em MST:
           ├── peso_aresta ← peso(u, v)
           └── SE peso_aresta < key[v]:
               ├── key[v] ← peso_aresta
               ├── parent[v] ← u
               └── Atualizar v na fila de prioridade

🏆 RESULTADO:
   └── parent[] define as arestas da MST
```

### **2.3 Visualização Passo a Passo**

Considere o grafo (mesmo exemplo do Kruskal para comparação):

```
        2         3
    A ─────── B ─────── C
    │    ╲    │    ╱    │
   6│     ╲5  │7  ╱8    │9
    │      ╲  │  ╱      │
    D ─────── E ─────── F
        1         4
```

**Começando de A:**

| Iteração | u | key[A] | key[B] | key[C] | key[D] | key[E] | key[F] | parent[] | inMST |
|----------|---|--------|--------|--------|--------|--------|--------|----------|-------|
| 0 (init) | - | 0 | ∞ | ∞ | ∞ | ∞ | ∞ | - | {} |
| 1 | A | 0 | 2 | ∞ | 6 | 5 | ∞ | A→B,D,E | {A} |
| 2 | B | 0 | 2 | 3 | 6 | 5 | ∞ | A→B,D; B→C | {A,B} |
| 3 | C | 0 | 2 | 3 | 6 | 5 | 9 | C→F | {A,B,C} |
| 4 | E | 0 | 2 | 3 | 1 | 5 | 4 | E→D,F | {A,B,C,E} |
| 5 | D | 0 | 2 | 3 | 1 | 5 | 4 | - | {A,B,C,E,D} |
| 6 | F | 0 | 2 | 3 | 1 | 5 | 4 | - | {A,B,C,E,D,F} |

**MST Final (arestas definidas por parent[]):**
- A → B (2)
- B → C (3)
- A → E (5)
- E → D (1)
- E → F (4)
- **Peso total: 15** (mesmo que Kruskal!)

---

## **3. Implementação**

### **3.1 Pseudocódigo Completo**

```python
função PRIM(Grafo G, vértice início):
    # Inicialização
    para cada vértice v em G.vertices:
        key[v] ← INFINITO
        parent[v] ← NULL
        inMST[v] ← False
    
    key[início] ← 0
    
    # Criar fila de prioridade
    Q ← FILA_PRIORIDADE(G.vertices, chave=key)
    
    # Loop principal
    enquanto Q não está vazia:
        u ← Q.extrair_minimo()
        inMST[u] ← True
        
        # Atualizar vizinhos
        para cada vizinho v de u:
            se v não está em inMST:
                peso_aresta ← peso(u, v)
                
                se peso_aresta < key[v]:
                    key[v] ← peso_aresta
                    parent[v] ← u
                    Q.diminuir_chave(v, peso_aresta)
    
    # Construir MST a partir de parent[]
    MST ← lista vazia
    peso_total ← 0
    
    para cada vértice v (exceto início):
        se parent[v] ≠ NULL:
            MST.adicionar((parent[v], v, key[v]))
            peso_total ← peso_total + key[v]
    
    retornar (MST, peso_total)
```

### **3.2 Implementação em Python com Heap**

```python
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Aresta:
    """Representa uma aresta ponderada."""
    u: int
    v: int
    peso: float
    
    def __repr__(self):
        return f"({self.u}--{self.v}: {self.peso})"

class GrafoPrim:
    """Grafo não-direcionado ponderado para algoritmo de Prim."""
    
    def __init__(self, num_vertices: int):
        self.V = num_vertices
        self.adj = defaultdict(list)  # Lista de adjacência: u -> [(v, peso), ...]
    
    def adicionar_aresta(self, u: int, v: int, peso: float):
        """Adiciona aresta não-direcionada."""
        self.adj[u].append((v, peso))
        self.adj[v].append((u, peso))
    
    def __repr__(self):
        return f"Grafo({self.V} vértices)"


def prim(grafo: GrafoPrim, inicio: int = 0) -> Tuple[List[Aresta], float]:
    """
    Implementa o algoritmo de Prim para encontrar MST.
    
    Args:
        grafo: Grafo não-direcionado e conectado
        inicio: Vértice inicial (padrão: 0)
    
    Returns:
        Tupla (mst_arestas, peso_total)
    
    Complexidade: O((V + E) log V) com heap binário
    """
    # Inicialização
    key = [float('inf')] * grafo.V
    parent = [None] * grafo.V
    inMST = [False] * grafo.V
    
    key[inicio] = 0
    
    # Fila de prioridade: (key, vértice)
    heap = [(0, inicio)]
    
    while heap:
        # Extrair vértice com menor key
        k, u = heapq.heappop(heap)
        
        # Ignorar se já processado
        if inMST[u]:
            continue
        
        inMST[u] = True
        
        # Atualizar vizinhos
        for v, peso in grafo.adj[u]:
            if not inMST[v] and peso < key[v]:
                key[v] = peso
                parent[v] = u
                heapq.heappush(heap, (peso, v))
    
    # Construir MST
    mst = []
    peso_total = 0
    
    for v in range(grafo.V):
        if parent[v] is not None:
            mst.append(Aresta(parent[v], v, key[v]))
            peso_total += key[v]
    
    return mst, peso_total


def prim_verboso(grafo: GrafoPrim, inicio: int = 0) -> Tuple[List[Aresta], float]:
    """Versão verbosa do Prim para fins educacionais."""
    print("=" * 60)
    print("ALGORITMO DE PRIM - EXECUÇÃO PASSO A PASSO")
    print("=" * 60)
    print(f"\n📊 Grafo: {grafo.V} vértices")
    print(f"🌱 Vértice inicial: {inicio}")
    
    # Inicialização
    key = [float('inf')] * grafo.V
    parent = [None] * grafo.V
    inMST = [False] * grafo.V
    key[inicio] = 0
    
    heap = [(0, inicio)]
    iteracao = 0
    
    print(f"\n🔄 Crescendo a árvore:\n")
    
    while heap:
        k, u = heapq.heappop(heap)
        
        if inMST[u]:
            continue
        
        inMST[u] = True
        iteracao += 1
        
        print(f"✅ Iteração {iteracao}: Adicionar vértice {u} à MST")
        if parent[u] is not None:
            print(f"   Aresta: {parent[u]} → {u} (peso: {key[u]})")
        
        # Atualizar vizinhos
        vizinhos_atualizados = []
        for v, peso in grafo.adj[u]:
            if not inMST[v] and peso < key[v]:
                old_key = key[v]
                key[v] = peso
                parent[v] = u
                heapq.heappush(heap, (peso, v))
                vizinhos_atualizados.append((v, peso, old_key))
        
        if vizinhos_atualizados:
            print(f"   Vizinhos atualizados:")
            for v, novo, antigo in vizinhos_atualizados:
                print(f"      {v}: key {antigo} → {novo}")
        
        print(f"   Vértices na MST: {[i for i in range(grafo.V) if inMST[i]]}")
        print()
    
    # Construir MST
    mst = []
    peso_total = 0
    
    for v in range(grafo.V):
        if parent[v] is not None:
            mst.append(Aresta(parent[v], v, key[v]))
            peso_total += key[v]
    
    print("=" * 60)
    print("🏆 RESULTADO FINAL")
    print("=" * 60)
    print("Arestas na MST:")
    for aresta in mst:
        print(f"   {aresta}")
    print(f"\n💰 Peso total da MST: {peso_total}")
    print("=" * 60)
    
    return mst, peso_total


# Exemplo de uso
if __name__ == "__main__":
    # Criar grafo do exemplo
    g = GrafoPrim(6)  # Vértices A=0, B=1, C=2, D=3, E=4, F=5
    
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
    
    # Executar Prim
    mst, peso = prim(g)
    
    print("Árvore Geradora Mínima (Prim):")
    for aresta in mst:
        print(f"  {aresta}")
    print(f"Peso total: {peso}")
    
    print("\n" + "="*80)
    print("VERSÃO DETALHADA:")
    print("="*80)
    prim_verboso(g)
```

---

## **4. Análise de Complexidade**

### **4.1 Complexidade de Tempo**

A complexidade depende da implementação da fila de prioridade:

| Implementação | Extrair Mínimo | Diminuir Chave | Complexidade Total |
|--------------|----------------|----------------|-------------------|
| Array simples | O(V) | O(1) | **O(V²)** |
| Heap binário | O(log V) | O(log V) | **O((V + E) log V)** |
| Heap Fibonacci | O(log V) amort. | O(1) amort. | **O(E + V log V)** |

**Análise detalhada com heap binário:**
```
V extrações do mínimo: V × O(log V) = O(V log V)
E atualizações de chave: E × O(log V) = O(E log V)
Total: O((V + E) log V)
```

Para grafos densos (E ≈ V²): O(V² log V)
Para grafos esparsos (E ≈ V): O(V log V)

### **4.2 Complexidade de Espaço**

```
💾 MEMÓRIA:
   ├── key[]: O(V)
   ├── parent[]: O(V)
   ├── inMST[]: O(V)
   ├── heap: O(V)
   ├── Lista de adjacência: O(V + E)
   └── Total: O(V + E)
```

### **4.3 Comparação Detalhada: Prim vs Kruskal**

| Característica | Prim | Kruskal |
|----------------|------|---------|
| **Complexidade (heap)** | O((V+E) log V) | O(E log E) = O(E log V) |
| **Grafo esparso (E≈V)** | O(V log V) | O(V log V) |
| **Grafo denso (E≈V²)** | O(V² log V) | O(V² log V) |
| **Implementação array** | O(V²) ótimo para denso | Não aplicável |
| **Paralelização** | Difícil | Possível |
| **Ordem de processamento** | Vértices (BFS-like) | Arestas (ordenadas) |
| **Estrutura intermediária** | Sempre conexa | Pode ser desconecta |

**Quando usar cada um:**

```python
def escolher_algoritmo_mst(num_vertices, num_arestas):
    """Heurística para escolher algoritmo MST."""
    densidade = num_arestas / (num_vertices * (num_vertices - 1) / 2)
    
    if densidade > 0.5:
        return "Prim com array (O(V²))"
    elif densidade > 0.3:
        return "Prim com heap binário"
    else:
        return "Kruskal"
```

---

## **5. Prova de Correção**

### **5.1 Invariante de Loop**

**Invariante:** Após cada iteração, as arestas escolhidas formam uma árvore T que está contida em alguma MST do grafo.

**Prova por indução:**

**Base:** T = ∅ está em toda MST. ✅

**Passo indutivo:**
1. Suponha T está em alguma MST T*
2. Prim adiciona aresta e = (u,v) onde u ∈ T e v ∉ T, e é a menor aresta cruzando o corte
3. **Se e ∈ T*:** ótimo, T ∪ {e} ⊆ T* ✅
4. **Se e ∉ T*:** 
   - Adicionar e a T* cria ciclo C
   - C contém outra aresta e' = (u',v') cruzando o mesmo corte
   - peso(e) ≤ peso(e') (e foi escolhida primeiro)
   - Substituir e' por e em T* produz T** também MST
   - T ∪ {e} ⊆ T** ✅

### **5.2 Propriedade de Corte (Revisitada)**

**Definição:** Corte (S, V-S) particiona vértices em dois conjuntos.

**Lema:** Se e é a aresta de menor peso cruzando um corte e nenhuma aresta do corte está na MST parcial, então e está em alguma MST.

**Aplicação no Prim:**
- A cada iteração, T (vértices na MST) e V-T formam um corte
- Prim escolhe a menor aresta cruzando esse corte
- Logo, a escolha é segura ✅

### **5.3 Unicidade da MST**

**Teorema:** Se todos os pesos das arestas são distintos, a MST é única.

**Prova:**
1. Suponha duas MSTs distintas T₁ e T₂
2. Seja e a aresta de menor peso em T₁ mas não em T₂
3. Adicionar e a T₂ cria ciclo com alguma aresta e' ∉ T₁
4. Como todos os pesos são distintos, peso(e) ≠ peso(e')
5. Se peso(e) < peso(e'): substituir e' por e reduz peso de T₂, contradição
6. Se peso(e') < peso(e): substituir e por e' reduz peso de T₁, contradição
7. Logo, T₁ = T₂ ✅

---

## **6. Variações e Otimizações**

### **6.1 Prim com Matriz de Adjacência**

Para grafos densos, implementação com array é O(V²) e mais simples:

```python
def prim_matriz(matriz_adj: List[List[float]], inicio: int = 0) -> Tuple[List[Aresta], float]:
    """
    Prim para grafos densos usando matriz de adjacência.
    Não usa heap - O(V²) mas constante menor.
    
    Args:
        matriz_adj: matriz V×V com pesos (∞ para ausência de aresta)
        inicio: vértice inicial
    
    Returns:
        Tupla (mst_arestas, peso_total)
    
    Complexidade: O(V²)
    """
    V = len(matriz_adj)
    
    key = [float('inf')] * V
    parent = [None] * V
    inMST = [False] * V
    
    key[inicio] = 0
    
    for _ in range(V):
        # Encontrar vértice não processado com menor key
        u = -1
        min_key = float('inf')
        
        for v in range(V):
            if not inMST[v] and key[v] < min_key:
                min_key = key[v]
                u = v
        
        if u == -1:
            break
        
        inMST[u] = True
        
        # Atualizar vizinhos
        for v in range(V):
            # Se há aresta u-v, v não está na MST, e peso é menor
            if matriz_adj[u][v] != float('inf') and not inMST[v]:
                if matriz_adj[u][v] < key[v]:
                    key[v] = matriz_adj[u][v]
                    parent[v] = u
    
    # Construir MST
    mst = []
    peso_total = 0
    
    for v in range(V):
        if parent[v] is not None:
            mst.append(Aresta(parent[v], v, key[v]))
            peso_total += key[v]
    
    return mst, peso_total
```

### **6.2 Prim com Heap de Fibonacci**

Heap de Fibonacci oferece O(E + V log V) mas é complexo:

```python
from fibonacci_heap_mod import Fibonacci_heap

def prim_fibonacci(grafo: GrafoPrim, inicio: int = 0):
    """
    Prim com Heap de Fibonacci.
    Complexidade: O(E + V log V) amortizado
    
    Nota: Raramente usado na prática devido a constantes altas
    """
    V = grafo.V
    
    # Estruturas
    fib_heap = Fibonacci_heap()
    nodes = {}  # vértice -> nó no heap
    parent = [None] * V
    inMST = [False] * V
    
    # Inserir todos os vértices
    for v in range(V):
        key = 0 if v == inicio else float('inf')
        nodes[v] = fib_heap.insert(key, v)
    
    mst = []
    peso_total = 0
    
    while fib_heap.total_nodes > 0:
        # Extrair mínimo: O(log V) amortizado
        u_node = fib_heap.extract_min()
        u = u_node.value
        inMST[u] = True
        
        if parent[u] is not None:
            peso = u_node.key
            mst.append(Aresta(parent[u], u, peso))
            peso_total += peso
        
        # Atualizar vizinhos: diminuir_chave é O(1) amortizado
        for v, peso in grafo.adj[u]:
            if not inMST[v]:
                v_node = nodes[v]
                if peso < v_node.key:
                    fib_heap.decrease_key(v_node, peso)
                    parent[v] = u
    
    return mst, peso_total
```

### **6.3 Prim Paralelo**

```python
import multiprocessing as mp
from queue import PriorityQueue

def prim_paralelo(grafo: GrafoPrim, inicio: int = 0, num_threads: int = 4):
    """
    Tentativa de paralelizar Prim (desafiador!).
    
    Estratégia:
    1. Particionar grafo em regiões
    2. Executar Prim em cada região em paralelo
    3. Mesclar MSTs das regiões
    
    Nota: Não oferece speedup significativo devido à natureza sequencial
    """
    # Implementação desafiadora - Prim é inerentemente sequencial
    # Alternativa: usar Borůvka (paralelizável) ou Kruskal paralelo
    pass
```

---

## **7. Aplicações Práticas**

### **7.1 🌐 Redes de Telecomunicações**

```python
class RedeTelecomunicacoes:
    """
    Projetar rede de telecomunicações com Prim.
    Útil quando o crescimento é naturalmente incremental.
    """
    
    def __init__(self, central: int):
        """
        Args:
            central: ID do nó central (ponto de partida natural)
        """
        self.central = central
        self.grafo = None
        self.locais = {}
    
    def adicionar_local(self, id_local: int, nome: str, tipo: str):
        """Adiciona local à rede."""
        self.locais[id_local] = {
            'nome': nome,
            'tipo': tipo  # 'central', 'subestacao', 'terminal'
        }
    
    def projetar_rede_incremental(self, custos_conexao):
        """
        Projeta rede crescendo a partir da central.
        Prim é natural aqui - a rede cresce da central para fora.
        """
        n = len(self.locais)
        self.grafo = GrafoPrim(n)
        
        # Adicionar todas as conexões possíveis
        for (u, v), custo in custos_conexao.items():
            self.grafo.adicionar_aresta(u, v, custo)
        
        # Executar Prim começando da central
        mst, custo_total = prim(self.grafo, self.central)
        
        # Analisar resultado
        ordem_conexao = self._determinar_ordem(mst)
        
        return {
            'conexoes': mst,
            'custo_total': custo_total,
            'ordem_implantacao': ordem_conexao,
            'fases': self._dividir_em_fases(ordem_conexao)
        }
    
    def _determinar_ordem(self, mst):
        """Determina ordem de implantação (BFS da central)."""
        # Implementação BFS a partir da central
        pass
    
    def _dividir_em_fases(self, ordem):
        """Divide implantação em fases temporais."""
        pass

# Exemplo
rede = RedeTelecomunicacoes(central=0)
rede.adicionar_local(0, "Central SP", "central")
rede.adicionar_local(1, "Subestação A", "subestacao")
rede.adicionar_local(2, "Terminal B", "terminal")
# ... adicionar mais locais

custos = {
    (0, 1): 100000,
    (0, 2): 150000,
    (1, 2): 80000,
    # ... mais conexões
}

projeto = rede.projetar_rede_incremental(custos)
print(f"Ordem de implantação: {projeto['ordem_implantacao']}")
```

### **7.2 🔋 Distribuição de Energia**

```python
class RedeEletrica:
    """
    Design de rede elétrica usando Prim.
    Natural começar da subestação principal.
    """
    
    def __init__(self, subestacao_principal: int):
        self.subestacao = subestacao_principal
        self.consumidores = {}
        self.capacidades = {}
    
    def adicionar_consumidor(self, id_cons: int, demanda_kw: float, 
                            localizacao: Tuple[float, float]):
        """Adiciona consumidor à rede."""
        self.consumidores[id_cons] = {
            'demanda': demanda_kw,
            'localizacao': localizacao
        }
    
    def calcular_custo_conexao(self, ponto1, ponto2):
        """
        Calcula custo de conexão baseado em:
        - Distância
        - Capacidade necessária
        - Tipo de terreno
        """
        # Implementação específica
        pass
    
    def projetar_rede_distribuicao(self):
        """
        Projeta rede de distribuição elétrica.
        Prim garante que a rede cresce da subestação,
        mantendo sempre uma árvore conectada (importante para fluxo elétrico).
        """
        # Montar grafo
        n = len(self.consumidores) + 1  # +1 para subestação
        grafo = GrafoPrim(n)
        
        # Adicionar conexões possíveis com custos
        for i in range(n):
            for j in range(i + 1, n):
                custo = self.calcular_custo_conexao(i, j)
                grafo.adicionar_aresta(i, j, custo)
        
        # Executar Prim da subestação
        mst, custo_total = prim(grafo, self.subestacao)
        
        # Validar capacidades (fluxo da subestação)
        if self._validar_capacidades(mst):
            return mst, custo_total
        else:
            return self._ajustar_para_capacidade(mst)
    
    def _validar_capacidades(self, mst):
        """Verifica se a rede suporta as demandas."""
        pass
    
    def _ajustar_para_capacidade(self, mst):
        """Ajusta rede para atender restrições de capacidade."""
        pass
```

### **7.3 🚰 Sistemas de Distribuição de Água**

```python
class RedeDistribuicaoAgua:
    """Projeto de rede de distribuição de água com Prim."""
    
    def __init__(self, reservatorio: int):
        self.reservatorio = reservatorio
        self.pontos_consumo = {}
    
    def adicionar_ponto_consumo(self, id_ponto: int, altitude: float, 
                                demanda_ls: float):
        """
        Adiciona ponto de consumo.
        Altitude é importante para pressão!
        """
        self.pontos_consumo[id_ponto] = {
            'altitude': altitude,
            'demanda': demanda_ls
        }
    
    def calcular_custo_tubulacao(self, p1, p2):
        """
        Custo de tubulação considerando:
        - Distância
        - Diferença de altitude (bombas necessárias)
        - Diâmetro necessário (demanda)
        """
        dist = self._distancia_euclidiana(p1, p2)
        dif_alt = abs(self.pontos_consumo[p1]['altitude'] - 
                     self.pontos_consumo[p2]['altitude'])
        
        # Custo base + custo de bombeamento
        custo_base = dist * 1000  # R$/metro
        custo_bomba = dif_alt * 500 if dif_alt > 10 else 0
        
        return custo_base + custo_bomba
    
    def projetar_rede(self):
        """
        Projeta rede de distribuição.
        Prim é ideal: a água flui do reservatório para fora.
        """
        n = len(self.pontos_consumo) + 1
        grafo = GrafoPrim(n)
        
        # Adicionar conexões
        for i in range(n):
            for j in range(i + 1, n):
                custo = self.calcular_custo_tubulacao(i, j)
                grafo.adicionar_aresta(i, j, custo)
        
        # Prim do reservatório
        mst, custo_total = prim(grafo, self.reservatorio)
        
        return {
            'tubulacoes': mst,
            'custo_total': custo_total,
            'analise_pressao': self._analisar_pressao(mst)
        }
    
    def _analisar_pressao(self, mst):
        """Analisa se a pressão é adequada em todos os pontos."""
        # Simular fluxo e pressão
        pass
```

---

## **8. Comparação: Prim vs Kruskal em Cenários Reais**

### **8.1 Benchmark Empírico**

```python
import time
import random
from typing import Callable

def benchmark_mst(num_vertices: int, densidade: float, 
                 num_testes: int = 10):
    """
    Compara Prim e Kruskal empiricamente.
    
    Args:
        num_vertices: número de vértices
        densidade: 0.0 a 1.0 (0.5 = 50% das arestas possíveis)
        num_testes: número de repetições
    """
    num_arestas = int(densidade * num_vertices * (num_vertices - 1) / 2)
    
    tempos_prim = []
    tempos_kruskal = []
    
    for _ in range(num_testes):
        # Gerar grafo aleatório
        grafo_prim = GrafoPrim(num_vertices)
        grafo_kruskal = Grafo(num_vertices)
        
        arestas_geradas = set()
        while len(arestas_geradas) < num_arestas:
            u = random.randint(0, num_vertices - 1)
            v = random.randint(0, num_vertices - 1)
            if u != v and (min(u,v), max(u,v)) not in arestas_geradas:
                peso = random.uniform(1, 100)
                grafo_prim.adicionar_aresta(u, v, peso)
                grafo_kruskal.adicionar_aresta(u, v, peso)
                arestas_geradas.add((min(u,v), max(u,v)))
        
        # Testar Prim
        start = time.time()
        mst_prim, _ = prim(grafo_prim)
        tempo_prim = time.time() - start
        tempos_prim.append(tempo_prim)
        
        # Testar Kruskal
        start = time.time()
        mst_kruskal, _ = kruskal(grafo_kruskal)
        tempo_kruskal = time.time() - start
        tempos_kruskal.append(tempo_kruskal)
    
    # Resultados
    media_prim = sum(tempos_prim) / num_testes
    media_kruskal = sum(tempos_kruskal) / num_testes
    
    print(f"\nBenchmark: V={num_vertices}, E={num_arestas}, "
          f"densidade={densidade:.2f}")
    print(f"Prim:    {media_prim*1000:.2f} ms (±{std(tempos_prim)*1000:.2f})")
    print(f"Kruskal: {media_kruskal*1000:.2f} ms (±{std(tempos_kruskal)*1000:.2f})")
    
    if media_prim < media_kruskal:
        print(f"→ Prim é {media_kruskal/media_prim:.2f}x mais rápido")
    else:
        print(f"→ Kruskal é {media_prim/media_kruskal:.2f}x mais rápido")

def std(values):
    """Desvio padrão simples."""
    media = sum(values) / len(values)
    return (sum((x - media)**2 for x in values) / len(values))**0.5

# Executar benchmarks
print("="*60)
print("COMPARAÇÃO EMPÍRICA: PRIM VS KRUSKAL")
print("="*60)

benchmark_mst(100, 0.1)   # Grafo esparso
benchmark_mst(100, 0.5)   # Grafo médio
benchmark_mst(100, 0.9)   # Grafo denso
```

### **8.2 Guia de Escolha**

```python
class EscolhedorMST:
    """Classe para ajudar na escolha entre Prim e Kruskal."""
    
    @staticmethod
    def recomendar(num_vertices: int, num_arestas: int, 
                   caracteristicas: dict) -> str:
        """
        Recomenda algoritmo MST baseado nas características do problema.
        
        Args:
            num_vertices: número de vértices
            num_arestas: número de arestas
            caracteristicas: dict com:
                - 'tipo_grafo': 'esparso', 'medio', 'denso'
                - 'tem_ponto_inicial_natural': bool
                - 'grafo_dinamico': bool
                - 'precisa_ordem_crescimento': bool
                - 'memoria_limitada': bool
        
        Returns:
            Recomendação com justificativa
        """
        densidade = num_arestas / (num_vertices * (num_vertices - 1) / 2)
        
        pontos_prim = 0
        pontos_kruskal = 0
        justificativas = []
        
        # Análise de densidade
        if densidade < 0.3:
            pontos_kruskal += 2
            justificativas.append("Grafo esparso favorece Kruskal")
        elif densidade > 0.6:
            pontos_prim += 2
            justificativas.append("Grafo denso favorece Prim")
        
        # Ponto inicial natural
        if caracteristicas.get('tem_ponto_inicial_natural'):
            pontos_prim += 1
            justificativas.append("Ponto inicial natural favorece Prim")
        
        # Ordem de crescimento
        if caracteristicas.get('precisa_ordem_crescimento'):
            pontos_prim += 2
            justificativas.append("Necessidade de ordem de crescimento favorece Prim")
        
        # Grafo dinâmico
        if caracteristicas.get('grafo_dinamico'):
            pontos_kruskal += 1
            justificativas.append("Grafo dinâmico é mais fácil com Kruskal")
        
        # Memória limitada
        if caracteristicas.get('memoria_limitada'):
            pontos_prim += 1
            justificativas.append("Memória limitada favorece Prim")
        
        # Decisão
        if pontos_prim > pontos_kruskal:
            algoritmo = "Prim"
        elif pontos_kruskal > pontos_prim:
            algoritmo = "Kruskal"
        else:
            algoritmo = "Ambos são equivalentes"
        
        resultado = f"\nRecomendação: {algoritmo}\n"
        resultado += f"Pontuação: Prim={pontos_prim}, Kruskal={pontos_kruskal}\n\n"
        resultado += "Justificativas:\n"
        for j in justificativas:
            resultado += f"  • {j}\n"
        
        return resultado

# Exemplo de uso
escolhedor = EscolhedorMST()

# Cenário 1: Rede de telecomunicações
print("Cenário: Rede de Telecomunicações")
print(escolhedor.recomendar(
    num_vertices=100,
    num_arestas=500,
    caracteristicas={
        'tem_ponto_inicial_natural': True,  # Central
        'precisa_ordem_crescimento': True,  # Implantação incremental
        'grafo_dinamico': False,
        'memoria_limitada': False
    }
))

# Cenário 2: Clustering de dados
print("\nCenário: Clustering de Dados")
print(escolhedor.recomendar(
    num_vertices=1000,
    num_arestas=5000,
    caracteristicas={
        'tem_ponto_inicial_natural': False,
        'precisa_ordem_crescimento': False,
        'grafo_dinamico': False,
        'memoria_limitada': True
    }
))
```

---

## **9. Exercícios Práticos**

### **9.1 🎯 Nível Básico**

#### **Exercício 1: Implementação Manual**
```python
"""
Implemente Prim usando apenas estruturas básicas (sem heapq).
Use lista simples para encontrar mínimo.
"""

def prim_basico(grafo_adj_list, inicio=0):
    """
    Args:
        grafo_adj_list: dict[int, list[(int, float)]]
                       vértice -> [(vizinho, peso), ...]
    """
    # Seu código aqui
    pass

# Teste
grafo = {
    0: [(1, 2), (2, 3)],
    1: [(0, 2), (2, 1)],
    2: [(0, 3), (1, 1)]
}
mst, peso = prim_basico(grafo)
```

#### **Exercício 2: Rastreamento de Iterações**
```python
"""
Modifique Prim para retornar informações de cada iteração.
"""

def prim_com_historico(grafo, inicio=0):
    """
    Returns:
        (mst, peso_total, historico)
        onde historico é lista de dict com:
        - iteracao: número
        - vertice_adicionado: int
        - aresta_adicionada: (u, v, peso) ou None
        - key_atualizado: dict[vertice, novo_key]
    """
    # Seu código aqui
    pass
```

### **9.2 🎯 Nível Intermediário**

#### **Exercício 3: Prim com Restrições de Grau**
```python
"""
Implemente Prim que limita o grau máximo de cada vértice.
Quando um vértice atinge grau máximo, não pode ter mais conexões.
"""

def prim_grau_limitado(grafo, inicio, grau_maximo):
    """
    Retorna MST (ou floresta) respeitando restrição de grau.
    Pode não conectar todos os vértices se grau é muito restrito.
    """
    # Seu código aqui
    pass
```

#### **Exercício 4: Prim Multi-início**
```python
"""
Execute Prim a partir de vários vértices iniciais e compare resultados.
Útil para entender que a MST é única (se pesos são únicos).
"""

def prim_multi_inicio(grafo):
    """
    Executa Prim de cada vértice e compara MSTs resultantes.
    
    Returns:
        dict com:
        - todas_mstsiguais: bool
        - peso_unico: float
        - diferentes_origens: list[int] (se houver diferenças)
    """
    # Seu código aqui
    pass
```

### **9.3 🎯 Nível Avançado**

#### **Exercício 5: Prim Online**
```python
"""
Implemente estrutura que mantém MST e a atualiza quando:
- Novo vértice é adicionado
- Novo aresta é adicionada
- Peso de aresta muda
"""

class PrimOnline:
    def __init__(self, num_vertices_inicial):
        # Seu código aqui
        pass
    
    def adicionar_vertice(self, conexoes: list[(int, float)]):
        """Adiciona vértice conectado a vértices existentes."""
        pass
    
    def adicionar_aresta(self, u, v, peso):
        """Adiciona aresta entre vértices existentes."""
        pass
    
    def atualizar_peso(self, u, v, novo_peso):
        """Atualiza peso de aresta existente."""
        pass
    
    def get_mst(self):
        """Retorna MST atual."""
        pass
```

#### **Exercício 6: Comparação Prim vs Dijkstra**
```python
"""
Implemente função que mostra semelhanças e diferenças
entre Prim e Dijkstra lado a lado.
"""

def comparar_prim_dijkstra(grafo, vertice_inicial):
    """
    Executa ambos e mostra:
    - Estruturas de dados usadas
    - Ordem de processamento de vértices
    - Como key[] evolui
    - Resultado final (MST vs caminhos mais curtos)
    """
    # Desafio: visualização comparativa
    pass
```

---

## **10. Recursos e Referências**

### **10.1 📚 Literatura Essencial**

1. **"Introduction to Algorithms" (CLRS)** - Capítulo 23
   - Análise comparativa Prim vs Kruskal
   - Implementações otimizadas
   - Provas formais

2. **"Algorithm Design" (Kleinberg & Tardos)**
   - Abordagem intuitiva
   - Muitos exemplos práticos

3. **"The Algorithm Design Manual" (Skiena)**
   - Implementações práticas
   - Casos de uso reais

### **10.2 🌐 Recursos Online**

**Visualizações:**
- VisuAlgo MST: https://visualgo.net/en/mst
- Algorithm Visualizer: Prim's Animation
- Graph Online: Interactive MST

**Tutoriais:**
- GeeksforGeeks: Prim's Algorithm
- CP-Algorithms: Minimum Spanning Tree - Prim
- Khan Academy: Prim's Algorithm

### **10.3 🛠️ Bibliotecas**

```python
# NetworkX - Prim
import networkx as nx
mst_prim = nx.minimum_spanning_tree(G, algorithm='prim')

# SciPy - MST genérico
from scipy.sparse.csgraph import minimum_spanning_tree
mst = minimum_spanning_tree(csr_matrix)

# graph-tool - Alta performance
import graph_tool.all as gt
tree_map = gt.min_spanning_tree(g, weights=edge_weights)
```

---

## **11. 🎯 Conclusão**

O Algoritmo de Prim exemplifica perfeitamente como uma estratégia gulosa pode ser tanto elegante quanto eficiente.

### **🔑 Principais Aprendizados**

1. **Crescimento Natural:** Árvore cresce organicamente de um ponto
2. **Semelhança com Dijkstra:** Estrutura muito similar, problemas diferentes
3. **Eficiência Adaptativa:** Performan depende da densidade do grafo
4. **Aplicações Práticas:** Natural para redes que crescem de um ponto central
5. **Garantia de Otimalidade:** Escolhas gulosas levam à solução ótima

### **💡 Prim vs Kruskal: Escolha Prática**

```
Escolha PRIM quando:
  ✓ Grafo é denso (muitas arestas)
  ✓ Tem ponto de partida natural (central, reservatório, etc.)
  ✓ Precisa da ordem de crescimento da árvore
  ✓ Implementação com matriz de adjacência

Escolha KRUSKAL quando:
  ✓ Grafo é esparso (poucas arestas)
  ✓ Não há ponto de partida natural
  ✓ Arestas já estão ordenadas
  ✓ Precisa de floresta geradora (grafo desconectado)

Ambos são equivalentes quando:
  ✓ Densidade média (30%-60%)
  ✓ Apenas MST final importa
  ✓ Performance não é crítica
```

### **🚀 Próximos Passos**

1. **Implemente** as duas versões (heap e array)
2. **Compare** empiricamente com Kruskal
3. **Entenda** relação com Dijkstra profundamente
4. **Aplique** em projeto real de sua área
5. **Explore** algoritmo de Borůvka (outro MST)

### **🌟 Reflexão Final**

Prim nos ensina que algoritmos podem resolver o mesmo problema de formas fundamentalmente diferentes. Enquanto Kruskal processa arestas globalmente, Prim cresce uma árvore localmente. Ambos chegam ao mesmo destino ótimo, mas o caminho importa quando consideramos aplicações práticas e características dos dados.

> *"Como uma árvore que cresce de uma semente, Prim constrói a solução ótima galho por galho."*

---

**Voltar para:** [Documentação de Algoritmos Gulosos](README.md) | [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

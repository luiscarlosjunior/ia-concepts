# Seleção de Atividades: O Exemplo Clássico de Algoritmo Guloso

O Problema de Seleção de Atividades é um exemplo clássico e didático de algoritmo guloso. Ele demonstra perfeitamente como uma estratégia gulosa simples pode levar à solução ótima. O problema consiste em selecionar o máximo número de atividades compatíveis (que não se sobrepõem no tempo) de um conjunto dado.

![Activity Selection Concept](../../images/activity_selection_concept.png)

---

## **1. O Problema de Seleção de Atividades**

### **1.1 Definição Formal**

**Entrada:**
- Conjunto de n atividades S = {a₁, a₂, ..., aₙ}
- Cada atividade aᵢ tem:
  - Tempo de início: sᵢ
  - Tempo de término: fᵢ

**Restrição:**
- Duas atividades são compatíveis se não se sobrepõem no tempo
- aᵢ e aⱼ são compatíveis se: fᵢ ≤ sⱼ ou fⱼ ≤ sᵢ

**Objetivo:**
- Selecionar o máximo número de atividades mutuamente compatíveis

### **1.2 Exemplo Prático**

**Cenário: Agendamento de Sala de Reuniões**

| Atividade | Início | Término | Duração |
|-----------|--------|---------|---------|
| a₁ | 9:00 | 10:00 | 1h |
| a₂ | 9:30 | 11:00 | 1.5h |
| a₃ | 10:00 | 11:30 | 1.5h |
| a₄ | 11:00 | 12:00 | 1h |
| a₅ | 11:30 | 13:00 | 1.5h |
| a₆ | 13:00 | 14:00 | 1h |

**Solução ótima:** {a₁, a₄, a₆} - 3 atividades
- a₁ (9:00-10:00) → a₄ (11:00-12:00) → a₆ (13:00-14:00)

### **1.3 Analogia Intuitiva**

Imagine que você está em uma conferência com várias palestras acontecendo simultaneamente:
- **Objetivo:** Assistir ao máximo de palestras possível
- **Restrição:** Você não pode estar em dois lugares ao mesmo tempo
- **Estratégia gulosa:** Sempre escolha a palestra que termina mais cedo

---

## **2. Algoritmo Guloso**

### **2.1 Estratégia Gulosa**

A escolha gulosa é:
> **"Sempre selecione a atividade compatível que termina mais cedo"**

**Por que isso funciona?**
1. Escolher a atividade que termina mais cedo libera o recurso o quanto antes
2. Isso maximiza o tempo disponível para atividades futuras
3. Deixa mais "espaço" para outras atividades

### **2.2 Algoritmo**

```
🚀 PRÉ-PROCESSAMENTO:
   └── Ordenar atividades por tempo de término crescente

🔄 ITERAÇÃO:
   ├── Inicializar: solução = {primeira atividade}
   ├── último_término = tempo de término da primeira atividade
   │
   └── Para cada atividade seguinte:
       ├── SE atividade.início ≥ último_término:
       │   ├── Adicionar atividade à solução
       │   └── último_término = atividade.término
       └── SENÃO:
           └── Descartar atividade (incompatível)

🏆 RETORNAR solução
```

### **2.3 Visualização do Processo**

**Linha do tempo:**
```
         a₁        a₃           a₅          
    |████████|           |████████████|    
9   10   11   12   13   14   15   16   17  
         |████████████|     |████████|     
             a₂                  a₄        
         
Ordenadas por término: a₁ < a₂ < a₃ < a₄ < a₅

Execução:
1. Seleciona a₁ (termina em 10) ✅
2. a₂ incompatível (começa em 9 < 10) ❌
3. a₃ compatível (começa em 12 ≥ 10) ✅
4. a₄ incompatível (começa em 14 < 15) ❌
5. a₅ compatível (começa em 16 ≥ 15) ✅

Solução: {a₁, a₃, a₅} = 3 atividades
```

---

## **3. Implementação**

### **3.1 Estrutura de Dados**

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Atividade:
    """Representa uma atividade com tempo de início e término."""
    id: str
    inicio: float
    termino: float
    descricao: str = ""
    
    def __repr__(self):
        return f"{self.id}({self.inicio:.1f}-{self.termino:.1f})"
    
    def compativel_com(self, outra: 'Atividade') -> bool:
        """Verifica se duas atividades são compatíveis."""
        return self.termino <= outra.inicio or outra.termino <= self.inicio
    
    def duracao(self) -> float:
        """Retorna a duração da atividade."""
        return self.termino - self.inicio


def selecao_atividades(atividades: List[Atividade]) -> List[Atividade]:
    """
    Algoritmo guloso de seleção de atividades.
    
    Args:
        atividades: Lista de atividades
    
    Returns:
        Lista com máximo de atividades compatíveis
    
    Complexidade: O(n log n) devido à ordenação
    """
    if not atividades:
        return []
    
    # Ordenar por tempo de término
    atividades_ordenadas = sorted(atividades, key=lambda a: a.termino)
    
    # Inicializar solução com primeira atividade
    solucao = [atividades_ordenadas[0]]
    ultimo_termino = atividades_ordenadas[0].termino
    
    # Processar atividades restantes
    for atividade in atividades_ordenadas[1:]:
        # Se compatível, adicionar à solução
        if atividade.inicio >= ultimo_termino:
            solucao.append(atividade)
            ultimo_termino = atividade.termino
    
    return solucao


def selecao_atividades_verboso(atividades: List[Atividade]) -> List[Atividade]:
    """Versão verbosa para fins educacionais."""
    print("=" * 70)
    print("ALGORITMO DE SELEÇÃO DE ATIVIDADES")
    print("=" * 70)
    
    if not atividades:
        print("Nenhuma atividade fornecida!")
        return []
    
    # Mostrar atividades originais
    print("\\nAtividades originais:")
    for a in atividades:
        print(f"  {a.id}: {a.inicio:.1f} → {a.termino:.1f} (duração: {a.duracao():.1f})")
    
    # Ordenar
    atividades_ordenadas = sorted(atividades, key=lambda a: a.termino)
    
    print("\\nApós ordenação por tempo de término:")
    for a in atividades_ordenadas:
        print(f"  {a.id}: {a.inicio:.1f} → {a.termino:.1f}")
    
    # Processar
    print("\\nProcessamento:")
    print("-" * 70)
    
    solucao = [atividades_ordenadas[0]]
    ultimo_termino = atividades_ordenadas[0].termino
    
    print(f"✅ Iteração 1: Selecionar {atividades_ordenadas[0].id} "
          f"(primeira atividade)")
    print(f"   Último término: {ultimo_termino:.1f}")
    
    for i, atividade in enumerate(atividades_ordenadas[1:], 2):
        compativel = atividade.inicio >= ultimo_termino
        
        if compativel:
            solucao.append(atividade)
            print(f"✅ Iteração {i}: Selecionar {atividade.id}")
            print(f"   {atividade.inicio:.1f} ≥ {ultimo_termino:.1f} → Compatível!")
            ultimo_termino = atividade.termino
            print(f"   Último término atualizado: {ultimo_termino:.1f}")
        else:
            print(f"❌ Iteração {i}: Rejeitar {atividade.id}")
            print(f"   {atividade.inicio:.1f} < {ultimo_termino:.1f} → Incompatível!")
    
    print("=" * 70)
    print(f"\\n🏆 SOLUÇÃO ÓTIMA: {len(solucao)} atividades selecionadas")
    print("=" * 70)
    for a in solucao:
        print(f"  {a.id}: {a.inicio:.1f} → {a.termino:.1f}")
    
    return solucao


# Exemplo de uso
if __name__ == "__main__":
    # Criar atividades do exemplo
    atividades = [
        Atividade("a1", 9.0, 10.0, "Reunião A"),
        Atividade("a2", 9.5, 11.0, "Apresentação"),
        Atividade("a3", 10.0, 11.5, "Workshop"),
        Atividade("a4", 11.0, 12.0, "Reunião B"),
        Atividade("a5", 11.5, 13.0, "Treinamento"),
        Atividade("a6", 13.0, 14.0, "Reunião C"),
    ]
    
    print("Versão simples:")
    print("-" * 40)
    resultado = selecao_atividades(atividades)
    print(f"Atividades selecionadas: {resultado}")
    print(f"Total: {len(resultado)} atividades")
    
    print("\\n" + "=" * 80)
    print("Versão detalhada:")
    print("=" * 80)
    resultado_verboso = selecao_atividades_verboso(atividades)
```

**Saída esperada:**
```
Versão simples:
----------------------------------------
Atividades selecionadas: [a1(9.0-10.0), a4(11.0-12.0), a6(13.0-14.0)]
Total: 3 atividades
```

---

## **4. Análise de Complexidade**

### **4.1 Complexidade de Tempo**

```
FASE 1: Ordenação por tempo de término
   └── O(n log n)

FASE 2: Loop através das atividades
   ├── n iterações
   └── O(1) por iteração
   └── Total: O(n)

COMPLEXIDADE TOTAL: O(n log n)
```

**Dominada pela ordenação!**

### **4.2 Complexidade de Espaço**

```
💾 MEMÓRIA:
   ├── Lista original: O(n)
   ├── Lista ordenada: O(n) (se criar cópia) ou O(1) (se ordenar in-place)
   ├── Lista de solução: O(n) no pior caso
   └── Total: O(n)
```

### **4.3 Otimizações Possíveis**

```python
def selecao_atividades_otimizado(atividades: List[Atividade]) -> int:
    """
    Versão otimizada que retorna apenas a contagem.
    Economiza memória não armazenando a solução.
    """
    if not atividades:
        return 0
    
    # Ordenar
    atividades.sort(key=lambda a: a.termino)
    
    # Contar
    contador = 1
    ultimo_termino = atividades[0].termino
    
    for atividade in atividades[1:]:
        if atividade.inicio >= ultimo_termino:
            contador += 1
            ultimo_termino = atividade.termino
    
    return contador
```

---

## **5. Prova de Correção**

### **5.1 Teorema: Seleção Gulosa é Ótima**

**Teorema:** O algoritmo guloso de seleção de atividades produz uma solução ótima.

**Prova (por indução):**

**Definições:**
- Seja A = {a₁, a₂, ..., aₙ} ordenado por tempo de término
- Seja G = solução gulosa
- Seja O = uma solução ótima qualquer

**Lema (Greedy Choice):**
> Existe uma solução ótima que contém a₁ (atividade que termina mais cedo)

**Prova do Lema:**
1. Se a₁ ∈ O, pronto! ✅
2. Se a₁ ∉ O, seja aₖ a primeira atividade em O
3. Como a₁ termina antes de aₖ, podemos substituir aₖ por a₁
4. A nova solução O' = (O - {aₖ}) ∪ {a₁} ainda é ótima
5. Logo, existe solução ótima contendo a₁ ✅

**Prova Principal (por indução):**

**Base (n=1):** Trivial - selecionar a única atividade é ótimo ✅

**Passo Indutivo:**
1. Guloso seleciona a₁
2. Por lema, existe solução ótima O contendo a₁
3. Remover a₁ deixa subproblema com atividades compatíveis com a₁
4. Guloso é ótimo para este subproblema (por indução)
5. Logo, guloso é ótimo para problema original ✅

### **5.2 Propriedade de Subestrutura Ótima**

**Propriedade:** Se removemos a primeira atividade escolhida pelo algoritmo guloso, o problema restante tem a mesma estrutura.

**Formal:**
```
Seja Sₖ = subconjunto de atividades que começam após aₖ terminar
Se escolhemos aₖ, a solução ótima para S é:
   {aₖ} ∪ (solução ótima para Sₖ)
```

Isso justifica a abordagem recursiva e a programação dinâmica (se necessário).

---

## **6. Variações do Problema**

### **6.1 🎯 Seleção com Pesos**

```python
@dataclass
class AtividadePonderada:
    """Atividade com valor/peso associado."""
    id: str
    inicio: float
    termino: float
    valor: float
    
    def __repr__(self):
        return f"{self.id}({self.inicio:.1f}-{self.termino:.1f}, v={self.valor})"


def selecao_atividades_ponderada_dp(atividades: List[AtividadePonderada]) -> List[AtividadePonderada]:
    """
    Seleção de atividades ponderadas.
    NOTA: Algoritmo guloso NÃO funciona aqui!
    Precisa de programação dinâmica.
    
    Complexidade: O(n²) ou O(n log n) com busca binária
    """
    if not atividades:
        return []
    
    # Ordenar por tempo de término
    atividades = sorted(atividades, key=lambda a: a.termino)
    n = len(atividades)
    
    # DP: dp[i] = valor máximo considerando atividades 0..i
    dp = [0] * n
    escolha = [None] * n
    
    # Base
    dp[0] = atividades[0].valor
    escolha[0] = []
    
    for i in range(1, n):
        # Opção 1: Não incluir atividade i
        valor_sem = dp[i-1]
        
        # Opção 2: Incluir atividade i
        # Encontrar última atividade compatível
        j = i - 1
        while j >= 0 and atividades[j].termino > atividades[i].inicio:
            j -= 1
        
        valor_com = atividades[i].valor
        if j >= 0:
            valor_com += dp[j]
        
        # Escolher melhor opção
        if valor_com > valor_sem:
            dp[i] = valor_com
            escolha[i] = j
        else:
            dp[i] = valor_sem
            escolha[i] = None
    
    # Reconstruir solução
    solucao = []
    i = n - 1
    while i >= 0:
        if escolha[i] is not None:
            solucao.append(atividades[i])
            i = escolha[i]
        else:
            i -= 1
    
    solucao.reverse()
    return solucao
```

### **6.2 📅 Múltiplas Salas**

```python
def selecao_atividades_multiplas_salas(atividades: List[Atividade]) -> dict:
    """
    Alocar atividades em múltiplas salas.
    Objetivo: Minimizar número de salas necessárias.
    
    Estratégia: Algoritmo guloso baseado em início das atividades
    """
    if not atividades:
        return {}
    
    # Ordenar por início
    atividades = sorted(atividades, key=lambda a: a.inicio)
    
    # Heap de salas: (tempo_livre, id_sala, atividades)
    import heapq
    salas = []  # Min-heap por tempo_livre
    proximo_id_sala = 0
    alocacao = {}
    
    for atividade in atividades:
        # Verificar se alguma sala está livre
        if salas and salas[0][0] <= atividade.inicio:
            # Reusar sala
            tempo_livre, id_sala, ativs = heapq.heappop(salas)
            ativs.append(atividade)
            heapq.heappush(salas, (atividade.termino, id_sala, ativs))
        else:
            # Criar nova sala
            id_sala = f"Sala_{proximo_id_sala}"
            proximo_id_sala += 1
            heapq.heappush(salas, (atividade.termino, id_sala, [atividade]))
    
    # Formatar resultado
    resultado = {}
    for _, id_sala, ativs in salas:
        resultado[id_sala] = ativs
    
    return resultado

# Exemplo
atividades_sobrepostas = [
    Atividade("a1", 9.0, 10.0),
    Atividade("a2", 9.5, 11.0),
    Atividade("a3", 10.5, 12.0),
    Atividade("a4", 11.0, 12.5),
]

alocacao = selecao_atividades_multiplas_salas(atividades_sobrepostas)
print(f"\\nNúmero de salas necessárias: {len(alocacao)}")
for sala, ativs in alocacao.items():
    print(f"{sala}: {ativs}")
```

### **6.3 ⏰ Intervalo de Tempo Limitado**

```python
def selecao_atividades_janela_tempo(atividades: List[Atividade], 
                                     inicio_janela: float, 
                                     fim_janela: float) -> List[Atividade]:
    """
    Selecionar atividades dentro de uma janela de tempo específica.
    
    Exemplo: Maximizar atividades entre 9h e 17h
    """
    # Filtrar atividades que cabem na janela
    atividades_validas = [
        a for a in atividades 
        if a.inicio >= inicio_janela and a.termino <= fim_janela
    ]
    
    # Aplicar algoritmo guloso normal
    return selecao_atividades(atividades_validas)
```

---

## **7. Aplicações Práticas**

### **7.1 📊 Agendamento de CPU**

```python
class Processo:
    """Representa um processo/tarefa."""
    def __init__(self, pid: int, tempo_chegada: float, 
                 tempo_execucao: float, prioridade: int = 0):
        self.pid = pid
        self.tempo_chegada = tempo_chegada
        self.tempo_inicio_exec = None
        self.tempo_fim_exec = None
        self.tempo_execucao = tempo_execucao
        self.prioridade = prioridade
    
    def to_atividade(self) -> Atividade:
        """Converte para atividade."""
        return Atividade(
            id=f"P{self.pid}",
            inicio=self.tempo_chegada,
            termino=self.tempo_chegada + self.tempo_execucao
        )


class EscalonadorSJF:
    """
    Shortest Job First (SJF) Scheduling.
    Caso especial de seleção de atividades.
    """
    
    def __init__(self):
        self.processos = []
        self.tempo_atual = 0
    
    def adicionar_processo(self, processo: Processo):
        self.processos.append(processo)
    
    def escalonar(self) -> List[Processo]:
        """
        Escalona processos usando SJF (não-preemptivo).
        Minimiza tempo médio de espera.
        """
        # Ordenar por tempo de execução (guloso!)
        processos_ordenados = sorted(self.processos, 
                                    key=lambda p: p.tempo_execucao)
        
        escalonamento = []
        tempo_atual = 0
        
        for processo in processos_ordenados:
            processo.tempo_inicio_exec = max(tempo_atual, processo.tempo_chegada)
            processo.tempo_fim_exec = processo.tempo_inicio_exec + processo.tempo_execucao
            tempo_atual = processo.tempo_fim_exec
            escalonamento.append(processo)
        
        return escalonamento
    
    def calcular_metricas(self, escalonamento: List[Processo]) -> dict:
        """Calcula métricas de desempenho."""
        tempos_espera = []
        tempos_retorno = []
        
        for p in escalonamento:
            tempo_espera = p.tempo_inicio_exec - p.tempo_chegada
            tempo_retorno = p.tempo_fim_exec - p.tempo_chegada
            tempos_espera.append(tempo_espera)
            tempos_retorno.append(tempo_retorno)
        
        return {
            'tempo_espera_medio': sum(tempos_espera) / len(tempos_espera),
            'tempo_retorno_medio': sum(tempos_retorno) / len(tempos_retorno)
        }
```

### **7.2 🏭 Agendamento de Produção**

```python
class TarefaProducao:
    """Tarefa de produção em uma máquina."""
    def __init__(self, id_tarefa: str, tempo_setup: float, 
                 tempo_producao: float, prazo: float):
        self.id = id_tarefa
        self.tempo_setup = tempo_setup
        self.tempo_producao = tempo_producao
        self.prazo = prazo
        self.tempo_total = tempo_setup + tempo_producao
    
    def to_atividade(self, inicio: float) -> Atividade:
        return Atividade(
            self.id,
            inicio,
            inicio + self.tempo_total
        )


def otimizar_sequencia_producao(tarefas: List[TarefaProducao]) -> List[TarefaProducao]:
    """
    Otimiza sequência de produção.
    Estratégia: Minimizar número de tarefas atrasadas.
    
    Abordagem gulosa: Ordenar por prazo (EDD - Earliest Due Date)
    """
    # Ordenar por prazo
    tarefas_ordenadas = sorted(tarefas, key=lambda t: t.prazo)
    
    tempo_atual = 0
    sequencia = []
    tarefas_atrasadas = []
    
    for tarefa in tarefas_ordenadas:
        tempo_conclusao = tempo_atual + tarefa.tempo_total
        
        if tempo_conclusao <= tarefa.prazo:
            # Tarefa será concluída no prazo
            sequencia.append(tarefa)
            tempo_atual = tempo_conclusao
        else:
            # Tarefa ficará atrasada
            tarefas_atrasadas.append(tarefa)
    
    return {
        'sequencia_otima': sequencia,
        'tarefas_atrasadas': tarefas_atrasadas,
        'numero_no_prazo': len(sequencia),
        'numero_atrasadas': len(tarefas_atrasadas)
    }
```

### **7.3 📺 Programação de TV**

```python
class Programa:
    """Programa de TV."""
    def __init__(self, nome: str, duracao_minutos: int, 
                 horario_inicio: str, audiencia_esperada: int):
        self.nome = nome
        self.duracao = duracao_minutos
        self.horario = horario_inicio
        self.audiencia = audiencia_esperada
    
    def to_atividade(self) -> Atividade:
        # Converter horário para minutos desde meia-noite
        h, m = map(int, self.horario.split(':'))
        inicio_min = h * 60 + m
        fim_min = inicio_min + self.duracao
        
        return Atividade(
            self.nome,
            inicio_min,
            fim_min,
            f"{self.horario} ({self.audiencia} viewers)"
        )


def montar_grade_programacao(programas: List[Programa]) -> List[Programa]:
    """
    Monta grade de programação maximizando número de programas.
    """
    # Converter para atividades
    atividades = [p.to_atividade() for p in programas]
    
    # Aplicar seleção de atividades
    selecionadas = selecao_atividades(atividades)
    
    # Converter de volta para programas
    nomes_selecionados = {a.id for a in selecionadas}
    return [p for p in programas if p.nome in nomes_selecionados]
```

---

## **8. Exercícios Práticos**

### **8.1 🎯 Nível Básico**

#### **Exercício 1: Implementação Recursiva**
```python
"""
Implemente versão recursiva do algoritmo de seleção de atividades.
"""

def selecao_atividades_recursivo(atividades: List[Atividade], 
                                  indice: int = 0, 
                                  ultimo_termino: float = 0) -> List[Atividade]:
    """
    Versão recursiva.
    Assume que atividades já estão ordenadas por término.
    """
    # Seu código aqui
    pass
```

#### **Exercício 2: Visualização Gráfica**
```python
"""
Crie visualização da solução usando matplotlib.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualizar_solucao(atividades: List[Atividade], 
                      solucao: List[Atividade]):
    """Cria gráfico de Gantt das atividades."""
    # Seu código aqui
    # Dica: Use plt.barh() para barras horizontais
    pass
```

### **8.2 🎯 Nível Intermediário**

#### **Exercício 3: Todas as Soluções Ótimas**
```python
"""
Encontre TODAS as soluções ótimas possíveis.
(Pode haver múltiplas soluções com mesmo número de atividades)
"""

def todas_solucoes_otimas(atividades: List[Atividade]) -> List[List[Atividade]]:
    """
    Retorna todas as combinações ótimas.
    Dica: Use backtracking
    """
    # Seu código aqui
    pass
```

#### **Exercício 4: Análise de Diferentes Heurísticas**
```python
"""
Compare diferentes estratégias gulosas:
1. Menor tempo de término (ótimo)
2. Menor duração
3. Menor tempo de início
4. Maior folga (prazo - duração)
"""

def comparar_heuristicas(atividades: List[Atividade]):
    """
    Testa diferentes heurísticas e compara resultados.
    """
    heuristicas = {
        'termino_minimo': lambda a: a.termino,
        'duracao_minima': lambda a: a.duracao(),
        'inicio_minimo': lambda a: a.inicio,
        # ... adicionar mais
    }
    
    # Seu código aqui
    pass
```

### **8.3 🎯 Nível Avançado**

#### **Exercício 5: Sistema de Agendamento Completo**
```python
"""
Implemente sistema completo de agendamento de recursos com:
- Múltiplos tipos de recursos
- Prioridades
- Restrições adicionais
- Interface web simples
"""

class SistemaAgendamento:
    def __init__(self):
        # Seu código aqui
        pass
    
    def adicionar_atividade(self, atividade, prioridade, recursos):
        # Seu código aqui
        pass
    
    def otimizar_agenda(self):
        # Aplicar seleção de atividades com restrições
        pass
    
    def gerar_relatorio(self):
        # Criar relatório detalhado
        pass
```

---

## **9. Comparação: Guloso vs Programação Dinâmica**

### **9.1 Quando Guloso Funciona**

**Seleção de Atividades SEM Pesos:**
```python
# Algoritmo guloso: O(n log n)
def guloso_simples(atividades):
    atividades.sort(key=lambda a: a.termino)
    # ... seleção gulosa
    return solucao
```

**✅ Vantagens:**
- Simples de implementar
- Rápido: O(n log n)
- Usa pouca memória: O(n)

### **9.2 Quando Precisa de DP**

**Seleção de Atividades COM Pesos:**
```python
# Programação dinâmica: O(n²) ou O(n log n)
def dp_ponderado(atividades):
    # Precisa considerar todas as combinações
    # ... algoritmo DP
    return solucao
```

**✅ Vantagens:**
- Garante otimalidade com pesos
- Mais flexível para restrições

**❌ Desvantagens:**
- Mais complexo
- Mais lento
- Usa mais memória

### **9.3 Exemplo Comparativo**

```python
# Atividades com pesos
atividades = [
    AtividadePonderada("a1", 0, 3, valor=5),
    AtividadePonderada("a2", 1, 4, valor=6),
    AtividadePonderada("a3", 3, 6, valor=5),
    AtividadePonderada("a4", 5, 7, valor=4),
]

# Guloso escolhe por término: {a1, a3} = valor 10
# DP encontra ótimo: {a2, a4} = valor 10 (empate)
# Ou {a1, a4} = valor 9

# Mas se mudarmos valores:
# a2.valor = 10
# Guloso ainda escolhe {a1, a3} = 10
# DP corretamente escolhe {a2, a4} = 14 ✅
```

---

## **10. Recursos e Referências**

### **10.1 📚 Literatura Clássica**

1. **"Introduction to Algorithms" (CLRS)** - Capítulo 16.1
   - Apresentação canônica do problema
   - Prova detalhada de correção

2. **"Algorithm Design" (Kleinberg & Tardos)** - Capítulo 4.1
   - Exemplo introdutório perfeito
   - Múltiplas variações

3. **"The Algorithm Design Manual" (Skiena)**
   - Aplicações práticas
   - Problemas relacionados

### **10.2 🌐 Recursos Online**

**Visualizações:**
- Algorithm Visualizer: Activity Selection
- VisuAlgo: Greedy Algorithms

**Tutoriais:**
- GeeksforGeeks: Activity Selection Problem
- CP-Algorithms: Activity Selection
- LeetCode: Non-overlapping Intervals

### **10.3 🎓 Problemas Relacionados**

**Problemas de Programação Competitiva:**
1. **Interval Scheduling** - LeetCode 435
2. **Meeting Rooms II** - LeetCode 253
3. **Minimum Number of Arrows** - LeetCode 452
4. **Non-overlapping Intervals** - LeetCode 435

---

## **11. 🎯 Conclusão**

O Problema de Seleção de Atividades é o exemplo didático perfeito de algoritmo guloso.

### **🔑 Principais Aprendizados**

1. **Simplicidade Poderosa:** Estratégia simples leva a solução ótima
2. **Prova de Correção:** Exemplo claro de como provar que guloso funciona
3. **Subestrutura Ótima:** Demonstra propriedade fundamental
4. **Aplicabilidade:** Modelo para inúmeros problemas reais
5. **Limitações:** Mostra quando guloso não funciona (com pesos)

### **💡 Quando Usar Seleção de Atividades**

| **✅ Use quando:** | **❌ Evite quando:** |
|-------------------|---------------------|
| Maximizar número de atividades | Maximizar valor total |
| Atividades têm prioridades iguais | Atividades têm pesos diferentes |
| Recurso único a ser alocado | Múltiplos recursos interdependentes |
| Solução rápida necessária | Ótimo absoluto com complexidade extra vale a pena |

### **🚀 Próximos Passos**

1. **Implemente** todas as variações apresentadas
2. **Resolva** problemas de programação competitiva
3. **Compare** com programação dinâmica no caso ponderado
4. **Aplique** em projetos reais de agendamento
5. **Estude** problemas relacionados (Job Scheduling, Interval Partitioning)

### **🌟 Reflexão Final**

Seleção de Atividades nos ensina uma lição fundamental sobre algoritmos gulosos: a estratégia "sempre terminar mais cedo" é contra-intuitiva (poderíamos pensar em escolher atividades mais curtas), mas é provadamente ótima. Isso demonstra a importância de análise matemática rigorosa em design de algoritmos.

> *"Na vida e nos algoritmos, terminar cedo pode ser a chave para fazer mais!"*

---

**Voltar para:** [Documentação de Algoritmos Gulosos](README.md) | [Documentação de Algoritmos](../README.md) | [Documentação Principal](../../README.md)

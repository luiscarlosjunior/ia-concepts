import math

class CalcDistancias():
    """
    O cálculo de distância é uma ferramenta essencial em Inteligência Artificial (IA) e otimização. 
    Em IA, é frequentemente utilizado em algoritmos de aprendizado de máquina, 
    como K-Nearest Neighbors (KNN), onde a distância entre pontos de dados determina a 
    classificação ou regressão. 
    Em otimização, a distância é usada para medir a similaridade ou dissimilaridade entre soluções, 
    ajudando a encontrar a solução ótima em problemas como o de roteamento de veículos ou o de 
    alocação de recursos.
    """
    
    def euclidean_distance(ponto1, ponto2):
        """
        A distância Euclidiana é uma medida de distância direta entre dois pontos em um 
        espaço Euclidiano. É frequentemente usada em várias áreas da ciência e 
        engenharia devido à sua simplicidade e intuição geométrica. 
        Uma explicação sobre onde e por que essa métrica é usada, além de listar 
        alguns problemas potenciais.

        Explicação e Uso da Distância Euclidiana
        
        A distância Euclidiana é calculada usando a fórmula: 
            [ \text{distância} = \sqrt{(x2 - x1)^2 + (y2 - y1)^2} ]
        
        Onde é Usada:
            -Ciência de Dados e Machine Learning: Para calcular a similaridade entre pontos de dados, 
            como em algoritmos de clustering (ex: K-means) e classificação (ex: K-Nearest Neighbors).
            
            -Computação Gráfica: Para determinar a proximidade entre objetos ou pontos em uma cena.
            
            -Robótica e Navegação: Para calcular a distância entre a posição atual de um robô e um 
            ponto de destino.
            
            -Geometria Computacional: Em problemas que envolvem a análise de formas e tamanhos de objetos.
        
        Por Que é Usada:
            -Simplicidade: A fórmula é simples e direta.
            -Intuitiva: Facilmente compreensível e visualizável em um espaço 2D ou 3D.
            -Eficiência Computacional: Relativamente barata em termos de cálculo computacional.
        
        Problemas Potenciais:
            -Sensibilidade a Escalas: A distância Euclidiana pode ser distorcida se os dados 
            tiverem diferentes escalas. Por exemplo, se uma dimensão tem valores muito 
            maiores que outra, ela dominará a distância.
            -Não Considera Correlações: Não leva em conta a correlação entre as dimensões dos dados.
            -Alta Dimensionalidade: Em espaços de alta dimensionalidade, a distância Euclidiana 
            pode se tornar menos discriminativa, um fenômeno conhecido como a "maldição da dimensionalidade".
        
        Args:
        ponto1 (tuple): O primeiro ponto como uma tupla (x1, y1).
        ponto2 (tuple): O segundo ponto como uma tupla (x2, y2).
        
        Retorna:
        float: A distância Euclidiana entre os dois pontos.
        """
        return math.sqrt((ponto1[0] - ponto2[0])**2 + (ponto1[1] - ponto2[1])**2)

    def manhattan_distance(ponto1, ponto2):
        """
        A distância Manhattan, também conhecida como distância L1 ou distância de bloco, 
        é uma métrica usada para calcular a distância entre dois pontos em um espaço 2D. 
        Ela é chamada assim porque reflete a maneira como se desloca em uma grade de ruas, 
        como as de Manhattan, onde só se pode mover horizontalmente ou verticalmente.
        
        Fórmula
            A fórmula para calcular a distância Manhattan entre dois pontos 
            ( (x_1, y_1) ) e ( (x_2, y_2) ) é: [ \text{Distância Manhattan} = |x_1 - x_2| + |y_1 - y_2| ]

        Uso
        A distância Manhattan é usada em várias áreas, incluindo:

            -Ciência da Computação: Em algoritmos de busca e otimização, 
            como o algoritmo A* para encontrar o caminho mais curto em um grid.
            
            -Visão Computacional: Para comparar imagens ou padrões onde a 
            diferença absoluta entre pixels é mais relevante.
            
            -Economia e Logística: Para calcular distâncias em cidades com layouts de grade.
            
        Problemas:
        
        Não é adequada para distâncias Euclidianas: Em espaços onde o movimento diagonal é 
        permitido ou relevante, a distância Euclidianas (L2) pode ser mais apropriada.
        Sensível a rotações: A distância Manhattan pode variar significativamente com a 
        rotação do sistema de coordenadas.
        Não considera obstáculos: Em ambientes com obstáculos, a distância real pode ser 
        maior do que a distância Manhattan calculada.
        
        Args:
        ponto1 (tuple): O primeiro ponto como uma tupla (x1, y1).
        ponto2 (tuple): O segundo ponto como uma tupla (x2, y2).
        
        Retorna:
        float: A distância Manhattan entre os dois pontos.
        """
        return abs(ponto1[0] - ponto2[0]) + abs(ponto1[1] - ponto2[1])

    def chebyshev_distance(ponto1, ponto2):
        """
        A distância Chebyshev é uma métrica utilizada para calcular a distância entre dois pontos 
        em um espaço 2D, onde a distância é definida como o maior valor absoluto das diferenças 
        entre as coordenadas correspondentes dos pontos. 
        Essa métrica é frequentemente usada em contextos onde o movimento é permitido 
        em qualquer direção, mas o custo é determinado pela maior diferença em qualquer dimensão.

        *Onde é usada a distância Chebyshev*:
            -Jogos de tabuleiro: Em jogos como xadrez, onde a movimentação das peças pode ser feita 
            em qualquer direção (horizontal, vertical ou diagonal), a distância Chebyshev é usada 
            para calcular o número mínimo de movimentos necessários para uma peça alcançar outra posição.
        
            - Robótica: Em ambientes onde um robô pode se mover em qualquer direção, 
            a distância Chebyshev pode ser usada para planejar caminhos e evitar obstáculos.
            
            - Processamento de imagens: Em algumas técnicas de processamento de imagens, 
            a distância Chebyshev pode ser usada para medir a similaridade entre pixels ou regiões.
            Problemas com a distância Chebyshev:
        
        *Não é adequada para todos os contextos*: Em situações onde o movimento é restrito 
        a direções específicas (por exemplo, apenas horizontal e vertical), 
        a distância Chebyshev pode não ser a métrica mais apropriada.
        
            -Sensibilidade a ruídos: Em dados com ruídos ou outliers, 
            a distância Chebyshev pode ser menos robusta, pois considera apenas a maior 
            diferença em uma dimensão, ignorando as outras.
        
            -Não reflete a distância real: Em alguns casos, a distância Chebyshev pode não refletir 
            a distância real percorrida, especialmente em espaços onde o movimento não é 
            uniforme em todas as direções.
        
        Args:
        ponto1 (tuple): O primeiro ponto como uma tupla (x1, y1).
        ponto2 (tuple): O segundo ponto como uma tupla (x2, y2).
        
        Retorna:
        float: A distância Chebyshev entre os dois pontos.
        """
        return max(abs(ponto1[0] - ponto2[0]), abs(ponto1[1] - ponto2[1]))

    def minkowski_distance(ponto1, ponto2, p):
        """
        A distância de Minkowski é uma métrica usada para medir a distância 
        entre dois pontos em um espaço n-dimensional. 
        É uma generalização das distâncias Euclidiana e Manhattan. 
        
        A fórmula da distância de Minkowski é dada por:
        [ D(p) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}} ]

        Onde:
        ( x ) e ( y ) são os pontos no espaço.
        ( p ) é um parâmetro que determina o tipo de distância.

        Onde é usada a distância de Minkowski
            -Machine Learning: Usada em algoritmos de classificação e regressão, como K-Nearest Neighbors (KNN).
            
            -Análise de Dados: Para medir similaridades ou dissimilaridades entre conjuntos de dados.
            
            -Visão Computacional: Para comparar características de imagens.

        Por que usar a distância de Minkowski
            Flexibilidade: Ajustando o parâmetro ( p ), pode-se obter diferentes tipos de distâncias:
            ( p = 1 ): Distância Manhattan.
            ( p = 2 ): Distância Euclidiana.
            ( p \to \infty ): Distância Chebyshev.
            
        Aplicabilidade: Pode ser usada em diferentes contextos e tipos de dados.

        Problemas e Considerações
            -Escolha do Parâmetro ( p ): A escolha do valor de ( p ) pode afetar significativamente os 
            resultados. Não há um valor universalmente ótimo.
            Dimensionalidade: Em espaços de alta dimensionalidade, a distinção entre distâncias 
            pode se tornar menos significativa (maldição da dimensionalidade).
            Cálculo: Para grandes conjuntos de dados, o cálculo pode ser computacionalmente intensivo.
        
        Args:
        ponto1 (tuple): O primeiro ponto como uma tupla (x1, y1).
        ponto2 (tuple): O segundo ponto como uma tupla (x2, y2).
        p (int): A ordem da distância Minkowski.
        
        Retorna:
        float: A distância Minkowski entre os dois pontos.
        """
        return (abs(ponto1[0] - ponto2[0])**p + abs(ponto1[1] - ponto2[1])**p)**(1/p)
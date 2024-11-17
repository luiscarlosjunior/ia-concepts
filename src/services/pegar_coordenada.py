import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class PegarCoordenadaJanela:
    def __init__(self):        
        self.coordinates = []  # Lista para armazenar as coordenadas clicadas
        self.vertical_line = None  # Linha vertical para seguir o cursor
        self.horizontal_line = None  # Linha horizontal para seguir o cursor

    def on_click(self, event):
        """Função chamada ao clicar na janela."""
        if event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            self.coordinates.append((x, y))
            print(f"Coordenada clicada: ({x:.2f}, {y:.2f})")
            
            # Plota o ponto e adiciona o número da sequência
            ax.scatter(x, y, color='red')
            ax.text(x, y, f'{len(self.coordinates)}', fontsize=8, color='blue')
            plt.draw()

    def on_motion(self, event):
        """Função chamada ao mover o mouse para desenhar linhas de referência."""
        # Verifica e remove a linha vertical, se ela existir
        if self.vertical_line is not None:
            try:
                self.vertical_line.remove()
            except ValueError:
                pass  # A linha já foi removida

        # Verifica e remove a linha horizontal, se ela existir
        if self.horizontal_line is not None:
            try:
                self.horizontal_line.remove()
            except ValueError:
                pass  # A linha já foi removida

        # Adiciona novas linhas de referência, se o mouse estiver dentro dos limites do gráfico
        if event.xdata is not None and event.ydata is not None:
            self.vertical_line = ax.axvline(event.xdata, color='gray', linestyle='--', linewidth=0.7)
            self.horizontal_line = ax.axhline(event.ydata, color='gray', linestyle='--', linewidth=0.7)
            plt.draw()


    def reset_coordinates(self, event):
        """Reseta as coordenadas e limpa o gráfico."""
        self.coordinates = []
        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("Clique para adicionar coordenadas. Feche a janela para finalizar a ação")
        ax.set_xlabel("Eixo X")
        ax.set_ylabel("Eixo Y")
        ax.grid(True)
        print("As coordenadas foram resetadas.")
        plt.draw()

    def janela(self):
        """Configuração do gráfico e interação."""
        global ax  # Torna o eixo acessível para funções externas
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Configuração do gráfico
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("Clique para adicionar coordenadas. Feche a janela para finalizar a ação")
        ax.set_xlabel("Eixo X")
        ax.set_ylabel("Eixo Y")
        ax.grid(True)  # Adiciona uma grade

        # Adiciona o botão de reset
        reset_ax = plt.axes([0.8, 0.01, 0.1, 0.05])  # Posição do botão (esquerda, embaixo, largura, altura)
        reset_button = Button(reset_ax, 'Reset', color='lightgray', hovercolor='gray')
        reset_button.on_clicked(self.reset_coordinates)

        # Conecta os eventos de clique e movimento do mouse
        cid_click = fig.canvas.mpl_connect('button_press_event', self.on_click)
        cid_motion = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Exibe a janela
        plt.show()

        # Desconecta os eventos após o fechamento
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_motion)

        # Mostra as coordenadas finais
        print("\nCoordenadas registradas:")
        for i, (x, y) in enumerate(self.coordinates):
            print(f"Ponto {i+1}: ({x:.2f}, {y:.2f})")
        return self.coordinates

# Uso da classe
if __name__ == "__main__":
    capturador = PegarCoordenadaJanela()
    coordenadas = capturador.janela()

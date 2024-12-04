import os
import pandas as pd
from datetime import datetime
import json

class SaveFile():
    def __init__(self):
        pass

    def save_json_to_file(self, df: pd.DataFrame, path: str = None, informacao: dict = None):
        """
        Função para salvar um DataFrame diretamente em um arquivo JSON usando pandas.
        Cria uma estrutura de pastas ano/mês/dia e salva o arquivo com um nome único.

        Args:
            df (pd.DataFrame): DataFrame a ser salvo.
            path (str, optional): Nome da pasta do algoritmo. Defaults to None.
            informacao (dict, optional): Configurações adicionais para serem salvas como a primeira linha. Defaults to None.
        """
        try:
            # Se informacao for None, inicialize com um dicionário vazio ou valores padrão
            if informacao is None:
                informacao = {"status": "sem informações"}

            now = datetime.now()
            path_default = f'./output/{now.year}/{now.month}/{now.day}/'
            path_complete = path_default + (path if path else "")
            os.makedirs(path_complete, exist_ok=True)
            
            # Gera um nome de arquivo único
            file_count = len(os.listdir(path_complete))
            file_name = f"output_{file_count + 1}.json"
            file_path = os.path.join(path_complete, file_name)
            
            # Adiciona o conteúdo do DataFrame à chave "dados" de informacao
            informacao["dados"] = df.to_dict(orient='records')  # Converte df para lista de dicionários

            # Salva o dicionário informacao no arquivo JSON, tudo em uma linha
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(informacao, file, ensure_ascii=False)  # Remove o indent para gravar tudo em uma linha

            print(f"Dados salvos com sucesso no arquivo {file_name}")
        except Exception as e:
            print(f"Erro ao salvar o arquivo JSON: {e}")

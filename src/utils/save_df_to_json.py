from datetime import datetime
import os
import pandas as pd

class SaveFile():
    def __init__(self):
        pass

    def save_json_to_file(self, df: pd.DataFrame, path: str = None):
        """
            Função para salvar um DataFrame diretamente em um arquivo JSON usando pandas.
            Cria uma estrutura de pastas ano/mês/dia e salva o arquivo com um nome único.
        Args:
            df (pd.DataFrame): _description_
            path (str, optional): nome da pasta do algoritmo. Defaults to None.
        """
        try:
            now = datetime.now()
            path_default = f'./output/{now.year}/{now.month}/{now.day}/'
            path_complete = path_default + path
            os.makedirs(path_complete, exist_ok=True)
            
            # Gera um nome de arquivo único
            file_count = len(os.listdir(path_complete))
            file_name = f"output_{file_count + 1}.json"
            
            df.to_json(os.path.join(path_complete, file_name), orient='records', lines=True)
            print(f"Dados salvos com sucesso no arquivo {file_name}")
        except Exception as e:
            print(f"Erro ao salvar o arquivo JSON: {e}")
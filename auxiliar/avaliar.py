import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

def carregar_pesos_e_avaliar_modelos(model, rede, id_rodada, dataset_teste, dados_teste, batch_size, imagenet_descongelada, data_aug, nome_otimizador, nome_loss, lr, attention, camada_pooling, dense_1, dropout_1, dense_2, dropout_2, dense_3, dropout_3):
    """
    Carrega os pesos dos modelos salvos, avalia os modelos no conjunto de teste e registra os resultados.

    Args:
        model (tf.keras.Model): O modelo a ser avaliado.
        rede (str): Nome do modelo rede.
        id_rodada (int): Identificador da rodada de treinamento.
        dataset_teste (tf.data.Dataset): Dataset de teste.
        dados_teste (pd.DataFrame): DataFrame contendo os dados de teste.
        batch_size (int): Tamanho do batch para a avaliação.
        imagenet_descongelada (bool): Se os pesos da ImageNet foram descongelados.
        data_aug (bool): Se a data augmentation foi usada.
        nome_otimizador (str): Nome do otimizador.
        nome_loss (str): Nome da função de perda.
        lr (float): Taxa de aprendizado.
        attention (str): Tipo de bloco de atenção usado.
        camada_pooling (str): Tipo de camada de pooling usada.
        dense_1 (int): Número de unidades na primeira camada densa.
        dropout_1 (float): Taxa de dropout após a primeira camada densa.
        dense_2 (int): Número de unidades na segunda camada densa.
        dropout_2 (float): Taxa de dropout após a segunda camada densa.
        dense_3 (int): Número de unidades na terceira camada densa.
        dropout_3 (float): Taxa de dropout após a terceira camada densa.
    """
    nomes_metricas = ['Loss', 'MAE', 'MSE', 'RMSE']
    dir_modelos_salvos_para_teste = f'./redes/{rede}/modelos_salvos'
    resultados_lista = []

    os.makedirs('./log_teste/previsoes', exist_ok=True)

    for nome_modelo in os.listdir(dir_modelos_salvos_para_teste):
        if nome_modelo.startswith(f'{rede}_{id_rodada}'):
            caminho_completo = os.path.join(dir_modelos_salvos_para_teste, f'{nome_modelo}')
            model.load_weights(caminho_completo)
            
            resultado_teste = model.evaluate(dataset_teste, steps=np.ceil(len(dados_teste) / batch_size))
            print(f'modelo {nome_modelo} - resultado: {resultado_teste}')

            resultados_teste_float = [float(x) for x in resultado_teste]
            resultados_dict = dict(zip(
                ['arquivo', 'rede', 'imagenet1k', 'dataaug', 'otimizador', 'lossname', 'lr', 'attention', 'batchsize', 'pooling2d', 'dense_1', 'dropout_1', 'dense_2', 'dropout_2', 'dense_3', 'dropout_3'] + 
                nomes_metricas, 
                [nome_modelo, rede, imagenet_descongelada, data_aug, nome_otimizador, nome_loss, lr, attention, batch_size, camada_pooling, dense_1, dropout_1, dense_2, dropout_2, dense_3, dropout_3] + 
                resultados_teste_float
            ))

            resultados_lista.append(resultados_dict)
            salvar_previsoes(model, dataset_teste, dados_teste, nome_modelo, id_rodada, batch_size)
    
    salvar_resultados(resultados_lista)

def salvar_previsoes(model, dataset_teste, dados_teste, nome_modelo, id_rodada, batch_size):
    """
    Salva as previsões do modelo para o conjunto de teste.

    Args:
        model (tf.keras.Model): O modelo usado para gerar as previsões.
        dataset_teste (tf.data.Dataset): Dataset de teste.
        dados_teste (pd.DataFrame): DataFrame contendo os dados de teste.
        nome_modelo (str): Nome do modelo.
        id_rodada (int): Identificador da rodada de treinamento.
        batch_size (int): Tamanho do batch para a predição.
    """
    num_steps = np.ceil(len(dados_teste) / batch_size)
    predicoes = model.predict(dataset_teste, steps=num_steps)
    labels_teste = dados_teste['boneage'].values
    ids = dados_teste['id'].values

    mae_manual = np.mean(np.abs(predicoes.flatten() - labels_teste))
    print(f'MAE Manual para o modelo {nome_modelo}: {mae_manual}')

    resultados_df = pd.DataFrame({
        'id': ids, 
        'real': labels_teste,  
        'predito': predicoes.flatten()
    })
    
    resultados_df.to_csv(f'./log_teste/previsoes/resultados_{nome_modelo}_{id_rodada}.csv', index=False)

def salvar_resultados(resultados_lista):
    """
    Salva os resultados da avaliação em um arquivo CSV.

    Args:
        resultados_lista (list): Lista de dicionários contendo os resultados da avaliação.
    """
    log_teste_df = pd.DataFrame(resultados_lista)
    cabeçalho = not os.path.isfile(f'./log_teste/log_teste.csv')
    log_teste_df.to_csv(f'./log_teste/log_teste.csv', mode='a', header=cabeçalho, index=False)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

def carregar_pesos_e_avaliar_modelos(model, rede, id_rodada, dataset_teste, dados_teste, batch_size, imagenet_descongelada, data_aug, nome_otimizador, nome_loss, lr, attention, camada_pooling, dense_1, dropout_1, dense_2, dropout_2, dense_3, dropout_3):
    '''
    avalia os modelos no conjunto de teste e registra os resultados.

    args:
        model (tf.keras.Model): modelo
        rede (str): nome do modelo
        id_rodada (int): rodada de treinamento
        dataset_teste (tf.data.Dataset): dataset de teste
        dados_teste (pd.DataFrame): dataFrame contendo os dados de teste
        batch_size (int): batch
        data_aug (bool): data augmentation
        nome_otimizador (str): nome do otimizador
        nome_loss (str): nome da função de perda
        lr (float): taxa de aprendizado
        attention (str): tipo de bloco de atenção usado
        camada_pooling (str): tipo de camada de pooling usada.
        dense_1 (int): primeira camada densa.
        dropout_1 (float): dropout depois da primeira camada densa
        dense_2 (int): segunda camada densa
        dropout_2 (float): dropout depois da segunda camada densa
        dense_3 (int): terceira camada densa
        dropout_3 (float): dropout depois da terceira camada densa

    maioria desses args é só pra gente salvar no csv de log e ficar registrada a arquitetura pra depois se a gente quiser dar o load do modelo
    '''
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

    num_steps = np.ceil(len(dados_teste) / batch_size)
    predicoes = model.predict(dataset_teste, steps=num_steps)
    labels_teste = dados_teste['boneage'].values
    ids = dados_teste['id'].values

    mae_manual = np.mean(np.abs(predicoes.flatten() - labels_teste))
    print(f'conferenciazinha {nome_modelo}: {mae_manual}')

    resultados_df = pd.DataFrame({
        'id': ids, 
        'real': labels_teste,  
        'predito': predicoes.flatten()
    })
    
    resultados_df.to_csv(f'./log_teste/previsoes/resultados_{nome_modelo}_{id_rodada}.csv', index=False)

def salvar_resultados(resultados_lista):

    log_teste_df = pd.DataFrame(resultados_lista)
    cabecalho = not os.path.isfile(f'./log_teste/log_teste.csv')
    log_teste_df.to_csv(f'./log_teste/log_teste.csv', mode='a', header=cabecalho, index=False)

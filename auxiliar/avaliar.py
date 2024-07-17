import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics import r2_score


'''
avalia o modelo salvo em diferentes rodadas de treinamento

argumentos:
    - model: modelo
    - rede: nome da rede
    - id_rodada: identificador da rodada
    - dataset_teste: dataset de teste
    - dados_teste: dataframe contendo os dados de teste
    - batch_size: tamanho do batch
    - imagenet: flag indicando se os pesos pre-treinados do ImageNet foram usados, se é 1k, 21k ou 21k-ft1k
    - data_aug: flag indicando se augmentacao de dados foi usada
    - nome_otimizador: nome do otimizador
    - nome_loss: nome da funcao de perda
    - lr: taxa de aprendizado
    - attention: tipo de atencao usada, pode ser em branco
    - camada_pooling: tipo de camada de pooling usada, pode ser flatten
    - denses: lista com o numero de unidades em cada camada densa
    - dropouts: lista com as taxas de dropout correspondentes para cada camada densa
    - tempo_treino: tempo de treinamento
retorna:
    - nenhum retorno, mas salva os resultados da avaliacao e previsoes em csv
'''

def avaliar_modelo(model, rede, id_rodada, dataset_teste, dados_teste, batch_size, imagenet, data_aug, nome_otimizador, nome_loss, lr, attention, camada_pooling, denses, dropouts, tempo_treino):

    nomes_metricas = ['Loss', 'MAE', 'MSE', 'RMSE', 'R2']
    dir_modelos_salvos_para_teste = f'./redes/{rede}/modelos_salvos'
    resultados_lista = []

    os.makedirs('./log_teste/previsoes', exist_ok=True)

    for nome_modelo in os.listdir(dir_modelos_salvos_para_teste):
        if nome_modelo.startswith(f'{rede}-{id_rodada}'):
            caminho_completo = os.path.join(dir_modelos_salvos_para_teste, f'{nome_modelo}')
            model.load_weights(caminho_completo)
            
            resultado_teste = model.evaluate(dataset_teste, steps=np.ceil(len(dados_teste) / batch_size))
            print(f'modelo {nome_modelo} - resultado: {resultado_teste}')

            resultados_teste_float = [float(x) for x in resultado_teste]
            
            densas_preenchidas = denses + [None] * (5 - len(denses))
            dropouts_preenchidos = dropouts + [None] * (5 - len(dropouts))

            salvar_previsoes(model, dataset_teste, dados_teste, nome_modelo, id_rodada, batch_size)
            r2 = calcular_r2(nome_modelo, id_rodada)

            resultados_dict = dict(zip(
                ['arquivo', 'rede', 'imagenet', 'dataaug', 'otimizador', 'lossname', 'lr', 'attention', 'batchsize', 'pooling2d'] + 
                [f'dense_{i+1}' for i in range(5)] + [f'dropout_{i+1}' for i in range(5)] + 
                nomes_metricas + ['tempo_treino'], 
                [nome_modelo, rede, imagenet, data_aug, nome_otimizador, nome_loss, lr, attention, batch_size, camada_pooling] + 
                densas_preenchidas + dropouts_preenchidos + 
                resultados_teste_float + [r2, tempo_treino]
            ))

            resultados_lista.append(resultados_dict)
    
    salvar_resultados(resultados_lista)


'''
salva as previsoes do modelo

argumentos:
    - model: modelo
    - dataset_teste: dataset de teste
    - dados_teste: dataframe contendo os dados de teste
    - nome_modelo: nome do modelo
    - id_rodada: identificador da rodada de treinamento
    - batch_size: tamanho do batch
retorna:
    - nenhum retorno, mas salva as previsoes em csv
'''

def salvar_previsoes(model, dataset_teste, dados_teste, nome_modelo, id_rodada, batch_size):

    num_steps = np.ceil(len(dados_teste) / batch_size)
    predicoes = model.predict(dataset_teste, steps=num_steps)
    labels_teste = dados_teste['boneage'].values
    ids = dados_teste['id'].values

    mae_manual = np.mean(np.abs(predicoes.flatten() - labels_teste))
    print(f'conferencia {nome_modelo}: {mae_manual}')

    resultados_df = pd.DataFrame({
        'id': ids, 
        'real': labels_teste,  
        'predito': predicoes.flatten()
    })
    
    resultados_df.to_csv(f'./log_teste/previsoes/resultados_{nome_modelo}_{id_rodada}.csv', index=False)


'''
tf nao tem R² nativo, tive que exportar do sklearn.
ele tira a metrica direto do csv de chutes

argumentos:
    - nome_modelo: nome do modelo
    - id_rodada: identificador da rodada de treinamento
retorna:
    - r2: coeficiente R2
'''

def calcular_r2(nome_modelo, id_rodada):
    resultados_df = pd.read_csv(f'./log_teste/previsoes/resultados_{nome_modelo}_{id_rodada}.csv')
    r2 = r2_score(resultados_df['real'], resultados_df['predito'])
    return r2


'''
salva os resultados da avaliacao em um arquivo csv

argumentos:
    - resultados_lista: lista de dicionarios contendo os resultados da avaliacao
retorna:
    - nenhum retorno, mas salva os resultados csv
'''

def salvar_resultados(resultados_lista):

    log_teste_df = pd.DataFrame(resultados_lista)
    cabecalho = not os.path.isfile(f'./log_teste/log_teste.csv')
    log_teste_df.to_csv(f'./log_teste/log_teste.csv', mode='a', header=cabecalho, index=False)



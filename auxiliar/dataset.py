import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .preprocess import get_preprocess
import matplotlib.pyplot as plt
import numpy as np

#########################
##                     ##
##  data augmentation  ##
##                     ##
#########################

'''
criando uma sequencia de augmentation de dados para aumentar a variabilidade dos dados de treinamento
    - RandomTranslation: desloca a imagem aleatoriamente na horizontal e vertical dentro dos fatores especificados
    - RandomFlip: inverte a imagem horizontalmente
    - RandomRotation: rotaciona a imagem aleatoriamente
    - RandomZoom: aplica um zoom aleatorio na imagem

o chat gpt sempre tenta colocar um data generator. nao use! é desatualizado com as coisas mais novas dokeras.
essa é uma abordagem defazada. usa o keras.sequential que é melhor.
'''


data_augmentation = keras.Sequential(
    [
        layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.15, 0.15), fill_mode="nearest"),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.15, fill_mode="nearest"),
        layers.RandomZoom(height_factor=(-0.2, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"),
    ]
)


#########################
##                     ##
##   processar imagem  ##
##                     ##
#########################

'''
argumentos:
    - nome_do_arquivo: caminho da imagem
    - label: rotulo da imagem
    - male: flag indicando se e do genero masculino
    - preprocess_input: funcao de preprocessamento especifica da rede
    - img_tamanho: tamanho da imagem apos redimensionamento
    - png: flag indicando se a imagem e png (default: True)
retorna:
    - imagem_normalizada: imagem processada e normalizada(caso a funçao o faça, ne? algumas é só pass through)
    - label: rotulo da imagem
'''


def processar_imagem(nome_do_arquivo, label, male, preprocess_input, img_tamanho, png=True):
    nome_da_imagem = tf.io.read_file(nome_do_arquivo)
    if png:
        imagem_decodificada = tf.image.decode_png(nome_da_imagem, channels=3)
    else:
        imagem_decodificada = tf.image.decode_jpeg(nome_da_imagem, channels=3)

    imagem_redimensionada = tf.image.resize(imagem_decodificada, img_tamanho)
    imagem_normalizada = preprocess_input(imagem_redimensionada)

    if male:
        mascara_red = tf.ones_like(imagem_normalizada) * [10, 0, 0]
        imagem_normalizada = imagem_normalizada + mascara_red
    else:
        mascara_green = tf.ones_like(imagem_normalizada) * [0, 10, 0]
        imagem_normalizada = imagem_normalizada + mascara_green

    return imagem_normalizada, label


#########################
##                     ##
##    criar dataset    ##
##                     ##
#########################

'''
argumentos:
    - dataframe: pandas dataframe contendo informacoes das imagens
    - diretorio: diretorio onde as imagens estao armazenadas
    - batch_size: tamanho do batch
    - rede: tipo de rede neural para obter a funcao de preprocessamento e tamanho da imagem
    - shuffle: flag indicando se o dataset deve ser embaralhado (default: False)
    - png: flag indicando se as imagens sao png (default: True)
    - repeat: flag indicando se o dataset deve ser repetido indefinidamente (default: True)
    - data_aug: flag indicando se a augmentacao de dados deve ser aplicada (default: False)
retorna:
    - dataset: tensorflow dataset com as imagens processadas e configuradas
'''


def criar_dataset(dataframe, diretorio, batch_size, rede, shuffle=False, png=True, repeat=True, data_aug=False):

    preprocess_input, img_tamanho = get_preprocess(rede)
    
    fotos = dataframe['id'].map(lambda x: f"{diretorio}/{x}").tolist()
    labels = dataframe['boneage'].tolist()
    male_flags = dataframe['male'].tolist()  
    
    dataset = tf.data.Dataset.from_tensor_slices((fotos, labels, male_flags))
    
    def map_func(nome_do_arquivo, label, male):
        return processar_imagem(nome_do_arquivo, label, male, preprocess_input, img_tamanho, png)
    
    dataset = dataset.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    
    if data_aug:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    
    dataset = dataset.batch(batch_size)
    
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

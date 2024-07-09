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

def criar_dataset(dataframe, diretorio, batch_size, rede, shuffle=False, png=True, repeat=True, data_aug=False):
    """
    Cria um dataset TensorFlow a partir de um dataframe contendo caminhos de arquivos e labels.

    Args:
        dataframe (pd.DataFrame): DataFrame contendo os caminhos das imagens e labels.
        diretorio (str): Diretório base onde as imagens estão armazenadas.
        batch_size (int): Tamanho do batch para o treinamento.
        rede (str): Nome do modelo CNN a ser usado para determinar o preprocessamento.
        shuffle (bool): Se deve ou não embaralhar o dataset.
        png (bool): Se as imagens são PNG (True) ou JPEG (False).
        repeat (bool): Se o dataset deve ser repetido indefinidamente.
        data_aug (bool): Se deve ou não aplicar data augmentation.

    Returns:
        tf.data.Dataset: Dataset processado e pronto para o treinamento.
    """
    preprocess_input, img_tamanho = get_preprocess(rede)
    
    nome_dos_arquivos = dataframe['id'].map(lambda x: f"{diretorio}/{x}").tolist()
    labels = dataframe['boneage'].tolist()
    male_flags = dataframe['male'].tolist()  
    
    dataset = tf.data.Dataset.from_tensor_slices((nome_dos_arquivos, labels, male_flags))
    
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

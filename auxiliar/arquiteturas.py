import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, ResNet152V2, ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, AveragePooling2D, Dropout, Dense, Flatten
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.models import Model
from .attention_blocks import cbam_block, squeeze_excite_block
from efficientnet_v2 import EfficientNetV2S, EfficientNetV2M, EfficientNetV2B3, EfficientNetV2L

'''
argumentos:
    - nome: nome do modelo base
    - weights: pesos pre-treinados a serem usados
    - include_top: flag indicando se inclui a parte superior da rede (default: False)
retorna:
    - modelo base especificado

por que include top é False? porque todas essas redes foram feitas para classificaçao. quando voce coloca True, voce ta deixando
os 1k ou 21k de neuronios de classificacao da imagenet. nossa tarefa é outra.
'''


def get_model(nome, weights, include_top=False):
    models_dict = {
        'EfficientNetV2S': EfficientNetV2S,
        'EfficientNetV2L': EfficientNetV2L,
        'InceptionV3': InceptionV3,
        'InceptionResNetV2': InceptionResNetV2,
        'EfficientNetV2M': EfficientNetV2M,
        'EfficientNetV2B3': EfficientNetV2B3,
        'ResNet152V2': ResNet152V2,
        'ResNet50V2': ResNet50V2
    }
    return models_dict[nome](weights=weights, include_top=include_top)

'''
argumentos:
    - base_model: modelo base pre-treinado
    - loss: funcao de perda
    - otimizador: otimizador
    - attention: tipo de bloco de atencao a ser usado ('se' ou 'cbam')
    - denses: lista com o numero de unidades em cada camada densa
    - dropouts: lista com as taxas de dropout correspondentes para cada camada densa
    - pooling: tipo de camada de pooling a ser usada ('global_max', 'global_avg', 'max', 'avg' ou 'flatten') ps. flatten não é pooling.
    - funcao_atv: funcao de ativacao nas camadas densas (default: 'relu')
retorna:
    - modelo compilado
'''


def build_model(
    base_model,
    loss,
    otimizador,
    attention,
    denses,
    dropouts,
    pooling,
    funcao_atv='relu'):

    x = base_model.output

    if attention == 'se':
        x = squeeze_excite_block(x)
    elif attention == 'cbam':
        x = cbam_block(x)

    if pooling == 'global_max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)
    elif pooling == 'global_avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = MaxPooling2D(pool_size=(2, 2), name='max_pool')(x)
    elif pooling == 'avg':
        x = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(x)
    else:
        x = Flatten(name='flatten')(x)

    for i, (dense_units, dropout_rate) in enumerate(zip(denses, dropouts)):
        if dense_units > 0:
            x = Dense(dense_units, activation=funcao_atv, name=f'dense_{i+1}')(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)

    camada_de_previsao = Dense(1, activation='linear', name='dense_prev')(x)

    model = Model(inputs=base_model.input, outputs=camada_de_previsao)
    model.compile(
        optimizer=otimizador,
        loss=loss,
        metrics=[MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError()]
    )
    print(f'modelo compilado')

    return model


'''
argumentos:
    - rede: nome do base model
    - loss: funcao de perda
    - weights: pesos pre-treinados a serem usados(21k, 1k, 21k-ft1k)
    - otimizador: otimizador
    - attention: tipo de bloco de atencao ('se' ou 'cbam')
    - denses: lista com o numero de unidades em cada camada densa
    - dropouts: lista com as taxas de dropout correspondentes para cada camada densa
    - pooling: tipo de camada de pooling a ser usada ('global_max', 'global_avg', 'max', 'avg' ou 'flatten') ps. flatten não é pooling.
retorna:
    - modelo compilado
'''


def custom_model(rede, loss, weights, otimizador, attention, denses, dropouts, pooling):
    base_model = get_model(rede, weights)
    base_model.trainable = True
    return build_model(base_model, loss, otimizador, attention, denses, dropouts, pooling)

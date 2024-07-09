import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, EfficientNetV2S, EfficientNetV2M, EfficientNetV2B3, EfficientNetB7, ResNet152V2, EfficientNetV2L, ConvNeXtBase
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from keras.models import Model
from .attention_blocks import cbam_block, squeeze_excite_block 

###########################
##                       ##
##     Arquiteturas      ##
##                       ##
###########################

def get_model(nome, weights='imagenet', include_top=False):
    models_dict = {
        'EfficientNetV2S': EfficientNetV2S,
        'EfficientNetV2L': EfficientNetV2L,
        'InceptionV3': InceptionV3,
        'InceptionResNetV2': InceptionResNetV2,
        'EfficientNetV2M': EfficientNetV2M,
        'EfficientNetV2B3': EfficientNetV2B3,
        'ResNet152V2': ResNet152V2,
        'ConvNeXtBase': ConvNeXtBase
    }
    return models_dict[nome](weights=weights, include_top=include_top)

def build_model(
    base_model,
    loss,
    otimizador,
    attention,
    camada_densa_1,
    dropout_1,
    camada_densa_2,
    dropout_2, 
    camada_densa_3,
    dropout_3,
    pooling,
    funcao_atv='relu'):

    x = base_model.output

    if attention == 'se':
        x = squeeze_excite_block(x)
    elif attention == 'cbam':
        x = cbam_block(x)
    
    if pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)
    else:  
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    
    if camada_densa_1 > 0:
        x = Dense(camada_densa_1, activation=funcao_atv, name='dense_1')(x)
        if dropout_1 > 0:
            x = Dropout(dropout_1, name='dropout_1')(x)

    if camada_densa_2 > 0:
        x = Dense(camada_densa_2, activation=funcao_atv, name='dense_2')(x)
        if dropout_2 > 0:
            x = Dropout(dropout_2, name='dropout_2')(x)

    if camada_densa_3 > 0:
        x = Dense(camada_densa_3, activation=funcao_atv, name='dense_3')(x)
        if dropout_3 > 0:
            x = Dropout(dropout_3, name='dropout_3')(x)

    camada_de_previsao = Dense(1, activation='linear', name='dense_prev')(x)

    model = Model(inputs=base_model.input, outputs=camada_de_previsao)
    model.compile(
        optimizer=otimizador,
        loss=loss,
        metrics=[MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError()]
    )
    print(f'modelo compilado')

    return model

def custom_model(rede, loss, otimizador, attention, camada_densa_1, dropout_1, camada_densa_2, dropout_2, camada_densa_3, dropout_3, pooling):
    base_model = get_model(rede)
    base_model.trainable = True
    return build_model(base_model, loss, otimizador, attention, camada_densa_1, dropout_1, camada_densa_2, dropout_2, camada_densa_3, dropout_3, pooling)

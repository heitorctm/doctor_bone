from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inception_resnet_v2
from keras.applications.efficientnet_v2 import preprocess_input as preprocess_efficientnet_v2s
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet50v2
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_convnext

'''
funcao pra pegar a funcao de preprocessamento e o tamanho de entrada para diferentes modelos de redes, esses tamanhos sao os que o modelo espera de acordo com a documentacao do keras.
os modelos suportados:
- InceptionV3
- InceptionResNetV2
- EfficientNetV2L
- EfficientNetV2S
- ResNet50V2
- ResNet152V2
- EfficientNetV2M
- EfficientNetV2B3

argumentos:
    - cnn: nome do modelo (str)
retorna:
    - preprocess_input: funcao de preprocessamento especifica do modelo
    - tamanho: tamanho da imagem de entrada esperada pelo modelo
'''


def get_preprocess(cnn):

    preprocess_dict = {
        'InceptionV3': (preprocess_inception_v3, (299, 299)),
        'InceptionResNetV2': (preprocess_inception_resnet_v2, (299, 299)),
        'EfficientNetV2L': (preprocess_efficientnet_v2s, (480, 480)),
        'EfficientNetV2S': (preprocess_efficientnet_v2s, (384, 384)),
        'ResNet50V2': (preprocess_resnet50v2, (224, 224)),
        'ResNet152V2': (preprocess_resnet50v2, (224, 224)),
        'EfficientNetV2M': (preprocess_efficientnet_v2s, (480, 480)),
        'EfficientNetV2B3': (preprocess_efficientnet_v2s, (300, 300))
    }
    if cnn not in preprocess_dict:
        raise ValueError(f'não tem esse modelo')
    return preprocess_dict[cnn]

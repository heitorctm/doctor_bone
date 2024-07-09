import json
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import ast

def carregar_json(json_path):
    with open(json_path, 'r') as file:
        annotations = json.load(file)
    return annotations

def adicionar_id_ao_csv(dataframe, annotations):
    images_info = {img['file_name']: img['id'] for img in annotations['images']}
    dataframe['image_id'] = dataframe['id'].map(lambda x: images_info.get(x, None))
    return dataframe

def merge(dataframe, annotations):
    annotations_info = {ann['image_id']: ann for ann in annotations['annotations']}
    
    def get_annotation(image_id):
        return annotations_info.get(image_id, None)
    
    def get_bbox(image_id):
        annotation = get_annotation(image_id)
        return annotation['bbox'] if annotation else None

    def get_keypoints(image_id):
        annotation = get_annotation(image_id)
        return annotation['keypoints'] if annotation else None
    
    dataframe['bbox'] = dataframe['image_id'].map(lambda x: get_bbox(x))
    dataframe['keypoints'] = dataframe['image_id'].map(lambda x: get_keypoints(x))
    
    return dataframe

def processar_imagem(nome_do_arquivo, bbox, keypoints, male=False, png=True):
    nome_da_imagem = tf.io.read_file(nome_do_arquivo)
    if png:
        imagem_decodificada = tf.image.decode_png(nome_da_imagem, channels=3)
    else:
        imagem_decodificada = tf.image.decode_jpeg(nome_da_imagem, channels=3)

    if bbox is not None:
        x, y, width, height = map(int, bbox)
        img_height, img_width, _ = imagem_decodificada.shape

        x = max(0, x)
        y = max(0, y)
        width = min(width, img_width - x)
        height = min(height, img_height - y)

        if width > 0 and height > 0:  
            imagem_decodificada = tf.image.crop_to_bounding_box(imagem_decodificada, y, x, height, width)
            
            if keypoints is not None:
                keypoints = [k if i % 3 == 2 else k - (x if i % 3 == 0 else y) for i, k in enumerate(keypoints)]

    imagem_decodificada = tf.image.convert_image_dtype(imagem_decodificada, dtype=tf.float32)
    return imagem_decodificada, keypoints

def salvar_imagem(imagem, output_path):
    tf.keras.preprocessing.image.save_img(output_path, imagem)

def atualizar_dataframe(dataframe, input_dir, output_dir, png=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for index, row in dataframe.iterrows():
        nome_do_arquivo = os.path.join(input_dir, row['id'])
        bbox = row['bbox']
        keypoints = row[[col for col in dataframe.columns if col.startswith('keypoint_')]].values
        male = row['male']

        imagem_normalizada, keypoints = processar_imagem(nome_do_arquivo, bbox, keypoints, male, png)

        if male:
            mask = tf.ones_like(imagem_normalizada) * [0.15, 0.0, 0.0]
            imagem_normalizada = imagem_normalizada + mask

        output_path = os.path.join(output_dir, row['id'])
        salvar_imagem(imagem_normalizada, output_path)

        dataframe.at[index, 'id'] = output_path
        dataframe.at[index, 'keypoints'] = keypoints.tolist()
    
    return dataframe

def plotar_imagens(dataframe, diretorio, num_images=10):
    selected_images = dataframe.sample(n=num_images)
    
    for index, row in selected_images.iterrows():
        nome_do_arquivo = os.path.join(diretorio, os.path.basename(row['id']))
        
        # Pegar os valores das novas colunas de bbox
        bbox_x = row['bbox_x']
        bbox_y = row['bbox_y']
        bbox_width = row['bbox_width']
        bbox_height = row['bbox_height']
        
        original_image = Image.open(nome_do_arquivo)
        
        fig, ax = plt.subplots(1)
        ax.imshow(original_image)
        
        # Desenhar a bbox
        if bbox_x is not None and bbox_y is not None and bbox_width is not None and bbox_height is not None:
            rect = patches.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        
        # Pegar os valores das novas colunas de keypoints
        keypoints = row[[col for col in dataframe.columns if col.startswith('keypoint_')]].values

        if keypoints is not None:
            for i in range(0, len(keypoints), 3):
                kx, ky, _ = keypoints[i:i+3]
                ax.plot(kx, ky, 'ro')
        
        plt.title(f"Image: {os.path.basename(row['id'])}")
        plt.show()

def ajustar_bbox_csv(dataframe):
    def ajustar_bbox_row(row):
        bbox = ast.literal_eval(row['bbox'])
        x, y, width, height = map(int, bbox)
        return [0, 0, width, height]
    
    dataframe['bbox'] = dataframe.apply(ajustar_bbox_row, axis=1)
    bbox_df = pd.DataFrame(dataframe['bbox'].tolist(), index=dataframe.index, columns=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'])
    dataframe = pd.concat([dataframe, bbox_df], axis=1)
    dataframe.drop(columns=['bbox'], inplace=True)
    return dataframe

def ajustar_keypoints_csv(dataframe):
    keypoints = dataframe['keypoints'].apply(lambda x: ast.literal_eval(x))
    keypoints_df = pd.DataFrame(keypoints.tolist(), index=dataframe.index, columns=[f'keypoint_{i}' for i in range(1, 52)])
    dataframe = pd.concat([dataframe, keypoints_df], axis=1)
    dataframe.drop(columns=['keypoints'], inplace=True)
    return dataframe

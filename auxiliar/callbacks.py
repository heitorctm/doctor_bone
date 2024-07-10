import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras import backend as K


    #########################
    ##                     ##
    ##      auxiliares     ##
    ##                     ##
    #########################


'''
callbacks sao algumas funções já pre estabelecidas pelo keras que te retorna ou faz algo enquanto treina.
depois uma funcaozinha pra filtrar por genero. nao tem como eu ficar treinando com todos os dados sempre. (nao tem mais isso)
fiquei testando varias so com o masculino, peguei as melhores e depois testei com fem e com os dois.
'''

def callbacks(dir_modelos_salvos, dir_csv_log, check_best, early_stop, log, reduce_lr, check_15):
    callbacks = []
    print("callbacks adicionados:") 

    if check_best:
        print('check best')
        checkpoint_best = ModelCheckpoint(
            monitor='val_loss',
            filepath=dir_modelos_salvos,
            save_weights_only=False,
            save_best_only=True,
            verbose=1,
        )
        callbacks.append(checkpoint_best)
        

    if check_15:
        print('check 15')
        checkpoint_15 = ModelCheckpoint(
            monitor='val_loss',
            filepath=dir_modelos_salvos,
            save_weights_only=False,
            save_best_only=False,
            verbose=1,
            period=10
        )
        callbacks.append(checkpoint_15)

    if early_stop:
        print('early stop')
        early_stopping = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=5,
            verbose=1,
            min_delta=0.01
        )
        callbacks.append(early_stopping)

    if log:
        print('log')
        csv_log = CSVLogger(
            dir_csv_log,
            append=True
        )
        callbacks.append(csv_log)

    if reduce_lr:

        fator = 0.3
        paciencia = 2
        print('reduzindo lr')
        reduce_learning_rate = ReduceLROnPlateau(
            monitor='val_loss',
            factor=fator,
            patience=paciencia,
            min_lr=0.0000001,
            verbose=1
        )
        callbacks.append(reduce_learning_rate)


    return callbacks

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.utils import pad_sequences

def Policy_net(
        state_seq,
        seq_len,
        n_actions: int, 
        n_features: int,
        height: int = 21, 
        width: int = 21, 
        depth: int = 11,
        hidden_dim = 64
    ):


    state_input = layers.Input(shape=(n_features,))
    goal_input = layers.Input(shape=(n_features,))
    map_input = layers.Input(shape=(height, width, depth))
    input_data = layers.Concatenate()([state_input, goal_input])
    
    ## StateSeqEmb class in the original code 
    input_data = pad_sequences(input_data, padding='post')
    embed = layers.Embedding(input_data.shape[0] + 1, hidden_dim, mask_zero = True)(input_data)
    x = layers.GRU(hidden_dim)(embed)
    x = layers.GRU(hidden_dim)(x)
    x = layers.GRU(hidden_dim)(x)


    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    output = layers.Dense(n_actions, activation='relu')(x)
    model = Model(input_data, output)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model


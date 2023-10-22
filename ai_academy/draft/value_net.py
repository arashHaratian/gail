import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.utils import pad_sequences



def Value_net(
        # state_seq,
        seq_len,
        action,
        n_actions: int, 
        n_features: int,
        height: int = 21, 
        width: int = 21, 
        depth: int = 11,
        hidden_dim = 64
    ):

    ## StateSeqEmb class in the original code 
    state_input = layers.Input(shape=(None, n_features)) ## batch, n_features(3)
    goal_input = layers.Input(shape=(None, n_features)) ## batch, n_features(3)
    state_seq_input = layers.Input(shape=(None, seq_len, n_features)) ## batch, seq_len, n_features(3) 3:(x,y,z)
    action_input = layers.Input(shape=(None, n_actions)) ## batch, n_actions(6)

    embed = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True)(state_seq_input)
    padded_embed = pad_sequences(embed, padding='post')
    x_rnn = layers.StackedRNNCells(
        [layers.GRU(hidden_dim),
        layers.GRU(hidden_dim),
        layers.GRU(hidden_dim)])(padded_embed)
    
    x = layers.Concatenate(axis=1)([x_rnn, goal_input, state_input, action_input])

    ## Vanilla Value net class in the original code 
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    value = layers.Dense(1, activation='linear')(x)

    model = Model([state_input, goal_input, state_seq_input, action_input], value)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    return model

    

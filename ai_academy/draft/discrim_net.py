import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.utils import pad_sequences



def Discrim_net(
        # state_seq,
        seq_len,
        # action,
        n_actions: int, 
        n_features: int,
        height: int = 21, 
        width: int = 21, 
        depth: int = 11,
        hidden_dim = 64
    ):

    ## StateSeqEmb class in the original code 
    state_input = layers.Input(shape=(n_features,))
    goal_input = layers.Input(shape=(n_features,))
    state_seq_input = layers.Input(shape=(seq_len,))
    action_input = layers.Input(shape=(n_actions,))

    embed = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True)(state_seq_input)
    padded_embed = pad_sequences(embed, padding='post')
    x_rnn = layers.StackedRNNCells(
        [layers.GRU(hidden_dim),
        layers.GRU(hidden_dim),
        layers.GRU(hidden_dim)])(padded_embed)
    
    x = layers.Concatenate(axis=1)([state_input, goal_input, state_seq_input, action_input])

    ## Vanilla Value net class in the original code 
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    prob = layers.Dense(1, activation='sigmoid')(x)
    
    # logit = layers.Dense(1, activation='linear')(x)
    # model = Model([state_input, goal_input, state_seq_input], logit)
    # model.compile(optimizer=Adam(learning_rate=0.01), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    model = Model([state_input, goal_input, state_seq_input, action_input], prob)
    model.compile(optimizer=Adam(learning_rate=0.01), loss=tf.keras.losses.BinaryCrossentropy())

    return model

    


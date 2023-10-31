import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.utils import pad_sequences



def Value_net(
        n_actions: int, 
        n_features: int,
        n_space: int = 3,
        height: int = 21, 
        width: int = 21, 
        depth: int = 11,
        hidden_dim = 64
    ):
    """Creates a keras network for value net
    
    Args:
        n_actions: The dimensions of the actions (for our env is 6)
        n_features: The dimensions of the each axis of the space (for our env is 40 is the max for x and y but 41 for the masking)
        n_space: The number the axis of the space (for our env is 3)
        hidden_dim: The dimension of the embedding vectors and hidden layers of the dense layers
    
    Returns:
        A network with a embedding layer for the sequence (3 GRU cells), and 4 Dense layers. The optimizer is Adam with learning_rate=0.01
        The model input is start_input (n_space) where it is the starting points , goal_input (n_space) where it is the goal points, state_seq_input(seq_len, n_space) where it is the traj, action_input (n_space) where it is the actions
        The model input is a single value representing the state value
    """
    ## StateSeqEmb class in the original code 
    start_input = layers.Input(shape=(n_space)) ## batch, n_space(3)    
    goal_input = layers.Input(shape=(n_space)) ## batch, n_space(3)
    state_seq_input = layers.Input(shape=(None, n_space)) ## batch, seq_len, n_space(3) 3:(x,y,z) ,  `seq_len` is varying each time so `None`
    action_input = layers.Input(shape=( n_actions)) ## batch, n_actions(6)

    ## Other options are 
    # 1- embeding for each dim and then concat 
    # 2- same as now but have big hidden_dim   (current implmentation, put a big hidden dim)
    # 3- somehow make x,y,z into one value (for instance sum) and then embed for that
    embed = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True)(state_seq_input)

    ## Embed is (None, seq_len, n_space, hidden_dim) ---reshape---> (None, seq_len, n_space * hidden_dim)
    embed = layers.Reshape((-1, n_space * hidden_dim))(embed)

    # padded_embed = pad_sequences(embed, padding='post')
    x_rnn = layers.RNN(
        layers.StackedRNNCells(
        [layers.GRUCell(hidden_dim),
        layers.GRUCell(hidden_dim),
        layers.GRUCell(hidden_dim)]))(embed)
    
    x = layers.Concatenate(axis=1)([x_rnn, goal_input, start_input, action_input])

    ## Vanilla Value net class in the original code 
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    value = layers.Dense(1, activation='linear')(x)

    model = Model([start_input, goal_input, state_seq_input, action_input], value)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    return model

    

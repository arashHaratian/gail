import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras.initializers import glorot_uniform, orthogonal, random_uniform
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.utils import pad_sequences



def Discrim_net(
        n_actions: int, 
        n_features: int,
        n_space: int = 3,
        height: int = 21, 
        width: int = 21, 
        depth: int = 11,
        hidden_dim = 64,
        lr = 0.001, # 5e-5,
        seed=None
    ):

    """Creates a keras network for discrim net
    
    Args:
        n_actions: The dimensions of the actions (for our env is 6)
        n_features: The dimensions of the each axis of the space (for our env is 40 is the max for x and y but 41 for the masking)
        n_space: The number the axis of the space (for our env is 3)
        hidden_dim: The dimension of the embedding vectors and hidden layers of the dense layers
    
    Returns:
        A network with a embedding layer for the sequence (3 GRU cells), and 4 Dense layers. The optimizer is Adam with learning_rate=0.01
        The model input is start_input (n_space) where it is the starting points , goal_input (n_space) where it is the goal points, state_seq_input(seq_len, n_space) where it is the traj, action_input (n_space) where it is the actions
        The model input is a single value representing the probability of the classes
    """

## StateSeqEmb class in the original code 
    start_input = layers.Input(shape=(n_space)) ## batch, n_space(3)
    goal_input = layers.Input(shape=( n_space)) ## batch, n_space(3)
    state_seq_input = layers.Input(shape=(None, n_space)) ## batch, seq_len, n_space(3)   `seq_len` is varying each time so `None`
    action_input = layers.Input(shape=(1)) ## batch, 1  1:(a single int ranging from 0 to 5)

    ## TODO: Other options are 
    # 1- embeding for each dim and then concat 
    # 2- same as now but have big hidden_dim   (current implmentation, put a big hidden dim)
    # 3- somehow make x,y,z into one value (for instance sum) and then embed for that
    # embed = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True, embeddings_initializer = random_uniform(seed = seed))(state_seq_input)
  
    
    embed_x = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True, embeddings_initializer = random_uniform(seed = seed))(state_seq_input[:, :, 0])
    embed_y = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True, embeddings_initializer = random_uniform(seed = seed))(state_seq_input[:, :, 1])
    embed_z = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True, embeddings_initializer = random_uniform(seed = seed))(state_seq_input[:, :, 2])
    
    ## Embed is (None, seq_len, n_space, hidden_dim) ---reshape---> (None, seq_len, n_space * hidden_dim)
    # embed = layers.Reshape((-1, n_space * hidden_dim))(embed)
    embed = layers.Concatenate(axis=2)([embed_x, embed_y, embed_z])
    
    one_hot_action = layers.CategoryEncoding(n_actions, "one_hot")(action_input)

    # padded_embed = pad_sequences(embed, padding='post')

    x_rnn = layers.RNN(
        layers.StackedRNNCells(
            [layers.GRUCell(hidden_dim, kernel_initializer=glorot_uniform(seed = seed), recurrent_initializer=orthogonal(seed = seed)),
            layers.GRUCell(hidden_dim, kernel_initializer=glorot_uniform(seed = seed), recurrent_initializer=orthogonal(seed = seed)),
            layers.GRUCell(hidden_dim, kernel_initializer=glorot_uniform(seed = seed), recurrent_initializer=orthogonal(seed = seed))]))(embed)
    
    x = layers.Concatenate(axis=1)([x_rnn, start_input, goal_input, one_hot_action])

    ## Vanilla Discrim net class in the original code 
    x = layers.Dense(hidden_dim, activation='relu', kernel_initializer=glorot_uniform(seed = seed))(x)
    x = layers.Dense(hidden_dim, activation='relu', kernel_initializer=glorot_uniform(seed = seed))(x)
    x = layers.Dense(hidden_dim, activation='relu', kernel_initializer=glorot_uniform(seed = seed))(x)
    prob = layers.Dense(1, activation='sigmoid')(x)
    
    # logit = layers.Dense(1, activation='linear')(x)
    # model = Model([start_input, goal_input, state_seq_input], logit)
    # model.compile(optimizer=Adam(learning_rate=0.01), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    model = Model([start_input, goal_input, state_seq_input, action_input], prob)
    model.compile(optimizer=Adam(learning_rate = lr), loss=tf.keras.losses.BinaryCrossentropy())

    return model

    


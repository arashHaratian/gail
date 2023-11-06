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
        hidden_dim = 64,
        lr = 5e-5
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
    action_input = layers.Input(shape=(1)) ## batch, 1  1:(a single int ranging from 0 to 5)

    ## TODO: Other options are 
    ## 1- embeding for each dim and then concat 
    ## 2- same as now but have big hidden_dim   (current implmentation, put a big hidden dim)
    ## 3- somehow make x,y,z into one value (for instance sum) and then embed for that


    ## 1- embeding for each dim and then concat 
    # x_embedding_layer = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True)
    # y_embedding_layer = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True)
    # z_embedding_layer = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True)

    # embed_x = x_embedding_layer(state_seq_input[:, :, 0])
    # embed_y = y_embedding_layer(state_seq_input[:, :, 1])
    # embed_z = z_embedding_layer(state_seq_input[:, :, 2])

    # start_embed_x = x_embedding_layer(start_input[:, 0:1])
    # start_embed_y = y_embedding_layer(start_input[:, 1:2])
    # start_embed_z = z_embedding_layer(start_input[:, 2.3])

    # end_embed_x = x_embedding_layer(goal_input[:, 0:1])
    # end_embed_y = y_embedding_layer(goal_input[:, 1:2])
    # end_embed_z = z_embedding_layer(goal_input[:, 2:3])

    # embed = layers.Concatenate(axis=2)([embed_x, embed_y, embed_z])
    # start_embed = layers.Concatenate(axis=2)([start_embed_x, start_embed_y, start_embed_z])
    # end_embed = layers.Concatenate(axis=2)([end_embed_x, end_embed_y, end_embed_z])


    ## 2- same as now but have big hidden_dim   (current implmentation, put a big hidden dim)
    embedding_layer = layers.Embedding(n_features + 1, hidden_dim, mask_zero = True)
    embed = embedding_layer(state_seq_input)
    start_embed = embedding_layer(start_input)
    end_embed = embedding_layer(goal_input)
    ## Embed is (None, seq_len, n_space, hidden_dim) ---reshape---> (None, seq_len, n_space * hidden_dim)
    ## Embeds of start and end are (None, n_space, hidden_dim) ---reshape---> (None, n_space * hidden_dim)
    embed = layers.Reshape((-1, n_space * hidden_dim))(embed)
    # start_embed = layers.Reshape((n_space * hidden_dim, ))(start_embed)
    # end_embed = layers.Reshape((n_space * hidden_dim, ))(end_embed)
    start_embed = layers.Reshape((1, n_space * hidden_dim))(start_embed)
    end_embed = layers.Reshape((1, n_space * hidden_dim))(end_embed)

    
    ## TODO: embed start and the end
    
     # padded_embed = pad_sequences(embed, padding='post')
    rnn_layer = layers.RNN(
        layers.StackedRNNCells(
        [layers.GRUCell(hidden_dim),
        layers.GRUCell(hidden_dim),
        layers.GRUCell(hidden_dim)]))
    
    x_rnn_embed = rnn_layer(embed)
    x_rnn_start_embed = rnn_layer(start_embed)
    x_rnn_end_embed = rnn_layer(end_embed)

    one_hot_actions = layers.CategoryEncoding(num_tokens = n_actions, output_mode = "one_hot")(action_input)


    x = layers.Concatenate(axis=1)([x_rnn_embed, x_rnn_start_embed, x_rnn_end_embed, one_hot_actions])

    ## Vanilla Value net class in the original code 
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.Dense(hidden_dim, activation='relu')(x)
    value = layers.Dense(1, activation='linear')(x)

    model = Model([start_input, goal_input, state_seq_input, action_input], value)
    model.compile(optimizer=Adam(learning_rate = lr), loss=tf.keras.losses.MeanSquaredError())

    return model

    
